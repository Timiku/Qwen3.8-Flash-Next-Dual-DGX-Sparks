# SPDX-License-Identifier: Apache-2.0
"""
NvmeDirectOffloadingSpec2: NVMe-direct KV offload with a model-keyed namespace.

GPU<->file streaming with NO resident CPU cache tier (GB10 unified memory has
none to spare - see repo issue #57): each IO thread holds one chunk-sized
pinned bounce buffer, D2H/H2D are event-fenced on private streams, stores are
tmp + fsync + atomic rename, loads verify byte counts and raise on shortfall.

The storage namespace (the fence):

    <root_dir>/<safe_model_name>_<digest[:16]>/r<rank>/<hash[:3]>/<hex>_g<g>.bin

`digest` is sha256 over a canonical field map (_fence_fields) covering
everything that changes the *meaning* of stored KV bytes: model id, resolved
weights revision, per-block KV bytes, hash/chunk geometry, parallel sizes, per-group
block sizes + layer names, engine version, recipe stamp, layout format. The
same map lands in config.json beside the data and is READ BACK at init: a
pre-existing directory whose sidecar disagrees is a boot error (fail closed);
data files without a sidecar are a boot error; a boot that resolves no
revision lands in the isolated `unresolved` namespace - cold cache, never a
merge into someone else's cache.

This is upstream vllm.v1.kv_offload.file_mapper.FileMapper's idea (its
config.json is written but never read back by the tiering managers) plus its
missing half: the weights revision and the read-back check.

Cross-boot key stability additionally requires PYTHONHASHSEED=0 (the block-hash
chain root NONE_HASH is seed-random otherwise) and expandable_segments off -
both enforced by the launcher block that selects this spec.

extra_config keys: root_dir (required, absolute), model_revision (optional),
capacity_bytes (optional, default 500 GiB), n_io_threads (optional, default 4).
Selected via kv_connector_extra_config spec_name/spec_module_path - the
kv_offload factory honors spec_module_path; the connector factory does not.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections.abc import Collection, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TransferResult,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.config import OffloadingConfig

# Under the vllm logger hierarchy so the engine's log config (level,
# handlers) reaches this module; a bare "kvoffload.*" logger would drop
# INFO lines into a hierarchy nobody configures.
logger = init_logger("vllm.kvoffload.nvme_direct2")

FORMAT_TAG = "nvme-direct/fn-v1"
SIDE_CAR = "config.json"
FENCE_VERSION = 1
_GSUF = re.compile(r"_g(\d+)\.bin$")


# --------------------------------------------------------------------------
# fence
# --------------------------------------------------------------------------
def _safe_model_name(name: str) -> str:
    return name.replace("/", "_")


def _env_or_none(key: str) -> str | None:
    v = os.environ.get(key)
    v = v.strip() if v else ""
    return v or None


def resolve_model_revision(model_name: str, override: str | None) -> str:
    """Weights identity for the fence. Order: explicit config, launcher env,
    offline HF-cache resolution, else the isolated sentinel 'unresolved'."""
    for candidate in (override, _env_or_none("MODEL_REVISION")):
        if candidate:
            return candidate
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(model_name, "config.json")
        if isinstance(p, str) and "/snapshots/" in p:
            return p.split("/snapshots/")[1].split("/")[0]
    except Exception as exc:  # offline, unparseable id, hub absent
        logger.warning("nvme2: revision resolution failed: %s", exc)
    return "unresolved"


def _fence_fields(config: OffloadingConfig, revision: str) -> dict[str, Any]:
    try:
        import vllm

        engine = f"vllm/{getattr(vllm, '__version__', 'unknown')}"
    except Exception:
        engine = "vllm/unknown"
    cache, par = config.cache, config.parallel
    # model.dtype is deliberately NOT in the fence: it reports the model's
    # PARAMS dtype and is process-dependent on this fork (fp8_ds_mla on the
    # EngineCore, fp8 elsewhere — measured, two namespaces in one boot).
    # What the stored bytes mean is pinned by worker_kv_bytes_per_block,
    # derived from the broadcast kv_cache_config: it changes with the KV
    # cache dtype and geometry, identically in every process.
    # Serving a local snapshot makes model.name the whole snapshots/<hash>
    # path; a model_name in the config (launcher passes $MODEL) keeps the
    # namespace and sidecar readable and topology-independent.
    name = str(config.extra_config.get("model_name") or config.model.name)
    groups = [
        {
            "tokens_per_block": int(g.tokens_per_block),
            "layer_names": sorted(g.layer_names) if g.layer_names else [],
        }
        for g in config.groups
    ]
    return {
        "v": FENCE_VERSION,
        "model": name,
        "revision": revision,
        "kv_bytes_per_block": int(config.worker_kv_bytes_per_block),
        "tokens_per_hash": int(cache.tokens_per_hash),
        "blocks_per_file": 1,  # enforced; fixed constant, not config-dependent
        "tp_size": int(par.tp_size),
        "pp_size": int(par.pp_size),
        "pcp_size": int(par.pcp_size),
        "dcp_size": int(par.dcp_size),
        "groups": groups,
        # State semantics differ between align (state per chunk, restorable)
        # and none (one running state per request): a mode change must land
        # in its own namespace or restores read meaningless state bytes.
        "mamba_cache_mode": str(
            config.extra_config.get("mamba_cache_mode") or "none"
        ),
        "engine": engine,
        "stamp": _env_or_none("FN_KV_RECIPE_STAMP") or "unknown",
        "format": FORMAT_TAG,
    }


def _fence_digest(fields: dict[str, Any]) -> str:
    canon = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class Fence:
    """Namespace + sidecar for one boot. Called once per process by the spec
    ctor (scheduler process: rank 0; each worker process: its own rank)."""

    def __init__(self, root_dir: str, fields: dict[str, Any]):
        self.fields = fields
        self.digest = _fence_digest(fields)
        self.base_dir = os.path.join(
            root_dir,
            f"{_safe_model_name(fields['model'])}_{self.digest[:16]}",
        )
        self.sidecar_path = os.path.join(self.base_dir, SIDE_CAR)

    def adopt_or_create(self, who: str, rank: int | None = None) -> None:
        """Fail-closed sidecar check; create atomically when absent. `rank`
        narrows the data probe to this boot's own subtree (r<rank>/), so a
        shared root never cross-triggers on another rank's tree."""
        os.makedirs(self.base_dir, exist_ok=True)
        if os.path.exists(self.sidecar_path):
            try:
                with open(self.sidecar_path) as f:
                    on_disk = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"nvme2 {who}: unreadable fence sidecar "
                    f"{self.sidecar_path}: {exc}"
                ) from exc
            if on_disk.get("digest") != self.digest:
                raise RuntimeError(
                    f"nvme2 {who}: fence mismatch at {self.base_dir} - the "
                    f"cache was written by a different model/config (on disk: "
                    f"model={on_disk.get('model')!r} "
                    f"revision={on_disk.get('revision')!r}; this boot: "
                    f"model={self.fields['model']!r} "
                    f"revision={self.fields['revision']!r}). Refusing to mix "
                    "namespaces."
                )
            return
        # No sidecar: a tree that already holds rank data is of unknown
        # provenance.
        probe = (self.base_dir if rank is None
                 else os.path.join(self.base_dir, f"r{rank}"))
        if os.path.isdir(probe) and any(fs for _, _, fs in os.walk(probe)):
            raise RuntimeError(
                f"nvme2 {who}: {probe} holds data but {self.base_dir} has no "
                f"{SIDE_CAR}; refusing a cache with unknown provenance. "
                "Move it aside."
            )
        tmp = f"{self.sidecar_path}.tmp.{os.getpid()}"
        payload = dict(self.fields)
        payload["digest"] = self.digest
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, self.sidecar_path)  # atomic create-or-exists
        except FileExistsError:  # racing twin process; theirs is equivalent
            pass
        finally:
            os.unlink(tmp)


def _relpath(key: OffloadKey) -> str:
    h = get_offload_block_hash(key).hex()
    return os.path.join(h[:3], f"{h}_g{get_offload_group_idx(key)}.bin")


# --------------------------------------------------------------------------
# scheduler side
# --------------------------------------------------------------------------
class NvmeFileLoadStoreSpec(LoadStoreSpec):
    """Relative file names, positionally aligned with the job's block_ids
    (the LoadStoreSpec contract the CPU worker relies on)."""

    def __init__(self, relpaths: list[str]):
        self.relpaths = relpaths

    def __repr__(self) -> str:
        return f"NvmeFileLoadStoreSpec({len(self.relpaths)} files)"


class NvmeDirectManager2(OffloadingManager):
    """File-existence manager; no eviction (disk is the last tier; an
    external TTL sweeper owns cleanup)."""

    def __init__(self, rank_dir: str):
        self.medium = Medium.STORAGE
        self._rank_dir = rank_dir
        os.makedirs(rank_dir, exist_ok=True)
        self._pending_stores: set[OffloadKey] = set()
        self._exists: set[OffloadKey] = set()

    def _path(self, key: OffloadKey) -> str:
        return os.path.join(self._rank_dir, _relpath(key))

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._pending_stores:
            return LookupResult.HIT_PENDING
        if key in self._exists:
            # Recheck the filesystem so an external TTL sweeper cannot leave
            # a stale positive (fixes #232 review finding 1).
            if os.path.exists(self._path(key)):
                return LookupResult.HIT
            self._exists.discard(key)
            return LookupResult.MISS
        hit = os.path.exists(self._path(key))
        # [fn-kv-offload] diagnosis: every cold lookup names itself. The
        # volume is one line per 1664-token chunk; a session restores in
        # tens of lines. Demote via KV_OFFLOAD_QUIET_LOOKUP=1 if noisy.
        if not os.environ.get("KV_OFFLOAD_QUIET_LOOKUP"):
            logger.info(
                "nvme2 lookup %s -> %s", _relpath(key),
                "HIT" if hit else "miss",
            )
        if hit:
            self._exists.add(key)
            return LookupResult.HIT
        return LookupResult.MISS

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        keys_to_store = []
        for k in keys:
            if k in self._pending_stores or k in self._exists:
                continue
            if os.path.exists(self._path(k)):
                self._exists.add(k)
                continue
            keys_to_store.append(k)
        if not keys_to_store:
            return None
        self._pending_stores.update(keys_to_store)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=NvmeFileLoadStoreSpec(
                [_relpath(k) for k in keys_to_store]
            ),
            evicted_keys=[],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        self._pending_stores.difference_update(keys)
        if success:
            self._exists.update(keys)
        else:
            for k in keys:
                self._exists.discard(k)

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        # Callers pass keys already resolved to HIT; files are immutable and
        # nothing here is evictable, so there is nothing to protect - just
        # hand back the paths in key order.
        if not os.environ.get("KV_OFFLOAD_QUIET_LOOKUP"):
            logger.info("nvme2 prepare_load: %d keys", len(keys))
        return NvmeFileLoadStoreSpec([_relpath(k) for k in keys])

    @override
    def reset_cache(self) -> None:
        self._pending_stores.clear()
        self._exists.clear()


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------
def _write_all(fd: int, data: memoryview) -> None:
    """Loop os.write so a short write cannot silently truncate a KV file
    (fixes #232 review finding 2)."""
    off = 0
    while off < len(data):
        off += os.write(fd, data[off:])


class NvmeDirectWorker2(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        rank_dir: str,
        n_io_threads: int,
    ):
        self._dir = rank_dir
        os.makedirs(rank_dir, exist_ok=True)
        self._tensors = [t.tensor for t in kv_caches.tensors]
        self._group_refs = kv_caches.group_data_refs
        self._group_bytes = [
            sum(ref.page_size_bytes for ref in refs) if refs else 0
            for refs in self._group_refs
        ]
        self._max_chunk_bytes = max(self._group_bytes, default=0)
        self._tls = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=n_io_threads, thread_name_prefix="vllm_kv_nvme2"
        )
        # job_id -> (future, num_bytes, is_load)
        self._jobs: dict[int, tuple[Future, int, bool]] = {}
        logger.info(
            "NvmeDirectWorker2 dir=%s tensors=%d threads=%d bounce=%.1f MiB "
            "groups=%d",
            rank_dir,
            len(self._tensors),
            n_io_threads,
            self._max_chunk_bytes / (1 << 20),
            len(self._group_refs),
        )

    def _thread_state(self):
        st = getattr(self._tls, "state", None)
        if st is None:
            # One max-chunk pinned bounce per IO thread. A job row is ONE
            # block of ONE group (<= max_chunk_bytes); the group's refs write
            # disjoint [0, group_bytes) ranges of it.
            buf = torch.empty(
                self._max_chunk_bytes, dtype=torch.uint8, pin_memory=True
            )
            st = (buf, buf.numpy(), torch.cuda.Stream())
            self._tls.state = st
        return st

    def _plan(
        self, gpu_spec: GPULoadStoreSpec, relpaths: Sequence[str]
    ) -> tuple[list[tuple[str, int, int]], int]:
        """(abspath, group_idx, block_id) rows + total bytes. spec entries
        pair with block_ids positionally; the group index is read back from
        the file name this spec generated from the key, so the walk is
        correct independent of any assumed group ordering. group_sizes is
        cross-checked when present."""
        block_ids = gpu_spec.block_ids
        if len(relpaths) != len(block_ids):
            raise ValueError(
                f"{len(relpaths)} files vs {len(block_ids)} blocks"
            )
        gsz = gpu_spec.group_sizes
        if gsz is not None and sum(gsz) != len(block_ids):
            raise ValueError(f"group_sizes {gsz} vs {len(block_ids)} blocks")
        rows: list[tuple[str, int, int]] = []
        total = 0
        for path, b in zip(relpaths, block_ids):
            m = _GSUF.search(path)
            if m is None:
                raise ValueError(f"path missing group suffix: {path}")
            g = int(m.group(1))
            if g >= len(self._group_bytes):
                raise ValueError(f"group {g} out of range in {path}")
            rows.append((os.path.join(self._dir, path), g, int(b)))
            total += self._group_bytes[g]
        return rows, total

    def _store_task(self, event: torch.cuda.Event, rows) -> tuple[int, float]:
        buf, buf_np, stream = self._thread_state()
        stream.wait_event(event)
        t0 = time.monotonic()
        written = 0
        for path, g_idx, b in rows:
            off = 0
            with torch.cuda.stream(stream):
                for ref in self._group_refs[g_idx]:
                    n = ref.page_size_bytes
                    buf[off : off + n].copy_(
                        self._tensors[ref.tensor_idx][b, :n].view(torch.uint8),
                        non_blocking=True,
                    )
                    off += n
            stream.synchronize()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                _write_all(fd, buf_np[:off].data)
                os.fsync(fd)
                # DONTNEED only after fsync: the pages are clean by then, and
                # on GB10's 64 KiB pages an earlier eviction could drop a
                # dirty partial tail (#232 review finding 3).
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            except BaseException:
                os.close(fd)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            os.close(fd)
            os.replace(tmp, path)
            written += off
        return written, time.monotonic() - t0

    def _load_task(self, event: torch.cuda.Event, rows) -> tuple[int, float]:
        buf, buf_np, stream = self._thread_state()
        stream.wait_event(event)
        t0 = time.monotonic()
        read = 0
        for path, g_idx, b in rows:
            expected = self._group_bytes[g_idx]
            fd = os.open(path, os.O_RDONLY)
            try:
                got = os.readv(fd, [buf_np[:expected].data])
                if got != expected:
                    raise OSError(f"short read {got}/{expected} on {path}")
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            finally:
                os.close(fd)
            off = 0
            with torch.cuda.stream(stream):
                for ref in self._group_refs[g_idx]:
                    n = ref.page_size_bytes
                    self._tensors[ref.tensor_idx][b, :n].view(
                        torch.uint8
                    ).copy_(buf[off : off + n], non_blocking=True)
                    off += n
            stream.synchronize()
            read += off
        return read, time.monotonic() - t0

    def _submit(
        self,
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        file_spec: NvmeFileLoadStoreSpec,
        is_load: bool,
    ) -> bool:
        rows, total = self._plan(gpu_spec, file_spec.relpaths)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        task = self._load_task if is_load else self._store_task
        fut = self._pool.submit(task, event, rows)
        self._jobs[job_id] = (fut, total, is_load)
        return True

    @override
    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        assert isinstance(dst_spec, NvmeFileLoadStoreSpec)
        return self._submit(job_id, src_spec, dst_spec, is_load=False)

    @override
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        assert isinstance(src_spec, NvmeFileLoadStoreSpec)
        return self._submit(job_id, dst_spec, src_spec, is_load=True)

    @override
    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for jid in [j for j, (f, _, _) in self._jobs.items() if f.done()]:
            fut, _total, is_load = self._jobs.pop(jid)
            exc = fut.exception()
            total, elapsed = 0, 0.0
            if exc is not None:
                if is_load:
                    # The scheduler already committed these blocks to a
                    # restore; a missing/short file is unrecoverable.
                    logger.error("nvme2 KV load job %d failed: %s", jid, exc)
                    raise exc
                # Store failures keep the connector contract (submit's bool
                # is asserted, result.success is asserted): the file does not
                # exist, lookup rechecks the fs, the key decays to a clean
                # MISS.
                logger.warning(
                    "nvme2 KV store job %d failed (key stays MISS): %s",
                    jid,
                    exc,
                )
            else:
                total, elapsed = fut.result()
            # The connector records transfer metrics only when BOTH size and
            # time are present (worker.py) - transfer_time=None would leave
            # the load/store byte counters dead.
            results.append(
                TransferResult(
                    job_id=jid, success=True, transfer_size=total,
                    transfer_time=elapsed,
                )
            )
        return results

    @override
    def wait(self, job_ids: set[int]) -> None:
        futs = [self._jobs[j][0] for j in job_ids if j in self._jobs]
        if futs:
            futures_wait(futs)

    @override
    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------
class NvmeDirectOffloadingSpec2(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        root_dir = self.extra_config.get("root_dir")
        if not root_dir or not os.path.isabs(str(root_dir)):
            raise ValueError(
                "NvmeDirectOffloadingSpec2 requires an absolute root_dir in "
                "kv_connector_extra_config"
            )
        self.root_dir = str(root_dir)
        self.n_io_threads = int(self.extra_config.get("n_io_threads", 4))
        if int(self.blocks_per_chunk) != 1:
            raise ValueError(
                "NvmeDirectOffloadingSpec2 requires blocks_per_chunk == 1"
            )
        revision = resolve_model_revision(
            str(self.extra_config.get("model_name") or config.model.name),
            self.extra_config.get("model_revision"),
        )
        if revision == "unresolved":
            logger.warning(
                "nvme2: model revision unresolved - using an isolated cold "
                "namespace (correct, but no cross-boot reuse). Set "
                "MODEL_REVISION to get persistence."
            )
        self.fence = Fence(self.root_dir, _fence_fields(config, revision))
        self.fence.adopt_or_create(
            f"rank={config.parallel.rank}", config.parallel.rank
        )
        usage = shutil.disk_usage(self.root_dir)
        self.capacity_bytes = int(
            self.extra_config.get("capacity_bytes", 500 * (1 << 30))
        )
        if usage.free < self.capacity_bytes:
            raise RuntimeError(
                f"nvme2: {self.root_dir} has {usage.free / (1 << 30):.1f} "
                f"GiB free; {self.capacity_bytes / (1 << 30):.1f} GiB "
                "required - wrong volume or lower the budget"
            )
        logger.info(
            "nvme2 fence: base=%s digest=%s free=%.1f GiB revision=%s",
            self.fence.base_dir,
            self.fence.digest[:16],
            usage.free / (1 << 30),
            revision[:12],
        )
        self._manager: NvmeDirectManager2 | None = None
        self._worker: NvmeDirectWorker2 | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            # TP replicas store symmetric chunk sets (each rank holds a
            # slice of every block); rank 0's tree is the scheduler's
            # lookup authority.
            self._manager = NvmeDirectManager2(
                os.path.join(self.fence.base_dir, "r0")
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = NvmeDirectWorker2(
                kv_caches=kv_caches,
                rank_dir=os.path.join(
                    self.fence.base_dir, f"r{self.config.parallel.rank}"
                ),
                n_io_threads=self.n_io_threads,
            )
        return self._worker

    @classmethod
    @override
    def build_metric_definitions(cls, extra_config):
        return {}
