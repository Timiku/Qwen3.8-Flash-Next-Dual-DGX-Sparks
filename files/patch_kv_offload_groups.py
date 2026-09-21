#!/usr/bin/env python3
"""KV-offload scratch-group handling for the dual kit.

Two overlays, one story. The flash-next KV pool has one group that cannot
align to the offload hash chunk: the QSA indexer ring (CircularBufferSpec,
block 8, 13 layers = 12 full-attention layers' side caches + the MTP layer's).
It holds the open group's committed keys plus un-accepted speculative rows —
volatile per-request scratch by design (see qsa_cache.py's ring comment).
The connector's build_offloading_config asserts every group's
tokens_per_block is divisible by tokens_per_hash=1664; the ring's 8 is not,
and GLM issue #57 mode 1 records the same shape for their indexer tail.

  config.py    assert -> classified warning + geometry receipt. The ring is
               named at boot instead of killing it.
  scheduler.py GLM #58's touchpoint set, ported: an is_scratch flag on
               GroupOffloadConfig, set for any group whose tokens_per_block
               is misaligned or narrower than the widest group. Scratch
               groups contribute nothing to offload stores, loads, lookups,
               or hit bookkeeping; their data stays GPU-local and is rebuilt
               per request (exactly what GPU prefix caching already does for
               the ring). The window classifier learns to tolerate the
               unknown spec class with a warning instead of asserting.

Inputs:  files/kvoffload/orig/{config,scheduler}.py  (extracted from image)
Outputs: files/kvoffload/config_patched.py, files/kvoffload/scheduler_patched.py
Fail-closed: every anchor must exist exactly once; outputs compile-checked;
             anchor drift aborts the boot with the anchor text.
Kill switch: KV_SKIP_GROUPS_PATCH=1 (start.sh then skips this and the
             overlay mounts; KV_OFFLOAD must be false in that case).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "# [fn-kv-offload]"

if os.environ.get("KV_SKIP_GROUPS_PATCH") == "1":
    print("kv_offload groups patch skipped (KV_SKIP_GROUPS_PATCH=1)")
    raise SystemExit(0)


def apply(orig: str, out: str, edits) -> None:
    src = open(orig).read()
    for old, new in edits:
        count = src.count(old)
        if count != 1:
            sys.exit(f"{os.path.basename(orig)}: anchor count={count}, expected 1:\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"{os.path.basename(orig)}: patched output does not parse: {exc}")
    open(out, "w").write(src)
    print(f"patched {os.path.basename(out)}")


# --------------------------------------------------------------------------
# config.py: build_offloading_config receipt
# --------------------------------------------------------------------------
cfg_edits = [
    # logger for the geometry receipt
    (
        "from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes\n",
        "from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes\n"
        "from vllm.logger import init_logger\n",
    ),
    # assert -> classify (receipt prints the full geometry either way)
    (
        """    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group in groups:
        assert group.tokens_per_block % tokens_per_hash == 0, (
            f"tokens_per_block={group.tokens_per_block} not divisible by "
            f"tokens_per_hash={tokens_per_hash}. "
            f"Hybrid models (e.g. Mamba+Attention) need "
            f"--enable-prefix-caching to align block sizes."
        )""",
        """    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    # [fn-kv-offload] classify instead of assert: an unaligned group is the
    # QSA indexer ring (CircularBufferSpec, block 8) - volatile per-request
    # scratch, rebuilt on hit. The offloading scheduler's is_scratch
    # touchpoints (same patch, on scheduler.py) exclude it from offload
    # scheduling; a boot-time receipt records the full geometry.
    for _i, _g in enumerate(groups):
        if _g.tokens_per_block % tokens_per_hash:
            init_logger(__name__).warning(
                "[fn-kv-offload] scratch group %d (tokens_per_block=%d,"
                " %d layers) is not divisible by tokens_per_hash=%d -"
                " excluded from offload scheduling",
                _i, _g.tokens_per_block, len(_g.layer_names), tokens_per_hash,
            )
    init_logger(__name__).info(
        "[fn-kv-offload] %d KV groups offload-aligned (tokens_per_block=%s,"
        " tokens_per_hash=%d)",
        len(groups),
        [g.tokens_per_block for g in groups],
        tokens_per_hash,
    )""",
    ),
]

# --------------------------------------------------------------------------
# scheduler.py: is_scratch touchpoints (GLM #58 port, anchors re-verified
# against this image's offloading/scheduler.py — naming differs from GLM's
# fork: kv_spec not kv_cache_spec in from_spec)
# --------------------------------------------------------------------------
SCRATCH_TEST = (
    """            if (tokens_per_block % spec.tokens_per_hash != 0
                    or tokens_per_block < max(spec.tokens_per_block)):"""
)

sch_edits = [
    # 0) lookup entry + verdict diagnostics: which link of the chain is dead?
    (
        """        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:""",
        """        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()
        logger.info(
            "[fn-kv-offload] lookup req=%s computed=%d hashes=%d"
            " lookup_groups=%d skip_read=%s jobs=%d",
            request.request_id, num_computed_tokens,
            len(request.block_hashes), len(self._lookup_groups),
            request.skip_reading_prefix_cache, len(req_status.transfer_jobs),
        )

        if req_status.transfer_jobs:""",
    ),
    # 0b) _lookup verdict: what did the group scan converge to?
    (
        """    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        complete_hit = self._lookup_complete_chunks(req_status)""",
        """    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        complete_hit = self._lookup_complete_chunks(req_status)
        logger.info(
            "[fn-kv-offload] scan req=%s complete_hit=%s partial_tail_ok=%s",
            req_status.req.request_id, complete_hit,
            self.config.supports_partial_tail,
        )""",
    ),
    # 0d) per-group scan visibility: num_chunks, keys, and per-group verdict
    (
        """                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:""",
        """                if not _FNKV_QUIET:
                    logger.info(
                        "[fn-kv-offload] group %d (tpc=%d keys=%d"
                        " window=%s eagle=%s) -> chunks=%s",
                        group_idx, tokens_per_chunk, len(offload_keys),
                        group_config.sliding_window_size_in_chunks,
                        group_config.is_eagle_group, num_hit_chunks,
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:""",
    ),
    # 0e) early-return visibility in _lookup_complete_chunks head
    (
        """                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0""",
        """                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    logger.info(
                        "[fn-kv-offload] group %d early-return: max_hit=%d"
                        " computed=%d tpc=%d keys=%d",
                        group_idx, max_hit_size_tokens, num_computed_tokens,
                        tokens_per_chunk, len(offload_keys),
                    )
                    return 0""",
    ),
    # 0c) connector return: what the scheduler actually receives
    (
        """        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)""",
        """        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        logger.info(
            "[fn-kv-offload] verdict req=%s hit=%s async=%s",
            request.request_id, num_hit_tokens, bool(num_hit_tokens),
        )
        return num_hit_tokens, bool(num_hit_tokens)""",
    ),
    # 1) window classifier: tolerate the ring's unknown spec class
    (
        """    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None""",
        """    if not isinstance(kv_cache_spec, FullAttentionSpec):
        logger.warning_once(
            "[fn-kv-offload] spec %s treated as full attention for window"
            " classification; is_scratch decides offloading",
            type(kv_cache_spec).__name__,
        )
    return None""",
    ),
    # 1b) align-mode mamba groups are full-attention-like: state blocks per
    #     chunk must survive in the block table (see the classifier comment)
    (
        """    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1""",
        """    if isinstance(kv_cache_spec, MambaSpec):
        # [fn-kv-offload] align mode caches a state per chunk; the block
        # table must keep all of them or the per-chunk states never reach
        # the store and every external hit collapses to zero (measured:
        # boundary state file missing, complete_hit=0). The single-state
        # window only describes none-mode running states.
        if getattr(kv_cache_spec, "mamba_cache_mode", "none") == "align":
            return None
        # Mamba depends on a single state
        return 1""",
    ),
    # 2) GroupOffloadConfig: the is_scratch field
    (
        """import time
from collections.abc import Iterable, Sequence""",
        """import os
import time
from collections.abc import Iterable, Sequence

_FNKV_QUIET = os.environ.get("KV_OFFLOAD_QUIET_LOOKUP")""",
    ),
    # 2) GroupOffloadConfig: the is_scratch field
    (
        """    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False""",
        """    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False
    # [fn-kv-offload] per-request scratch group (e.g. the QSA indexer ring):
    # contributes nothing to offload stores/loads/lookups; its data is
    # rebuilt per request, the same way GPU prefix caching treats it.
    is_scratch: bool = False""",
    ),
    # 3) from_spec alignment scan: skip scratch candidates
    (
        """        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)""",
        """        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
"""
        + SCRATCH_TEST
        + """
                continue  # [fn-kv-offload] scratch group
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)""",
    ),
    # 4) from_spec emit loop: emit scratch entries untouched by window logic
    (
        """        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
            sw = get_sliding_window_size_in_chunks(""",
        """        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
"""
        + SCRATCH_TEST
        + """
                # [fn-kv-offload] per-request scratch group: no stores,
                # loads, or lookups; hashes_per_chunk is never consumed
                # (update_offload_keys skips scratch groups).
                kv_group_configs_list.append(
                    GroupOffloadConfig(
                        group_idx=idx,
                        tokens_per_block=tokens_per_block,
                        tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                        hashes_per_chunk=1,
                        sliding_window_size_in_chunks=None,
                        alignment_chunk_count=None,
                        kv_event_group_spec=get_offloading_event_group_spec(
                            kv_cache_group
                        ),
                        is_eagle_group=False,
                        is_scratch=True,
                    )
                )
                continue
            sw = get_sliding_window_size_in_chunks(""",
    ),
    # 5) ctor lookup-group lists: skip scratch
    (
        """        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)""",
        """        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.is_scratch:  # [fn-kv-offload]
                continue
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)""",
    ),
    # 6) update_offload_keys: scratch never gains keys
    (
        """    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(""",
        """    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if group_config.is_scratch:  # [fn-kv-offload]
                continue
            for req_block_hash in islice(""",
    ),
    # 7) storable_chunks: zero for scratch
    (
        """        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens""",
        """        if group_config.is_scratch:  # [fn-kv-offload]
            return 0
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens""",
    ),
    # 8) update_num_hit_chunks: scratch never registers hits
    (
        """    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )""",
        """    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if group_config.is_scratch:  # [fn-kv-offload]
                continue
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )""",
    ),
    # 9) _touch: scratch never touched
    (
        """    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:""",
        """    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.is_scratch:  # [fn-kv-offload]
                continue
            if group_config.sliding_window_size_in_chunks is None:""",
    ),
    # 10) load-build group loop: zero contribution
    (
        """            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block""",
        """            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            if group_config.is_scratch:  # [fn-kv-offload]
                group_sizes.append(0)
                block_indices.append(0)
                continue

            tokens_per_block = group_config.tokens_per_block""",
    ),
    # 11) _build_store_jobs key-collection: skip scratch
    (
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )""",
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.is_scratch:  # [fn-kv-offload]
                    continue
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )""",
    ),
    # 12) _build_store_jobs src-block loop: zero contribution
    (
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )""",
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.is_scratch:  # [fn-kv-offload]
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )""",
    ),
    # 13) per-step extension diagnostics: what actually lands in the
    # connector's block_ids list per group (zeros = null placeholders).
    (
        """        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)""",
        """        assert len(new_block_id_groups) == len(self.group_states)
        for _gi, (group_state, new_blocks) in enumerate(
            zip(self.group_states, new_block_id_groups)
        ):
            if new_blocks:
                _zeros = sum(1 for _b in new_blocks if _b == 0)
                logger.info(
                    "[fn-kv-offload] ext g%d +%d zeros=%d reals=%d len=%d",
                    _gi, len(new_blocks), _zeros,
                    len(new_blocks) - _zeros, len(group_state.block_ids),
                )
            group_state.block_ids.extend(new_blocks)""",
    ),
    # 14) store-pass skip diagnostics: which chunk positions arrive as 0.
    (
        """                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue""",
        """                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        logger.info(
                            "[fn-kv-offload] store-skip g%d chunk=%d",
                            group_config.group_idx,
                            start_chunk_idx + key_idx,
                        )
                        continue""",
    ),
    # 15) load-path diagnostics: null dsts and pending blocks per group.
    (
        """            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)""",
        """            _dst_range = group_blocks[
                num_locally_computed_gpu_blocks:num_gpu_blocks
            ]
            _dst_nulls = sum(1 for _b in _dst_range if _b.block_id == 0)
            _row_reals = (
                [
                    (_j, _b.block_id)
                    for _j, _b in enumerate(group_blocks)
                    if _b.block_id != 0
                ]
                if group_config.requires_cow_source
                else None
            )
            logger.info(
                "[fn-kv-offload] loadpath g%d nlp=%d ngb=%d pend=%d"
                " null_dst=%d row_reals=%s",
                group_config.group_idx, num_locally_computed_gpu_blocks,
                num_gpu_blocks, num_pending_gpu_blocks, _dst_nulls,
                _row_reals,
            )
            dst_block_ids.extend(
                block.block_id
                for block in _dst_range
            )
            group_sizes.append(num_pending_gpu_blocks)""",
    ),
    # 16) align-mode mamba boundary scan: the row materializes state blocks
    # only at the running tail (boot24 evidence: positions 0-2 null forever,
    # 3 skips per row), so per-chunk maximal-prefix semantics can never
    # hold. The state serving a resume at boundary B lives in the file
    # stored for chunk B/tpc - 1. Find the largest such boundary at or
    # below the ceiling, leaving room for the eagle tail pop applied just
    # after this branch.
    (
        """                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )""",
        """                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    if group_config.requires_cow_source:
                        # [fn-kv-offload] boundary-only scan for align mamba
                        _top = len(offload_keys) - 1
                        num_hit_chunks = 0
                        for _i in range(_top, -1, -1):
                            _res = self.manager.lookup(
                                offload_keys[_i], req_status.req_context
                            )
                            if _res is LookupResult.MISS:
                                continue
                            if _res is LookupResult.RETRY:
                                defer_lookup = True
                                continue
                            num_hit_chunks = (
                                _i + 2 if is_eagle_unverified else _i + 1
                            )
                            break
                        logger.info(
                            "[fn-kv-offload] mamba-boundary g%d start=%d"
                            " top=%d -> R=%d",
                            group_config.group_idx, start_chunk_idx, _top,
                            num_hit_chunks,
                        )
                    else:
                        num_hit_chunks = self._maximal_prefix_lookup(
                            offload_keys,
                            req_status.req_context,
                            req_status.req,
                            group_config,
                            start_chunk_idx,
                        )""",
    ),
]

apply(
    os.path.join(HERE, "kvoffload", "orig", "config.py"),
    os.path.join(HERE, "kvoffload", "config_patched.py"),
    cfg_edits,
)
apply(
    os.path.join(HERE, "kvoffload", "orig", "scheduler.py"),
    os.path.join(HERE, "kvoffload", "scheduler_patched.py"),
    sch_edits,
)
