# PLE NVMe-offload on the dual kit (2026-09-20)

Ported from the single-Spark kit: the 26.82 GiB NVFP4 PLE n-gram table leaves
GPU memory and lives as a packed file (`*.packed_u8`) that each node's
`PleOffloadWorker` process serves through `np.memmap` — the kernel page cache
decides residency, so the table's home is the node's NVMe and only the touched
pages cost RAM (evictable, `MADV_RANDOM`, batched `posix_fadvise(WILLNEED)`
prefetch in the gather). Boot: step 6c of `start.sh`; the table is built once
(`build_ple_packed_table.py`, ~40 s) and pushed to the worker once.

## The multi-node shape (this is the port, not the copy)

Stock vLLM refuses `VLLM_PLE_CPU_OFFLOAD` with `nnodes > 1`
(`_validate_ple_offload_config`), and the refusal is right for the stock
topology: the offload process is spawned once, by global rank 0, and the GPU
ranks register with it over CUDA-IPC handles + a node-local ZMQ IPC socket —
none of which cross a node boundary. With TP2 = one rank per Spark, rank 1
would be waiting for a CPU worker that can only ever exist on the other
machine.

The fix is **one `PleOffloadWorker` per node**, each serving its node's local
ranks (patched by `files/patch_ple_offload.py`, overlays `gpu_worker.py` +
`worker.py` + `connector.py`; formulas, no new env — single-node behaviour is
bit-identical because `rank % per_node` is then just "the node's TP0"):

- spawn gate: `local_rank == 0` and `rank % max(1, world//nnodes) == 0`;
  per-node registration count `(dp*tp)//nnodes` (= 1 on this kit);
- the connector's request/pin gates move from `tp_rank == 0` to a
  `_node_leader` test — on the worker node, TP1 *is* the node leader and
  stages into its own buffers;
- the worker's input-buffer pick prefers TP0's registration and falls back to
  the node's first (identical content; inputs are TP-replicated);
- registrations ride each node's own uuid ipc socket, and the CUDA-IPC output
  buffer + done-flag pair stay node-local, which is what the host-side
  handshake needs on GB10 anyway.

The table file must exist on **both** NVMe drives (each node's worker mmaps
its own); `start.sh` rsyncs it once to `$REMOTE_HOME/.cache/vllm/ple_cache/`.

Upstream context (checked 09-20): `vllm-project/vllm#53899` is the CPU-offload
PR this kit's image carries; `#54070` is an open draft adding exactly this
disk-backed placement (`VLLM_PLE_DISK_OFFLOAD_DIR`, same page-cache mechanism,
write-through first boot); `#54129` is the competing no-worker mmap-from-
safetensors path (forces PIECEWISE graphs; ~2x decode tax measured at TP2/PP4,
≈0 at TP1). Neither merged; our overlay ships the #54070 design earlier and
adds the fadvise prefetch. Neither kit repo (MiaAI-Lab single/dual) has PRs.

## Result — KV pool

| config | KV pool | Δ vs off |
|---|---|---|
| `PLE_OFFLOAD=false` (table in both ranks' VRAM) | 5,598,751 tok | — |
| `PLE_OFFLOAD=true`, NVMe, GMU 0.835 | **7,096,926 tok** | **+26.8%** (max-concurrency 21.4x → 27.1x @262K) |
| `PLE_OFFLOAD=true`, NVMe, GMU 0.78 (default) | 6,284,133 tok | +12.2% (23.97x @262K) |

Per node: `PleOffloadWorker` spawns (`num_workers=1`), attaches
`mmap table attached (320001536 rows x 90 B = 26.82 GiB) [MADV_RANDOM]`,
registers its local rank, `Registrations complete`. First-touch cold (probe 1:
28 tok/s) warms to steady within a few requests (71.5 tok/s single-stream).

## Result — decode throughput (sparkDash battery, 600 tok, 3 reps, `nvrm=0`)

Same harness/framing as `port-ab-20260920.md`; base row = the closed
`port` column there (PLE offload off, pool 5.6M). Two on-arms:
`ple-nvme` (GMU 0.835, pool 7.10M) and `ple-nvme-78b` (GMU 0.78, pool 6.28M).

| cell | port (off) | 0.835 arm | Δ | 0.78 arm | Δ |
|---|---|---|---|---|---|
| prose S=1 | 66.6 | 61.4 | -7.8% | 60.4 | -9.3% |
| code  S=1 | 83.1 | 77.4 | -6.9% | 79.3 | -4.6% |
| prose S=2 | 58.1 | 47.6 | -18.1% | 49.5 | -14.8% |
| code  S=2 | 73.1 | 65.0 | -11.1% | 67.8 | -7.2% |
| prose S=4 | 44.1 | 40.2 | -8.8% | 42.1 | -4.5% |
| code  S=4 | 59.9 | 54.0 | -9.8% | 56.6 | -5.5% |
| prose S=8 | 35.4 | 31.9 | -9.9% | 32.5 | -8.2% |
| code  S=8 | 48.9 | 43.0 | -12.1% | 45.4 | -7.2% |

The 0.835 arm ran with host `avail` pinned ~7.1 GiB throughout — the page
cache had no headroom over the 26.82 GiB table and gathers kept faulting. The
GMU-0.78 arm gives the cache ~11.1 GiB and cuts the mean penalty from -10.3%
to -6.4% across the eight cells. What remains is the offload mechanism itself:
per-position gather + host handshake on every decode step, with the table hot
in cache. The rep-by-rep rows show no cold/warm split (e.g. prose S1 at 0.78:
58.6/59.6/63.1), so this is steady-state cost, not faulting — and it matches
upstream `#54070`'s own numbers (-8% @c1, -17% @c32) in both shape and size.

## Placement verified (the NFS question, closed)

- Both nodes run their own `PleOffloadWorker` and mmap their own local
  `packed_u8` (head pid 468 / worker pid 361, both `/dev/nvme0n1p2` ext4,
  RssFile 1.6/1.68 GiB, lifetime majflt 114k/117k — symmetric, so the worker
  faults against its own drive).
- The only NFS mount in the serving path is the worker's **read-only HF
  weights volume** (`10.100.72.1:/`, streamed at load); `ple_cache` is a
  host bind of each node's own `~/.cache/vllm`. Three 50k-token requests
  added 0/3 major faults — table pages stay hot in both caches at rest.
- Conclusion: the slowdown is local page-cache behavior + gather overhead,
  **not** cross-node fetch.

## Memory posture

- GPU side: the table is off both cards (dummy placeholder + per-step gather);
  the ~13.4 GiB/rank it occupied now feeds the KV pool (the +1.5M above).
- Host side (per node): offload worker RSS stays tiny (anon ~0.3 GiB); the
  table is file-backed page cache — reclaimable under pressure, not pinned.
  Unified GB10 memory charges GPU weights+KV to `used` (~114 GiB at this
  pool); that is the same GPU-resident charge as the PLE-off boot minus the
  table.
- Both nodes serve the same table contents; NVMe reads are per-node (own
  drive, no cross-fabric reads).

## Verdict / default

Ship `PLE_OFFLOAD=true` with the kit's measured GMU default **0.835**: the
trade is +26.8% KV pool for ~10% decode, which is the whole point of the
offload. Operators whose workload is decode-rate-bound and runs near-full
context can drop to `GPU_MEMORY_UTILIZATION=0.78` (pool 6.28M, still +12.2%
over off, penalty ~6.4%). This box's `.env` currently sits at 0.78 from the
arm test; restored to 0.835 as the final default (2026-09-20).

`logs/sweep.jsonl` tags: `ple-nvme` (0.835 arm), `ple-nvme-78` (contaminated,
superseded), `ple-nvme-78b` (0.78 clean arm).
