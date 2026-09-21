#!/usr/bin/env python3
"""Mamba align-mode block-size fix (prefix caching on this hybrid model).

Port of blazux/qwen3.8-Flash-DGX's patch_mamba_block_size.py (Apache-2.0,
(c) vLLM project contributors), re-expressed as a generator over
image-extracted files per this repo's discipline. Verified against OUR image
(0.1.dev20073): both anchors sit verbatim in the sources —

  v1/worker/gpu/model_states/mamba_hybrid.py:116
      (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
  v1/core/sched/scheduler.py:392
      block_size = self.cache_config.block_size
      # The last block-aligned position whose state can be cached. ...

The bug: EngineCore overwrites cache_config.block_size with the MIN group
block size (v1/engine/core.py); the align-mode state-slot seed (worker) and
the block-aligned prefill split (scheduler) then use it as if it were the
MAMBA block size (1600 here). On this build the engine-min is the attention
block (1664, chosen so attention pages >= mamba pages), so the seed is
(computed-1)//1664 instead of //1600 — every aligned boundary drifts, and
the prefill split never lands on mamba boundaries, so states are barely
cached. Consequences per blazux's diagnosis on their build (block 8/16,
worse): a prefix hit seeds the state slot out-of-row -> null block ->
ALL-ZERO restored state (silent amnesia dressed as a cache hit).

The fix: seed with cache_config.mamba_block_size (this build HAS the
attribute; vllm/config/cache.py:125), and split on the scheduler's own
self.block_size (the LCM of group blocks == the mamba block when align mode
is coherent).

Second half of blazux's pair — mamba_utils_guarded.py (the vllm#50729
state-copy race + bounds guard) — is NOT ported here: it exists because
their restore path copies whole mamba state blocks and a stale block id
crashes the CUDA context (Xid 31). Whether our image shares that exposure
is unverified; the gate-0 experiment (two-turn warm-vs-cold checksums)
decides if the race matters before we replace a 1,618-line module.

Landscape note: this fix changes the KV geometry (which blocks align), so
it MUST be A/B'd before shipping in the default profile, and a kv-offload
namespace written without it should not be trusted after it (the fence's
group layer_names don't encode alignment).

Inputs:  files/kv/orig/{mamba_hybrid.py,scheduler.py}  (extracted from image)
Outputs: files/kv/{mamba_hybrid.py,scheduler.py}       (bind-mounted overlays)
Kill switch: KV_SKIP_MAMBA_FIX=1.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "kv", "orig")
OUT = os.path.join(HERE, "kv")


def patch(name: str, edits: list[tuple[str, str]]) -> None:
    src = open(os.path.join(ORIG, name)).read()
    for old, new in edits:
        count = src.count(old)
        if count != 1:
            raise SystemExit(
                f"{name}: anchor not unique/missing (count={count}):\n{old[:300]}"
            )
        src = src.replace(old, new)
    ast.parse(src)  # fail before writing
    open(os.path.join(OUT, name), "w").write(src)
    print(f"patched kv/{name}")


if os.environ.get("KV_SKIP_MAMBA_FIX") == "1":
    print("mamba block-size fix skipped (KV_SKIP_MAMBA_FIX=1)")
    raise SystemExit(0)

os.makedirs(OUT, exist_ok=True)

patch("mamba_hybrid.py", [
    (
        "                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size\n",
        "                # [fn-kv] align-mode seed must use the MAMBA block size, not the\n"
        "                # engine-min (this hybrid: 1664 attention vs 1600 mamba).\n"
        "                (new_req_data.num_computed_tokens - 1)\n"
        "                // (self.cache_config.mamba_block_size or self.cache_config.block_size)\n",
    ),
])

patch("scheduler.py", [
    (
        # Anchor stops mid-line: the source comment continues " With Eagle, ...".
        "        block_size = self.cache_config.block_size\n"
        "        # The last block-aligned position whose state can be cached.",
        "        # [fn-kv] self.block_size is the LCM of group blocks (== the\n"
        "        # mamba block in align mode); cache_config.block_size is the\n"
        "        # engine-min (1664 here) and splits prefill off the boundary.\n"
        "        block_size = self.block_size\n"
        "        # The last block-aligned position whose state can be cached.",
    ),
])
