#!/usr/bin/env python3
"""KV-offload group classifier for the dual kit (image: offloading/config.py).

The connector's build_offloading_config hard-asserts that every KV cache
group's tokens_per_block is divisible by tokens_per_hash — the boot blocker
GLM issue #57 mode 1 records for hybrids. This flash-next build's geometry
(attention 1664 / mamba 1600 vs hash 64) IS divisible, so the assert passes
and this patch's job is legibility insurance: replace the bare AssertionError
with a classified failure that names the offending group, and print the
geometry receipt on the aligned boot. Full scratch-group EXCLUSION (GLM
PR #58's 13 scheduler touchpoints) is deliberately not ported — it only pays
for a real scratch group; if one ever appears here, that exclusion is the
follow-up, not a silent half-measure baked into this patch.

Drafter groups need nothing here: this build's offloading scheduler already
classifies EAGLE/MTP draft groups (kv_cache_config.is_eagle_group, set in
kv_cache_utils) and excludes their volatile tail itself.

Input:  files/kvoffload/orig/config.py     (extracted from the image)
Output: files/kvoffload/config_patched.py  (bind-mounted over the package)
Fail-closed: anchors verified unique, compile-checked; anchor drift aborts.
Kill switch: KV_SKIP_GROUPS_PATCH=1 (start.sh then skips both this and the
connector mounts — KV_OFFLOAD must be false in that case).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "kvoffload", "orig", "config.py")
OUT = os.path.join(HERE, "kvoffload", "config_patched.py")

if os.environ.get("KV_SKIP_GROUPS_PATCH") == "1":
    print("kv_offload groups patch skipped (KV_SKIP_GROUPS_PATCH=1)")
    raise SystemExit(0)

src = open(ORIG).read()

EDITS = [
    # 1) logger for the geometry receipt
    (
        "from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes\n",
        "from vllm.logger import init_logger\n"
        "from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes\n",
    ),
    # 2) assert -> classify
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
    # [fn-kv-offload] classify instead of assert: an unaligned group is a
    # boot error that names itself (GLM #57 mode 1 shape); an aligned hybrid
    # (this kit's case) boots with a receipt of the geometry just validated.
    _unaligned = {
        i: g.tokens_per_block
        for i, g in enumerate(groups)
        if g.tokens_per_block % tokens_per_hash
    }
    if _unaligned:
        raise ValueError(
            "[fn-kv-offload] KV groups not divisible by tokens_per_hash="
            f"{tokens_per_hash}: {sorted(_unaligned.items())} "
            "({group_idx: tokens_per_block}). No scratch-group exclusion in "
            "this build (see files/patch_kv_offload_groups.py docstring):"
            " port GLM #58's scheduler touchpoints or run without KV_OFFLOAD."
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

for old, new in EDITS:
    count = src.count(old)
    if count != 1:
        sys.exit(f"config.py: anchor count={count}, expected 1:\n{old[:200]}")
    src = src.replace(old, new)

if "from vllm.logger import init_logger" not in src:
    sys.exit("config.py: init_logger import missing after edits")

try:
    ast.parse(src)
except SyntaxError as exc:
    sys.exit(f"config.py: patched output does not parse: {exc}")

open(OUT, "w").write(src)
print("patched kvoffload/config_patched.py")
