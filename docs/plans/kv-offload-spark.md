# KV-cache offload for the Spark kits — implementation plan (2026-09-20)

## Current state (2026-09-21, boot26)

**Restore gate GREEN.** Restore of a parked 118,144-token prefix: 4.3 s,
1.83 GB read off NVMe, generation bit-exact against the reference
(marker_text_match=true). Was 44.8-46.5 s of full re-prefill every time —
the 10× the tier exists for.

Root cause of the closed gate, proven with boot24-26 row diagnostics: the
align-mode mamba row materializes state blocks only at the running tail;
the row's first 3 positions are null placeholders forever, so the offload
store pass skips them (chunks 0,1,2 of every row never store — `ext g2
+7 zeros=3` then `store-skip g2 chunk=0/1/2`, exactly once per group).
The upstream per-chunk maximal-prefix scan semantics can therefore never
hold for mamba groups: the scan died at chunk 0 and complete_hit
collapsed to 0 for every request sharing a parked prefix.

Fix (dd3eb68): requires_cow_source groups scan boundary-only — the state
serving a resume at boundary B lives in the file stored for chunk
B/tpc - 1; search downward from the ceiling for the largest boundary
with a file, composing with the eagle tail pop (R = i+2 unverified,
i+1 verified). The engine cooperates: the align row pre-places a real
block at the hit boundary (`loadpath g2 nlp=70 pend=1 null_dst=0`), so
the load fetches exactly one boundary state per mamba group into a real
dst and the resume is bit-exact.

Decode parity battery (24 rows, connector on, full pool, GMU 0.78): no
collapse, zero NV errors, MTP accept unchanged; per-stream decode 3-9%
under the ple-nvme-78b baseline on code, 10-20% on prose — the arm's
price for the 10× restore. Remaining before wider ship: diagnostics
demotion (the ext/store-skip/loadpath lines are loud).

## Original plan

Goal: park evicted sessions' KV on NVMe so a returning session restores in
seconds instead of re-prefilling, and survives a restart. GLM lane precedent:
MiaAI-Lab/GLM-5.3 PR #58 (+ the #230/#232 stack, our `nvme_direct2`), issue
#57's six failure modes, the 09-19 six-gate qualification ladder (restore
105k tokens in 5.5 s bit-exact). NOT a prefill speeder: cold prefill on dual
measures ~10k tok/s (50k probe: 5.0 s cold, 0.9 s on the cache-hit replay),
on single ~1.6-2.5k; the tier turns a minutes-long return into a file read.

## Inventory (verified 09-20 inside the running container)

- `vllm 0.1.dev20073+g8e685d198` registers `OffloadingConnector` (plus
  LMCache*/Mooncake*/Nixl*/SimpleCPUOffload).
- `vllm/v1/kv_offload/`: `base`, `config`, `factory`, `file_mapper`, `cpu/`,
  `tiering/{fs,obj,p2p}`, `async_lookup`, `spec`, `metrics`.
- `tiering/fs` (`FileSystemTierManager`): block-per-file store
  (`<base>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin`), temp+`os.replace`
  store, `os.readv` load, `probe_o_direct`, `config.json` sidecar, keys
  through `FileMapper` (partition-by-model-digest namespace).
- `TieringOffloadingSpec` requires `cpu_bytes_to_use` (CPU primary tier) —
  on GB10 unified memory a primary CPU tier is a second claim on the same
  contested RAM; GLM's answer was the out-of-tree NVMe-direct spec
  (`spec_module_path`), ours to choose in gate 2.

## Gate 0 — the mamba block-size fix (prerequisite, standalone value)

Verified in our image: the exact line blazux's `patch_mamba_block_size.py`
asserts-and-replaces is present (`mamba_hybrid.py:116`:
`(num_computed_tokens - 1) // self.cache_config.block_size`), and so is the
scheduler anchor (`config.py:392` pair). `CacheConfig.mamba_block_size`
exists in this build. The fix makes align-mode prefix hits seed the state
slot on the mamba boundary instead of the engine-min block, and aligns the
prefill split the same way.
Their lane documents the failure concretely ("prefix hit seeds at
(num_computed-1)//16 -> null block -> **all-zero restored state**"; a
vllm#50729-adjacent wild-address copy observed as Xid 31 on GB10 — the second
file, `mamba_utils_guarded.py`, carries the race fix + bounds guard).
Note our boot log shows attention block 1664 (page-size alignment), so our
engine-min may be 1664, not 16/8 as theirs names; the bug's SHAPE (wrong
boundary divisor) survives either way.
**Verification experiment (must run before believing anything downstream):**
identical multi-turn prompt served with prefix caching warm vs every turn
salted (cold). Bit-identical outputs = fine; drift/hallucinated recall = the
zero-state bug is live on our kits today. This experiment gates the whole
plan and is worth running regardless.
Port shape: a generator `files/patch_mamba_block_size.py` (house pattern),
Apache-2.0-licensed source — attribution header, no AGPL contamination.

## Gate 1 — boot blocker in the connector config

`kv_connector/v1/offloading/config.py:60`:
`assert group.tokens_per_block % tokens_per_hash == 0` ("Hybrid models need
--enable-prefix-caching to align block sizes"). Flash-next is exactly the
hybrid case (attention 1664 vs mamba 1600 vs QSA ring — expect the assert to
fire; #57 failure mode #1). GLM's #58 replaces the assert with a
scratch-group classification (drafter by layer name `mtp`/`dflash`/
`drafter`, root-prefix minority rule). For our kits: same classifier pattern,
minus kpool specifics; port from the `kvoffload-lab` template
(`~/glm53-lab/overlay/patch_kv_offload_groups.py` on Spark1).

## Gate 2 — tier topology (the decision)

- **(a) In-tree minimum:** `OffloadingConnector` +
  `TieringOffloadingSpec`, tiny primary (`cpu_bytes_to_use ~ 2 GiB`) +
  `fs` secondary on each node's own NVMe. Zero new spec code; restore is
  disk->CPU->GPU two-hop; the small arena keeps the pinned-RAM claim flat.
  First boot; it answers "does the whole path work on this model" cheaply.
- **(b) NVMe-direct:** port GLM's `nvme_direct2` pattern against our
  `FileMapper` fence (their lesson: don't hand-roll hash trees — partition by
  `<safe_model_name>_<digest[:16]>/r<rank>`, revision from the launcher;
  `model.dtype` is NOT a valid fence field on these forks). Bump it to the
  default if (a) shows the bounce cost or RAM churn.
- Never on the NFS share: the worker's `ple_cache` precedent — per-node
  roots, rsync'd nothing; each rank stores its own shard files (block hashes
  carry group/layout; the GLM ladder verified both nodes' trees).

## Gate 3 — the qualification ladder (GLM's six, adapted)

1. boot: `--kv-transfer-config` parses, server ready (after gates 0/1).
2. store receipt: `kv_offload_*` bytes metrics move on a >pool-context run.
3. flood: pool/2+1 × long sessions evict; both nodes' trees get files.
4. restore bit-exact vs cold (needs gate 0 verified; a green-without-0 test
   is the zero-state bug wearing a party hat).
5. decode parity under load with the connector on (battery, 24 rows).
6. cross-boot: restart, old session continues from disk (the #232 feature).
7. flash-next-specific: MTP interaction — the drafter group must be excluded
   (classify, don't store draft KV).

## Payoff, honestly scaled

Dual: the 7.1M pool already parks ~14×500K sessions; the tier's win is beyond
that and across boots. Single: ~1.43M pool = ~3×500K parked, and a 500K
return costs minutes of re-prefill at their prefill rate — the tier matters
more there, same as PLE offload's asymmetry. Decision trigger: any workload
rotation > parked capacity. Until then this stays a plan.

## Related

wiki: `wiki/2x-dgx-spark/qwen3.8-flash-next/ple-nvme-offload.md` ("Future
work" section), `wiki/2x3090/qwen3.8-flash-next/ple-ssd-offload.md`, GLM
lane's `glm53-kv-offload-disk` / `kv-offload-implementation-cost` /
`glm53-nvme-pr-stack-230-232` pages.
