#!/usr/bin/env bash
# ============================================================================
# start.sh — Serve nvidia/Qwen3.8-Flash-Next-NVFP4 across a 2-node
#             DGX Spark cluster with vLLM TP2+EP+MTP3.
#
# Based on: https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark
#
# Weight distribution (default: rsync). Each node keeps its own copy of the
# checkpoint in ~/.cache/huggingface; the worker copy is rsync'd from the head
# once and reused. Set NFS_SHARE=true (or pass --nfs) to instead export the head
# cache over NFS on the ConnectX link, so the worker keeps no local copy — see
# "NFS weight sharing (optional)" in the README for the trade-offs.
#
# Usage:
#   ./download.sh             # fetch NVFP4 onto the head (optional; start.sh can too)
#   ./start.sh                # download on head if needed → sync to worker → patch → launch
#   ./start.sh --no-download  # skip download (weights already cached on head)
#   ./start.sh --no-launch    # download + sync only, don't start server
#   ./start.sh --launch       # skip download/sync; apply patch + launch
#   ./start.sh --nfs          # distribute weights over NFS instead of rsync
#   ./start.sh --no-nfs       # force rsync distribution (overrides NFS_SHARE=true)
#   ABLIT=1 ./start.sh        # gated Keys house QSA L3-47 checkpoint (download first)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# ---------------------------------------------------------------------------
# Load .env
# Environment wins over .env for ABLIT / HF_TOKEN (same 0/1 pattern as
# MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark). Other knobs still follow
# ".env beats the environment".
# ---------------------------------------------------------------------------
_CLI_ABLIT="${ABLIT:-}"
_CLI_HF_TOKEN="${HF_TOKEN:-}"
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    echo "  cp .env.sample .env"
    exit 1
fi


# Knobs that are NOT read through an explicit _CLI_ variable above still have
# to honour "environment > .env": sourcing .env would otherwise overwrite them.
# Snapshot anything set in the environment, then restore it after the source
# (same mechanism the single-Spark kit uses). Only the modern knobs are listed;
# the pre-existing ones keep this kit's documented ".env beats the environment"
# precedence.
_ENV_SNAPSHOT_VARS=(CUDAGRAPH_MODE COMPILATION_MODE CUDAGRAPH_CAPTURE_SIZES
                    MTP_K_SCHEDULE CHAT_TEMPLATE READY_TIMEOUT_S)
for _v in "${_ENV_SNAPSHOT_VARS[@]}"; do
    eval "_SNAP_$_v=\${$_v-}"
    eval "_SNAPSET_$_v=\${$_v+set}"
done
# shellcheck source=.env
source .env

for _v in "${_ENV_SNAPSHOT_VARS[@]}"; do
    if [[ -n "$(eval "printf %s \"\${_SNAPSET_$_v-}\"")" ]]; then
        eval "$_v=\$_SNAP_$_v"
    fi
done

[[ -n "$_CLI_ABLIT" ]] && ABLIT="$_CLI_ABLIT"
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" == "0" || "$ABLIT" == "1" ]] || err "ABLIT must be 0 or 1 (got: '$ABLIT')"
[[ -n "$_CLI_HF_TOKEN" ]] && HF_TOKEN="$_CLI_HF_TOKEN"
HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN
ABLIT_MODEL_ID="drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47"
ABLIT_PAGE="https://huggingface.co/${ABLIT_MODEL_ID}"

# Validate required variables
for var in HEAD_IP WORKER_IP IFACE IB_HCA IB_GID_INDEX MODEL_ID \
           MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS \
           MAX_NUM_BATCHED_TOKENS PORT TENSOR_PARALLEL_SIZE IMAGE \
           MASTER_PORT; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required variable $var is not set in .env"
        exit 1
    fi
done

WORKER_USER="${WORKER_USER:-}"
# Numeric sanity: the YaRN guard below does an arithmetic comparison on MAX_MODEL_LEN
if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
    exit 1
fi
# Per-node overrides — the two nodes may be cross-wired (head port f1 ↔ worker port f0),
# so the connected interface/HCA can have different names on each node.
WORKER_IFACE="${WORKER_IFACE:-$IFACE}"
WORKER_IB_HCA="${WORKER_IB_HCA:-$IB_HCA}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"   # fp8 needs files/patch_qsa_fp8_kv.py, applied automatically in step 4f; auto = bf16
# dtype of the GDN recurrent (SSM) state. The checkpoint asks for float32; the
# fused GDN kernel also accepts bfloat16 (FUSED_GDN_STATE_DTYPES in
# qwen_gdn_linear_attn.py). BF16 halves the ~0.23 GB per sequence the state
# costs to read and write every step, and halves the mamba page, which lets
# vLLM pick a smaller attention block. Empty keeps the checkpoint's float32.
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-}"
PLE_OFFLOAD="${PLE_OFFLOAD:-false}"
# KV-cache offload to NVMe (parking tier for sessions past the KV pool).
# Adds --kv-transfer-config with the out-of-tree nvme-direct spec
# (files/kvoffload/nvme_direct2.py): per-block files under a model-fenced
# namespace on each node's OWN NVMe (never the NFS weights share). Restore
# reads the file instead of re-prefilling; cross-boot keys need
# PYTHONHASHSEED=0, which the arm sets. See docs/plans/kv-offload-spark.md.
KV_OFFLOAD="${KV_OFFLOAD:-false}"
KV_ROOT="${KV_ROOT:-$HOME/fn-kv}"                 # head-side store root
KV_ROOT_WORKER="${KV_ROOT_WORKER:-}"              # empty -> worker $HOME/fn-kv (resolved in 6d; REMOTE_HOME is defined later)
KV_CAPACITY_GIB="${KV_CAPACITY_GIB:-500}"         # disk-free sanity floor
KV_IO_THREADS="${KV_IO_THREADS:-6}"               # per-process NVMe io threads
KV_PROMPT_ONLY="${KV_PROMPT_ONLY:-false}"         # false = persist generated turns too
# true = re-enable the [fn-kv-offload] per-step diagnostics (lookup/scan/
# store-skip/loadpath lines). Off by default: measured 09-21 at ~1200 log
# lines/s = 6-12 ms of every 64-90 ms scheduler step.
KV_OFFLOAD_DEBUG="${KV_OFFLOAD_DEBUG:-false}"
# Hybrid-state cache mode for KV_OFFLOAD boots. align = state blocks per
# hash chunk (REQUIRED for restores: the boundary state must exist on disk
# per chunk; "none" keeps one running state per request, which no external
# load can serve). Empty defers to the arm (6d defaults it to align there)
# or to the image default when the arm is off; set none/align to pin it.
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-}"
# Vision MLP intermediate_size=4304 is not divisible by 16 after TP split (4304/2=2152).
# NVFP4 kernels require input features % 16 == 0, so replicate the encoder on each GPU.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
HF_TOKEN="${HF_TOKEN:-}"
# Weight distribution. false (default) = each node keeps its own ~/.cache/huggingface
# copy, worker seeded by rsync from the head. true = head exports its cache over NFS
# on ConnectX and the worker mounts it read-only, keeping no local copy.
NFS_SHARE="${NFS_SHARE:-false}"
# Optional: head ConnectX address used as the NFS server (auto-detected from IFACE).
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
# Optional overrides from start-fp8.sh (applied after .env so FP8 can share
# the same cluster config). Weights still live on the head and are NFS-mounted.
if [[ -n "${OVERRIDE_MODEL_ID:-}" ]]; then
    MODEL_ID="$OVERRIDE_MODEL_ID"
fi
if [[ -n "${OVERRIDE_SERVED_MODEL_NAME:-}" ]]; then
    SERVED_MODEL_NAME="$OVERRIDE_SERVED_MODEL_NAME"
fi
if [[ -n "${OVERRIDE_MAX_MODEL_LEN:-}" ]]; then
    MAX_MODEL_LEN="$OVERRIDE_MAX_MODEL_LEN"
fi
if [[ -n "${OVERRIDE_YARN_ENABLE:-}" ]]; then
    YARN_ENABLE="$OVERRIDE_YARN_ENABLE"
fi
SKIP_PLE_PATCH="${SKIP_PLE_PATCH:-false}"
# FP8-dense hybrid checkpoint (NVFP4 experts + FP8 per-channel dense projections,
# built by files/fp8dense/make_fp8_dense_checkpoint.py). Needs the vLLM overlay
# patches in files/overlay (bind-mounted, no image rebuild).
FP8_DENSE="${FP8_DENSE:-false}"
FP8_DENSE_MODEL_ID="${FP8_DENSE_MODEL_ID:-MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense}"
if [[ "$FP8_DENSE" == "true" ]]; then
    MODEL_ID="$FP8_DENSE_MODEL_ID"
    DO_DOWNLOAD_DEFAULT=false   # local-only checkpoint, never on the Hub
fi
# ABLIT=1 serves the gated Keys house QSA o_proj checkpoint (L3–47).
# OVERRIDE_MODEL_ID (start-fp8.sh) and FP8_DENSE=true still win.
if [[ "$ABLIT" == "1" ]]; then
    if [[ "$FP8_DENSE" == "true" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: FP8_DENSE=true (MODEL_ID=$MODEL_ID)"
    elif [[ -n "${OVERRIDE_MODEL_ID:-}" && "$MODEL_ID" != "$ABLIT_MODEL_ID" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: OVERRIDE_MODEL_ID=$MODEL_ID"
    else
        MODEL_ID="$ABLIT_MODEL_ID"
    fi
fi
# Reduced-vocabulary MTP drafting: path to a token-id list from
# files/build_draft_vocab.py, or empty to draft over the full 248,320 vocabulary.
# Relative paths resolve against the repo root: the overlay mount below needs an
# absolute host path for docker -v, and `cd $SCRIPT_DIR` alone is not enough for
# callers that pass a path from a different working directory.
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-}"
if [[ -n "$MTP_DRAFT_VOCAB" && "$MTP_DRAFT_VOCAB" != /* ]]; then
    MTP_DRAFT_VOCAB="$SCRIPT_DIR/$MTP_DRAFT_VOCAB"
fi
# QSA Triton launch profile: stock | gb10 | path to JSON from files/qsa_gb10/bench_qsa_kernels.py
QSA_PROFILE="${QSA_PROFILE:-stock}"
# Refuse to launch when another process already holds the GPU (both nodes).
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
# torch.compile level. 0 = none (shipped). 3 = VLLM_COMPILE (Inductor fusion);
# on this arch it loads and keeps FULL decode graphs but buys nothing
# (+0.3%/+1.0%, inside noise; decode is bandwidth-bound) — measured on the
# single-Spark kit, 2026-09-06.
COMPILATION_MODE="${COMPILATION_MODE:-0}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"   # NONE for eager debug
# CUDA graph capture sizes for decode. vLLM's default list is [1,2,4] plus
# multiples of 8, each rounded up to a multiple of (1+MTP) and then filtered to
# <= (1+MTP)*MAX_NUM_SEQS before it becomes a decode key. At MTP=3,
# MAX_NUM_SEQS=8 that leaves keys {4,8,16,24,32}: the powers-of-two batches
# are graphed, but an odd-sequences batch pads up (a 5-sequence verify batch
# is 20 tokens and replays the 24-token graph, with 4 idle lanes). "auto"
# captures every (1+K(S))*S for S in 1..MAX_NUM_SEQS so each buildable batch
# gets its exact graph; a comma list sets them explicitly; empty keeps the
# vLLM default. The throughput effect of exact-vs-padded is inside noise at
# this shape (docs/bench/port-ab-20260920.md); capture costs ~1 s and a few
# MiB per size.
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-}"
# Batch-size schedule for the speculative token count, as
# "start:end:K,start:end:K" over inclusive batch-size (num_seqs) ranges. Empty
# keeps a constant MTP_NUM_SPECULATIVE_TOKENS at every batch size, which is
# what the static sweeps say you want (K=3 optimal at every concurrency).
# WARNING if you enable it: without a pinned V2 runner (EXTRA_DOCKER_ARGS
# "-e VLLM_USE_V2_MODEL_RUNNER=1") vLLM overrides cudagraph_mode to PIECEWISE
# from the draft model's config copy.
MTP_K_SCHEDULE="${MTP_K_SCHEDULE:-}"
# Replacement Jinja chat template (host path), mounted read-only into BOTH
# nodes' containers at /root/chat_template.jinja. Empty keeps the checkpoint's
# template + qwen3_coder. The shipped froggeric v22.5 template
# (files/chat-template/froggeric-qwen-fixed.jinja) fixes the stock template's
# raise_exception on reasoning_effort aliases, its crash on stringified-JSON
# tool arguments, and the xhigh-by-default token burn; it emits canonical XML
# tool calls, which pair with --tool-call-parser qwen3_xml (set automatically).
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
if [[ -n "$CHAT_TEMPLATE" && "$CHAT_TEMPLATE" != /* ]]; then
    CHAT_TEMPLATE="$SCRIPT_DIR/$CHAT_TEMPLATE"
fi
# Seconds the readiness loop waits for /health 200 before it archives the
# container's log, removes it, and exits non-zero (the supervisor/backoff
# relaunches). 20 min covers a cold JIT boot on both nodes.
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"

# YaRN only makes sense ABOVE the native 262144 context. At or below native,
# rope scaling degrades quality for zero benefit — force it off.
YARN_ENABLE="${YARN_ENABLE:-false}"
if [[ "$YARN_ENABLE" == "true" && "$MAX_MODEL_LEN" -le 262144 ]]; then
    echo "NOTE: MAX_MODEL_LEN=$MAX_MODEL_LEN <= native 262144 — YaRN force-disabled."
    YARN_ENABLE=false
fi

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
DO_DOWNLOAD="${DO_DOWNLOAD_DEFAULT:-true}"
DO_LAUNCH=true
DO_SYNC=true

for arg in "$@"; do
    case "$arg" in
        --no-download)  DO_DOWNLOAD=false ;;
        --no-launch)    DO_LAUNCH=false ;;
        --launch)       DO_DOWNLOAD=false; DO_SYNC=false ;;
        --nfs)          NFS_SHARE=true ;;
        --no-nfs)       NFS_SHARE=false ;;
        -h|--help)
            echo "Usage: $0 [--no-download] [--no-launch] [--launch] [--nfs|--no-nfs]"
            echo ""
            echo "  (default)      Download weights on head, rsync to worker, apply patch, launch"
            echo "  --no-download  Skip HF download (weights already cached on head)"
            echo "  --no-launch    Download + sync weights only, don't start vLLM"
            echo "  --launch       Skip download + sync; apply patch and launch"
            echo "  --nfs          Share the head cache over NFS instead of rsync (no worker copy)"
            echo "  --no-nfs       Force rsync distribution even if NFS_SHARE=true in .env"
            echo "  ABLIT=1        Serve the gated Keys house QSA L3-47 checkpoint"
            echo "                 (accept the Hugging Face terms, then ABLIT=1 ./download.sh)"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg (try --help)"
            ;;
    esac
done

if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    warn "ABLIT=1: serving gated Keys checkpoint ($ABLIT_MODEL_ID)."
    warn "     Safety refusals are removed. MTP, PLE, experts and the chat template stay stock."
    warn "     Compatible ONLY with the nvidia dual-Spark NVFP4 layout (this recipe)."
fi

# shellcheck source=files/nfs-share.sh
source "$SCRIPT_DIR/files/nfs-share.sh"

# ---------------------------------------------------------------------------
# Worker SSH helper
# ---------------------------------------------------------------------------
ssh_worker() {
    local user_prefix=""
    if [[ -n "$WORKER_USER" ]]; then
        user_prefix="${WORKER_USER}@"
    fi
    ssh -o StrictHostKeyChecking=no "${user_prefix}${WORKER_IP}" "$@"
}

# ---------------------------------------------------------------------------
# 1. Download the model weights (head node)
# ---------------------------------------------------------------------------
if $DO_DOWNLOAD; then
    "$SCRIPT_DIR/download.sh" "$MODEL_ID"
fi

# ---------------------------------------------------------------------------
# 2. Resolve local cache path
# ---------------------------------------------------------------------------
info "=== Step 2: Resolve cache path ==="

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
# Hub repo dir (blobs/refs/snapshots). Prefer this over `hf path`, which is not
# a real CLI command and (when it exists) often returns a snapshot subdir.
MODEL_DIR="$HUB_PATH/models--${ORG}--${NAME}"

if [[ ! -d "$MODEL_DIR" ]]; then
    local_guess=""
    for tool_cmd in "uvx hf path" "hf path" "huggingface-cli path"; do
        first_word="${tool_cmd%% *}"
        if command -v "$first_word" &>/dev/null; then
            local_guess=$($tool_cmd "$MODEL_ID" 2>/dev/null || true)
            [[ -n "$local_guess" && -d "$local_guess" ]] && break
            local_guess=""
        fi
    done
    if [[ -n "$local_guess" && -d "$local_guess" ]]; then
        # Walk up to models--org--name if the CLI pointed at a snapshot.
        case "$local_guess" in
            *"/models--${ORG}--${NAME}"*)
                MODEL_DIR="${local_guess%%/models--${ORG}--${NAME}*}/models--${ORG}--${NAME}"
                ;;
            *)
                MODEL_DIR="$local_guess"
                ;;
        esac
    fi
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Could not resolve local cache path for $MODEL_ID under $HUB_PATH
       Fetch it first (HF_TOKEN required):
         1. Set HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
         2. Open $ABLIT_PAGE
         3. Accept the terms on that page
         4. ABLIT=1 ./download.sh"
    fi
    err "Could not resolve local cache path for $MODEL_ID under $HUB_PATH"
fi
case "$MODEL_DIR" in
    "$HF_CACHE_DIR"|"$HF_CACHE_DIR"/*) ;;
    *) err "snapshot ${MODEL_DIR} is not under HF_HOME=${HF_CACHE_DIR}" ;;
esac

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
HEAD_MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"
if [[ ! -d "$HEAD_MODEL_PATH" ]]; then
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Could not find HF repo cache at $HEAD_MODEL_PATH
       Fetch it first (HF_TOKEN required):
         1. Set HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
         2. Open $ABLIT_PAGE
         3. Accept the terms on that page
         4. ABLIT=1 ./download.sh"
    fi
    err "Could not find HF repo cache at $HEAD_MODEL_PATH (resolved snapshot: ${MODEL_DIR:-none})"
fi

# This image ships huggingface_hub 1.28.0, whose offline branch reads
# refs/main with a bare f.read() and no .strip(). A hand-staged ref written the
# obvious way (`echo $SHA > refs/main`) carries a trailing newline, so offline
# resolution builds `snapshots/<sha>\n`, os.path.exists() fails, and vLLM dies
# at arg-parse with a repo-not-found error that names neither the file nor the
# newline. hf download writes the ref clean, so this only bites the staging
# flows this recipe advertises: NFS staging, rsync-your-own-hub-dir,
# --no-download. Normalise it here instead. Rewrites only when the bytes
# actually differ, so it is a no-op on hub-downloaded caches. Issue #36.
normalize_ref_main() {
    local ref="$1/refs/main"
    [[ -f "$ref" ]] || return 0
    local raw stripped
    raw=$(cat "$ref"; printf x); raw="${raw%x}"      # preserve trailing bytes
    stripped=$(printf '%s' "$raw" | tr -d '[:space:]')
    [[ -n "$stripped" && "$raw" != "$stripped" ]] || return 0
    printf '%s' "$stripped" > "$ref" || return 0
    warn "Normalised trailing whitespace in $ref (hf_hub 1.28 offline resolution
     reads this file without .strip(); see issue #36)"
}
normalize_ref_main "$HEAD_MODEL_PATH"

# Every shard named by the safetensors index must exist. A hub dir from an
# interrupted download is not enough — rsync would copy the hole to the worker
# and vLLM would die minutes into load.
SNAP=""
SNAP_RC=0
SNAP="$(python3 "$SCRIPT_DIR/files/resolve_snapshot.py" "$HEAD_MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
[[ -n "$SNAP" ]] || err "No snapshot under $HEAD_MODEL_PATH/snapshots"
if [[ "$SNAP_RC" -ne 0 ]]; then
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Checkpoint snapshot is incomplete (missing indexed weight shards).
       Resume with (HF_TOKEN required):  ABLIT=1 ./download.sh
       Accept the terms on $ABLIT_PAGE first."
    fi
    err "Checkpoint snapshot is incomplete (missing indexed weight shards).
       Resume with:  ./download.sh $MODEL_ID"
fi
ok "Model cache: $HEAD_MODEL_PATH  (snapshot $SNAP)"

# Checkpoints disagree about declaring text_config.ple_embedding_dtype, which is
# what the patched ple_layer.py dispatches on (nvidia/... omits it and declares
# the FP8 PLE table only in quantization_config.config_groups). Recover it from
# the checkpoint and feed it back via --hf-overrides below. Empty = already
# declared, or no quantized PLE table.
# PLE config must come from the snapshot the engine will load. refs/main
# names that revision. Directory order does not. Never guess by ls order.
# SNAP was already resolved by files/resolve_snapshot.py (complete shards,
# refs/main preferred).
PLE_CONFIG_DIR="$HEAD_MODEL_PATH/snapshots/$SNAP"
if [[ ! -f "$PLE_CONFIG_DIR/config.json" && -f "$MODEL_DIR/config.json" ]]; then
    PLE_CONFIG_DIR="$MODEL_DIR"
fi
if [[ ! -f "$PLE_CONFIG_DIR/config.json" ]]; then
    err "snapshot $SNAP has no config.json (partial download). Delete $PLE_CONFIG_DIR and re-run ./download.sh $MODEL_ID."
fi
PLE_EMBEDDING_DTYPE="${PLE_EMBEDDING_DTYPE:-}"
if [[ -z "$PLE_EMBEDDING_DTYPE" && -f "$PLE_CONFIG_DIR/config.json" ]]; then
    PLE_EMBEDDING_DTYPE=$(python3 "$SCRIPT_DIR/files/detect_ple_dtype.py" "$PLE_CONFIG_DIR")
fi
if [[ -n "$PLE_EMBEDDING_DTYPE" ]]; then
    ok "PLE table dtype not declared by checkpoint — overriding to $PLE_EMBEDDING_DTYPE"
fi
SNAPSHOT_SHA=$(basename "${PLE_CONFIG_DIR%/}")

# Resolve the worker's HF cache. It mirrors the head's absolute path unless that
# path lives under $HOME (then the prefix is rewritten to the worker's $HOME), or
# WORKER_HF_HOME overrides it outright.
REMOTE_HOME=$(ssh_worker "echo \"\$HOME\"")
[[ -n "$REMOTE_HOME" ]] || err "Could not resolve \$HOME on worker ($WORKER_IP). Check SSH / WORKER_USER."
if [[ -n "${WORKER_HF_HOME:-}" ]]; then
    REMOTE_HF="$WORKER_HF_HOME"
elif [[ "$HF_CACHE_DIR" == "$HOME" || "$HF_CACHE_DIR" == "$HOME/"* ]]; then
    REMOTE_HF="${REMOTE_HOME}${HF_CACHE_DIR#"$HOME"}"
else
    REMOTE_HF="$HF_CACHE_DIR"
fi
info "Head HF cache:   $HF_CACHE_DIR"
info "Worker HF cache: $REMOTE_HF"

# ---------------------------------------------------------------------------
# 3. Distribute weights to the worker.
#    Default: rsync into the worker's own HF cache (worker keeps a local copy).
#    NFS_SHARE=true: export the head cache over NFS (ConnectX); no worker copy.
# ---------------------------------------------------------------------------
REMOTE_HUB="${REMOTE_HF}/hub"

if [[ "$NFS_SHARE" == "true" ]]; then
    info "=== Step 3: NFS-share weights from head ==="
    nfs_ensure_server
else
    info "=== Step 3: Sync weights to worker ($WORKER_IP) ==="
    if ! $DO_SYNC; then
        info "  --launch: skipping sync (assuming worker cache is current)"
    else
        WORKER_SNAP_RC=2
        if ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
            set +e
            ssh_worker python3 - "$REMOTE_HUB/models--${ORG}--${NAME}" \
                < "$SCRIPT_DIR/files/resolve_snapshot.py" >/dev/null
            WORKER_SNAP_RC=$?
            set -e
        fi
        if [[ "$WORKER_SNAP_RC" -eq 0 ]]; then
            ok "Worker already has a complete snapshot of models--${ORG}--${NAME} — skipping rsync."
            info "  (delete $REMOTE_HUB/models--${ORG}--${NAME} on the worker to force a re-sync)"
        else
            [[ -d "$HEAD_MODEL_PATH" ]] || err "HEAD ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND. Nothing to sync; run ./download.sh first."
            if [[ "$WORKER_SNAP_RC" -eq 1 ]]; then
                warn "Worker snapshot is incomplete — re-syncing."
            fi
            info "  Worker cache: $REMOTE_HUB"
            warn "  This copies the full checkpoint over the network and needs the same"
            warn "  free space on the worker. NFS_SHARE=true avoids both — see the README."
            ssh_worker "mkdir -p '$REMOTE_HUB'"
            rsync -av --progress --partial \
                "${HEAD_MODEL_PATH}/" \
                "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${REMOTE_HUB}/models--${ORG}--${NAME}/"
            ok "Rsync complete."
        fi
    fi
    # Same hf_hub 1.28 refs/main hazard as the head (issue #36). Only the rsync
    # path needs this: under NFS_SHARE the worker mounts the head's cache, so
    # normalising the head above already covers it. Runs whether or not the
    # sync ran, since a worker cache staged by hand is exactly the exposed case.
    # Compare byte count to stripped length. Do NOT compare against
    # "$(cat "$ref")": command substitution strips trailing newlines, so the
    # comparison is blind to exactly the byte this is meant to catch.
    ssh_worker "ref='$REMOTE_HUB/models--${ORG}--${NAME}/refs/main'
        if [ -f \"\$ref\" ]; then
            s=\$(tr -d '[:space:]' < \"\$ref\")
            n=\$(wc -c < \"\$ref\")
            if [ -n \"\$s\" ] && [ \"\$n\" -ne \"\${#s}\" ]; then
                printf '%s' \"\$s\" > \"\$ref\" && echo normalized
            fi
        fi" 2>/dev/null | grep -q normalized \
        && warn "Normalised trailing whitespace in the worker's refs/main (issue #36)"
fi

# ---------------------------------------------------------------------------
# 4. Verify weights on head (the worker is checked once its cache is in place)
# ---------------------------------------------------------------------------
info "=== Step 4: Verify weights on head ==="
if [[ -d "$HEAD_MODEL_PATH" ]]; then
    HEAD_SIZE=$(du -sh "$HEAD_MODEL_PATH" 2>/dev/null | cut -f1)
    ok "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH ($HEAD_SIZE)"
else
    err "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND"
fi

if ! $DO_LAUNCH; then
    if [[ "$NFS_SHARE" == "true" ]]; then
        nfs_ensure_worker_volume
        if nfs_worker_has_model "hub/models--${ORG}--${NAME}"; then
            ok "WORKER ($WORKER_IP): nfs $NFS_VOLUME → hub/models--${ORG}--${NAME}"
        else
            err "WORKER cannot see hub/models--${ORG}--${NAME} over NFS. Check: docker logs $NFS_CONTAINER"
        fi
    elif ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
        WORKER_SIZE=$(ssh_worker "du -sh '$REMOTE_HUB/models--${ORG}--${NAME}' 2>/dev/null | cut -f1" || true)
        ok "WORKER ($WORKER_IP): $REMOTE_HUB/models--${ORG}--${NAME} (${WORKER_SIZE:-?})"
    else
        err "WORKER ($WORKER_IP): $REMOTE_HUB/models--${ORG}--${NAME} — NOT FOUND. Re-run without --launch to sync."
    fi
fi

# ---------------------------------------------------------------------------
# 4b. Preflight: both GPUs must be free (another vLLM/SGLang tenant would OOM us
#     ten minutes into weight loading).
# ---------------------------------------------------------------------------
gpu_tenants() {  # prints "pid,name,mem" lines for compute apps, empty if idle
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^$/d'
}
if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]]; then
    info "=== Step 4b: GPU preflight ==="
    HEAD_TENANTS=$(gpu_tenants || true)
    WORKER_TENANTS=$(ssh_worker "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^\$/d'" || true)
    if [[ -n "$HEAD_TENANTS" || -n "$WORKER_TENANTS" ]]; then
        echo "HEAD:   ${HEAD_TENANTS:-idle}"
        echo "WORKER: ${WORKER_TENANTS:-idle}"
        err "GPU is in use on at least one node (set REQUIRE_IDLE_GPU=false to override)."
    fi
    ok "Both GPUs idle."
fi

# ---------------------------------------------------------------------------
# 4b-2. Release the checkpoint's own page cache (issue #35)
#
# On GB10 the page cache shares one unified-memory pool with the model. A
# checkpoint that was just written -- hf download, the rsync above, or an NFS
# read on a previous launch -- leaves its bytes resident as clean pages, and
# weight loading can then die partway with CUDA OOM on an otherwise idle box.
#
# `echo 3 > /proc/sys/vm/drop_caches` needs root, which these nodes do not have
# passwordless; that is why the README tells you to do it yourself rather than
# claiming start.sh does it. posix_fadvise(POSIX_FADV_DONTNEED) drops clean
# pages of files we can open, needs no privileges, and touches only this
# checkpoint instead of the whole system's cache.
#
# Best-effort by construction: evicting the cache is an optimisation, so a
# failure here must never block a launch that would otherwise work.
# EVICT_PAGE_CACHE=false opts out.
EVICT_PAGE_CACHE="${EVICT_PAGE_CACHE:-true}"
if $DO_LAUNCH && [[ "$EVICT_PAGE_CACHE" == "true" ]]; then
    info "=== Step 4b-2: Release checkpoint page cache ==="
    EVICT_PY="$SCRIPT_DIR/files/evict_page_cache.py"
    python3 "$EVICT_PY" "$HEAD_MODEL_PATH" 2>&1 | sed 's/^/  HEAD   /' || true

    # Both nodes hold their own cache, but in different places:
    #
    #   rsync mode : the worker has its own copy on local disk, so fadvise
    #                runs over ssh against that path.
    #   NFS mode   : the worker has NO local copy. Its resident pages are NFS
    #                client cache, reachable only through the mount, and that
    #                mount exists only inside a container. So the pass runs in
    #                a throwaway container holding the same volume, the way
    #                nfs_worker_has_model() already probes it. Verified on
    #                NFS 4.2: fadvise evicts there exactly as it does locally
    #                (3.00 GiB -> 0.00 GiB resident, measured with mincore).
    #
    # One copy of the script on the worker serves both modes.
    if scp -q "$EVICT_PY" \
        "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/evict_page_cache.py" 2>/dev/null
    then
        if [[ "$NFS_SHARE" == "true" ]]; then
            ssh_worker "docker run --name vllm-fn-evict-\$\$ \
                -v '${NFS_VOLUME}:/hf:ro' \
                -v /tmp/evict_page_cache.py:/evict.py:ro \
                --entrypoint python3 '$IMAGE' \
                /evict.py '/hf/hub/models--${ORG}--${NAME}'; \
                rc=\$?; docker rm -f vllm-fn-evict-\$\$ >/dev/null 2>&1; exit \$rc" 2>&1 \
                | sed 's/^/  WORKER /' || true
        else
            ssh_worker "python3 /tmp/evict_page_cache.py \
                '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>&1 \
                | sed 's/^/  WORKER /' || true
        fi
    else
        warn "  Could not copy the evictor to the worker; skipping its pass."
    fi
fi

# ---------------------------------------------------------------------------
# 4c. vLLM overlay patches (bind-mounted files, no image rebuild).
#     files/overlay/*.py   -> FP8-dense loader support (only with FP8_DENSE=true)
#     files/qsa_gb10/qsa.py -> QSA launch-profile override (only with QSA_PROFILE != stock)
# ---------------------------------------------------------------------------
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
OVERLAY_MOUNTS=()        # head-side "-v host:container:ro"
OVERLAY_FILES=()         # host files to scp to the worker (/tmp/vllm-overlay/<name>)
OVERLAY_ENV=()
add_overlay() {          # add_overlay <host file> <container path>
    [[ -f "$1" ]] || err "overlay file missing: $1"
    # Two overlays on one container path would silently race in docker run, and
    # the worker copies land in a flat /tmp/vllm-overlay keyed by basename, so a
    # basename clash would have one file quietly overwrite the other there.
    for existing in "${OVERLAY_FILES[@]:-}"; do
        if [[ "${existing#*|}" == "$2" ]]; then
            err "overlay conflict on $2: already claimed by ${existing%%|*}, now $1"
        fi
        if [[ "$(basename "${existing%%|*}")" == "$(basename "$1")" ]]; then
            err "overlay basename clash on $(basename "$1"): ${existing%%|*} vs $1
       (worker overlays share a flat /tmp/vllm-overlay directory)"
        fi
    done
    OVERLAY_MOUNTS+=("-v $1:$2:ro")
    OVERLAY_FILES+=("$1|$2")
}
extract_from_image() {   # extract_from_image <container path> <host dest>
    [[ -f "$2" ]] && return 0
    info "  Extracting $(basename "$1") from image..."
    local c; c=$(docker create "$IMAGE" /bin/true)
    docker cp "$c:$1" "$2" >/dev/null
    docker rm "$c" >/dev/null 2>&1
    [[ -f "$2" ]] || err "Failed to extract $1 from image."
}
if $DO_LAUNCH && [[ "$FP8_DENSE" == "true" ]]; then
    info "=== Step 4c: FP8-dense overlay ==="
    OV="$SCRIPT_DIR/files/overlay"
    [[ -f "$OV/modelopt.py" ]] || python3 "$OV/apply_patches.py"
    add_overlay "$OV/modelopt.py"        "$VLLM_PKG/model_executor/layers/quantization/modelopt.py"
    add_overlay "$OV/model.py"           "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/model.py"
    add_overlay "$OV/hyperconnection.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/hyperconnection.py"
    add_overlay "$OV/mtp.py"             "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    ok "FP8-dense overlay: 4 files"
fi
if $DO_LAUNCH && [[ "$QSA_PROFILE" != "stock" ]]; then
    info "=== Step 4d: QSA profile overlay ($QSA_PROFILE) ==="
    QO="$SCRIPT_DIR/files/qsa_gb10"
    [[ -f "$QO/qsa.py" ]] || python3 "$QO/apply_patch.py"
    add_overlay "$QO/qsa.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    if [[ -f "$QSA_PROFILE" ]]; then
        add_overlay "$QSA_PROFILE" "/etc/vllm-qsa-profile.json"
        OVERLAY_ENV+=("-e VLLM_QSA_PROFILE_JSON=/etc/vllm-qsa-profile.json")
    else
        OVERLAY_ENV+=("-e VLLM_QSA_PROFILE=$QSA_PROFILE")
    fi
fi

# ---------------------------------------------------------------------------
# 4d. Reduced-vocabulary MTP drafting (FR-Spec style).
#     The drafter owns a full 248,320-row BF16 lm_head that is read once per
#     draft step; slicing it to a frequency-ranked subset is the single largest
#     bandwidth lever in a decode step. Output-safe: a draft outside the subset
#     is rejected at verification, never emitted. See files/patch_mtp_draft_vocab.py.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ -n "$MTP_DRAFT_VOCAB" ]]; then
    info "=== Step 4e: MTP reduced draft vocabulary ==="
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]]; then
        err "MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
    fi
    [[ -f "$MTP_DRAFT_VOCAB" ]] || err "MTP_DRAFT_VOCAB file not found: $MTP_DRAFT_VOCAB
       Build one with: python3 files/build_draft_vocab.py <corpus.jsonl> --out draft_vocab.txt --size 65536"
    if [[ "$FP8_DENSE" == "true" ]]; then
        err "MTP_DRAFT_VOCAB and FP8_DENSE both overlay nvidia/mtp.py - pick one."
    fi
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py" \
                       "$SCRIPT_DIR/files/mtp_patched.py.orig"
    python3 "$SCRIPT_DIR/files/patch_mtp_draft_vocab.py"
    add_overlay "$SCRIPT_DIR/files/mtp_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    add_overlay "$MTP_DRAFT_VOCAB" "/etc/vllm-draft-vocab.txt"
    OVERLAY_ENV+=("-e VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt")
    ok "Draft vocab: $(wc -l < "$MTP_DRAFT_VOCAB") ids from $MTP_DRAFT_VOCAB"
elif $DO_LAUNCH && [[ "$MTP_NUM_SPECULATIVE_TOKENS" != "0" ]]; then
    warn "MTP=$MTP_NUM_SPECULATIVE_TOKENS is drafting over the FULL 248,320-token head"
    warn "     (0.59 GiB/rank at TP=2, read once per draft step). Setting MTP_DRAFT_VOCAB"
    warn "     to files/draft_vocab_en_code_47k.txt cuts that ~5x; see .env.sample."
fi

# Chat template: reuse the overlay machinery so the file lands at the same
# container path on BOTH nodes (worker copies ride /tmp/vllm-overlay).
if $DO_LAUNCH && [[ -n "$CHAT_TEMPLATE" ]]; then
    info "=== Step 4e-2: chat template overlay ==="
    add_overlay "$CHAT_TEMPLATE" "/root/chat_template.jinja"
    ok "Chat template: $CHAT_TEMPLATE"
fi

# ---------------------------------------------------------------------------
# 4e. FP8 KV cache. The stock QSA kernels hard-refuse anything but BF16 KV
#     (supported_kv_cache_dtypes = ["auto","bfloat16"]); this teaches them to
#     read an FP8-e4m3 cache with per-tensor scales applied after the dots.
#     A capacity trade, not a free win - see the README before enabling.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
    info "=== Step 4f: FP8 KV cache patch ($KV_CACHE_DTYPE) ==="
    if [[ "$QSA_PROFILE" != "stock" ]]; then
        err "KV_CACHE_DTYPE=$KV_CACHE_DTYPE and QSA_PROFILE=$QSA_PROFILE both overlay ops/qsa.py - pick one."
    fi
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
                       "$SCRIPT_DIR/files/qsa_ops_patched.py.orig"
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py" \
                       "$SCRIPT_DIR/files/qsa_nvidia_patched.py.orig"
    python3 "$SCRIPT_DIR/files/patch_qsa_fp8_kv.py"
    add_overlay "$SCRIPT_DIR/files/qsa_ops_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    add_overlay "$SCRIPT_DIR/files/qsa_nvidia_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py"
    warn "FP8 KV is a quality trade on sparse attention - validate reasoning on your workload."
fi

# ---------------------------------------------------------------------------
# 4d. MTP layer-index alias overlay.
#     vLLM builds the MTP draft layer at the absolute index that continues the
#     main stack (mtp.layers.48 for num_hidden_layers=48) and matches that
#     prefix against quantization_config.quantized_layers by exact string.
#     nvidia/... records only mtp.layers.0, so the lookup misses, the MTP MoE is
#     built unquantized, and its FP8 weight_scale_inv tensors fail to load. We
#     bind-mount a config.json carrying both names (what the known-good
#     local-inference-lab checkpoint ships) — the HF cache is left untouched.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ -n "${SNAPSHOT_SHA:-}" ]]; then
    info "=== Step 4g: MTP layer-index alias ==="
    CONTAINER_SNAPSHOT="/root/.cache/huggingface/hub/models--${ORG}--${NAME}/snapshots/${SNAPSHOT_SHA}"
    rm -f "$SCRIPT_DIR/files/config_patched.json" \
          "$SCRIPT_DIR/files/hf_quant_config_patched.json"
    PATCHED_FILES=$(python3 "$SCRIPT_DIR/files/patch_checkpoint_config.py" \
        "$PLE_CONFIG_DIR" "$SCRIPT_DIR/files")
    if [[ -z "$PATCHED_FILES" ]]; then
        ok "Checkpoint already declares absolute MTP layer indices"
    else
        # quantized_layers lives in BOTH config.json and the legacy
        # hf_quant_config.json, and the two can disagree: nvidia/... rev
        # fc694b54 says FP8_PB_WO in config.json and FP8_BLOCK_SCALES in the
        # sidecar. Runtime evidence (issue #38) shows the MoE dispatch
        # consumes config.json, so that mount is the one that must be right.
        # The sidecar is mounted too, for consistency, not because it wins.
        for cfg_name in $PATCHED_FILES; do
            case "$cfg_name" in
                config.json)         host_file="$SCRIPT_DIR/files/config_patched.json" ;;
                hf_quant_config.json) host_file="$SCRIPT_DIR/files/hf_quant_config_patched.json" ;;
                *) err "unexpected patched config: $cfg_name" ;;
            esac
            add_overlay "$host_file" "$CONTAINER_SNAPSHOT/$cfg_name"
        done
        ok "MTP experts alias added to: $PATCHED_FILES"
    fi

    # Speculative decoding needs the MTP routed experts to be built with a
    # quantization method the mixed-precision dispatch actually implements.
    # ModelOptMixedPrecisionConfig.get_quant_method covers FP8 / NVFP4 /
    # W4A16_NVFP4 / MXFP8 for RoutedExperts and returns None for anything else,
    # which yields a silently *unquantized* MoE that then dies ~7 min into the
    # load. Fail fast here instead.
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        MTP_ALGO=$(python3 "$SCRIPT_DIR/files/patch_checkpoint_config.py" \
            --mtp-moe-algo "$PLE_CONFIG_DIR") && MTP_RC=0 || MTP_RC=$?
        if [[ "$MTP_RC" -eq 3 ]]; then
            err "MTP experts are ${MTP_ALGO}, which this image's mixed-precision MoE
       dispatch cannot build (supports FP8 / NVFP4 / W4A16_NVFP4 / MXFP8 /
       FP8_BLOCK_SCALES; FP8_PB_WO with group_size 128 is treated as
       FP8_BLOCK_SCALES). Set MTP_NUM_SPECULATIVE_TOKENS=0 in .env to serve
       without speculative decoding, or use a checkpoint whose MTP experts
       are NVFP4."
        fi
        ok "MTP experts quantization: ${MTP_ALGO:-unquantized} (supported)"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Ensure Docker image on both nodes
# ---------------------------------------------------------------------------
if $DO_LAUNCH; then
    info "=== Step 5: Docker image '$IMAGE' ==="

    if ! docker image inspect "$IMAGE" &>/dev/null; then
        info "Pulling $IMAGE ..."
        docker pull "$IMAGE"
    fi
    ok "Image ready on head."

    LOCAL_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")
    REMOTE_ID=$(ssh_worker "docker image inspect --format '{{.Id}}' '$IMAGE' 2>/dev/null" || echo "")

    if [[ "$LOCAL_ID" != "$REMOTE_ID" ]]; then
        info "Pulling image on worker..."
        ssh_worker "docker pull '$IMAGE'"
        ok "Image ready on worker."
    else
        ok "Image already on worker."
    fi

    # ---------------------------------------------------------------------------
    # 6. Prepare the PLE patch
    #    Mixed-quant NVFP4 checkpoints declare ple_embedding_dtype in config:
    #      nvfp4          -> packed uint8 PLE table (this checkpoint)
    #      float8_e4m3fn  -> FP8 PLE excluded from the parent ModelOpt config
    #    files/patch_ple_layer.py adds NVFP4 + mixed dispatch (vLLM PR #53899 logic).
    # ---------------------------------------------------------------------------
    PATCHED_PLE="$SCRIPT_DIR/files/ple_layer_patched.py"
    PLE_ORIG="$SCRIPT_DIR/files/ple_layer_patched.py.orig"
    HEAD_PLE_MOUNT=""
    WORKER_PLE_MOUNT=""

    if [[ "$SKIP_PLE_PATCH" == "true" ]]; then
        info "=== Step 6: PLE patch skipped (native FP8 checkpoint) ==="
    else
    info "=== Step 6: Prepare PLE patch ==="

    if [[ ! -f "$PLE_ORIG" ]]; then
        info "Extracting ple_layer.py from image..."
        mkdir -p "$SCRIPT_DIR/files"
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py" "$PLE_ORIG"
        docker rm "$tmp_container" >/dev/null 2>&1
        [[ -f "$PLE_ORIG" ]] || err "Failed to extract ple_layer.py from image. Is the image pulled?"
    fi

    python3 "$SCRIPT_DIR/files/patch_ple_layer.py"
    [[ -f "$PATCHED_PLE" ]] || err "PLE patch file not found after patch_ple_layer.py"

    ok "PLE patch ready: $PATCHED_PLE"
    HEAD_PLE_MOUNT="-v $PATCHED_PLE:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
    WORKER_PLE_MOUNT="-v /tmp/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
    fi

    # ---------------------------------------------------------------------------
    # 6b. MXFP8 kernel-fallback patch.
    #     FlashInfer mm_mxfp8 needs N,K >= 128 and both divisible by 32. Two
    #     shapes in this checkpoint miss that — linear_attn.in_proj_a/b [48,2560]
    #     (fatal at engine start) and the vision MLP fc1 [4304,1152] — so those
    #     layers are routed to the BF16 emulation kernel. visual.* stays fully
    #     emulated (verified multimodal path; global dequant would OOM).
    # ---------------------------------------------------------------------------
    PATCHED_MODELOPT="$SCRIPT_DIR/files/modelopt_patched.py"
    MODELOPT_ORIG="$SCRIPT_DIR/files/modelopt_patched.py.orig"
    HEAD_MODELOPT_MOUNT=""
    WORKER_MODELOPT_MOUNT=""
    MODEL_OPT_PKG="$VLLM_PKG/model_executor/layers/quantization/modelopt.py"

    info "=== Step 6b: Prepare MXFP8 kernel-fallback patch ==="
    if [[ ! -f "$MODELOPT_ORIG" ]]; then
        info "Extracting modelopt.py from image..."
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:$MODEL_OPT_PKG" "$MODELOPT_ORIG"
        docker rm "$tmp_container" >/dev/null 2>&1
        [[ -f "$MODELOPT_ORIG" ]] || err "Failed to extract modelopt.py from image."
    fi
    python3 "$SCRIPT_DIR/files/patch_modelopt_mxfp8.py"
    [[ -f "$PATCHED_MODELOPT" ]] || err "modelopt patch missing after patch_modelopt_mxfp8.py"
    # Stacks on top: adds the FP8_BLOCK_SCALES routed-expert branch that neither
    # this image nor upstream vLLM has, which is what MTP needs on this checkpoint.
    python3 "$SCRIPT_DIR/files/patch_modelopt_fp8_block_moe.py"
    ok "MXFP8 fallback patch ready: $PATCHED_MODELOPT"
    HEAD_MODELOPT_MOUNT="-v $PATCHED_MODELOPT:$MODEL_OPT_PKG:ro"
    WORKER_MODELOPT_MOUNT="-v /tmp/modelopt_patched.py:$MODEL_OPT_PKG:ro"

    # ---------------------------------------------------------------------------
    # 6c. PLE CPU-offload overlays (ported from the single-Spark kit).
    #     With PLE_OFFLOAD=true the 26.8 GiB NVFP4 n-gram table leaves the GPU:
    #     EACH NODE spawns its own PleOffloadWorker (the registrations ride on
    #     CUDA-IPC and node-local ZMQ, so they cannot cross nodes — see the
    #     multi-node section of files/patch_ple_offload.py), serving the table
    #     from a memory-mapped pre-packed file on that node's NVMe — page-cache
    #     backed, so the table's home is the disk with only touched pages
    #     resident, not a 26.8 GiB anonymous RAM copy. Both GB10 fixes ride
    #     along (no stream-memory-ops, done-flag handshake) plus MADV_RANDOM
    #     and the batched posix_fadvise prefetch.
    #     The five patched files bind-mount over the package on BOTH nodes
    #     (protocol/worker must be byte-identical for the registration pickle).
    # ---------------------------------------------------------------------------
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        info "=== Step 6c: PLE offload overlays + packed table ==="
        OFFLOAD_DIR="$SCRIPT_DIR/files/ple_offload"
        # The image's gpu_worker.py owns the spawn/registration wiring; keep a
        # local orig so the patcher can regenerate the overlay from it.
        extract_from_image "$VLLM_PKG/v1/worker/gpu_worker.py" "$OFFLOAD_DIR/orig/gpu_worker.py"
        # Regenerate from the tracked orig/ pair (fail-loud on anchor drift).
        python3 "$SCRIPT_DIR/files/patch_ple_offload.py"
        for _f in ple_offload_layer connector worker protocol gpu_worker; do
            [[ -f "$OFFLOAD_DIR/$_f.py" ]] || err "offload patch missing: $_f.py"
        done
        add_overlay "$OFFLOAD_DIR/ple_offload_layer.py" "$VLLM_PKG/model_executor/layers/ple_offload_layer.py"
        add_overlay "$OFFLOAD_DIR/connector.py"         "$VLLM_PKG/v1/ple_offload/connector.py"
        add_overlay "$OFFLOAD_DIR/worker.py"            "$VLLM_PKG/v1/ple_offload/worker.py"
        add_overlay "$OFFLOAD_DIR/protocol.py"          "$VLLM_PKG/v1/ple_offload/protocol.py"
        add_overlay "$OFFLOAD_DIR/gpu_worker.py"        "$VLLM_PKG/v1/worker/gpu_worker.py"

        # Packed table, built once on the head and pushed to the worker (each
        # node's offload process mmaps its OWN file).
        PLE_PACKED_HOST="$HOME/.cache/vllm/ple_cache/${ORG}--${NAME}"
        PLE_PACKED_CTR="/root/.cache/vllm/ple_cache/${ORG}--${NAME}"
        if ! ls "$PLE_PACKED_HOST"/*.packed_u8 >/dev/null 2>&1; then
            info "Building packed PLE table (one-time, ~40 s, <1 GiB RAM, no GPU)..."
            mkdir -p "$PLE_PACKED_HOST"
            # The snapshot's files are relative links that chain two levels up
            # (model blobs -> hub CAS blobs). Mount the model dir's parent so
            # the whole chain resolves inside the container; the model dir alone
            # leaves the second hop dangling.
            docker run --rm --name "${CONTAINER_NAME:-vllm-fn}-plebuild" --memory 6g --cpus 8 \
                -v "$(dirname "$MODEL_DIR"):/m:ro" -v "$HOME/.cache/vllm/ple_cache:/out" \
                -v "$SCRIPT_DIR/files/build_ple_packed_table.py:/b.py:ro" \
                --entrypoint python3 "$IMAGE" -u /b.py \
                "/m/$(basename "$MODEL_DIR")/snapshots/$SNAP" "/out/${ORG}--${NAME}"
        fi
        ls "$PLE_PACKED_HOST"/*.packed_u8 >/dev/null 2>&1 || err "packed table missing after build"
        ok "Packed PLE table: $(ls "$PLE_PACKED_HOST"/*.packed_u8 | head -1) ($(du -sh "$PLE_PACKED_HOST" | cut -f1))"

        # The worker's offload process mmaps its OWN copy — CUDA-IPC handles
        # never cross nodes, so the table lives on both NVMe drives. One-time
        # push; --size-only keeps the check cheap across reboots.
        if ! ssh_worker "test -f '$REMOTE_HOME/.cache/vllm/ple_cache/${ORG}--${NAME}/$(basename "$(ls "$PLE_PACKED_HOST"/*.packed_u8 | head -1)")'"; then
            info "  Pushing packed PLE table to worker (~27 GiB, a few seconds over IB)..."
            ssh_worker "mkdir -p '$REMOTE_HOME/.cache/vllm/ple_cache/${ORG}--${NAME}'"
            rsync -a --partial "$PLE_PACKED_HOST/" \
                "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${REMOTE_HOME}/.cache/vllm/ple_cache/${ORG}--${NAME}/"
        else
            ok "Worker already has the packed PLE table."
        fi

        OVERLAY_ENV+=("-e VLLM_PLE_PACKED_TABLE_DIR=$PLE_PACKED_CTR")
        OVERLAY_ENV+=("-e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300")
    fi

    # ---------------------------------------------------------------------------
    # 6d. KV-cache offload to NVMe (optional; KV_OFFLOAD=true).
    #     Parks evicted sessions' KV as per-block files under a model-fenced
    #     namespace on each node's OWN NVMe (never the NFS weights share — the
    #     PLE table rule). An out-of-tree spec (files/kvoffload/nvme_direct2.py,
    #     Apache-2.0, ported from the GLM-5.3 kit's qualified lane) rides the
    #     image's OffloadingConnector via PYTHONPATH — no vLLM source edits.
    #     Restore reads files; a wrong-model/wrong-revision boot lands in a
    #     DIFFERENT namespace (fail-closed sidecar), never on someone else's
    #     bytes. Cross-boot key stability REQUIRES PYTHONHASHSEED=0 (block-hash
    #     chain root is otherwise os.urandom), set on both nodes here.
    # ---------------------------------------------------------------------------
    KV_HEAD_MOUNTS=""; KV_WORKER_MOUNTS=""; KV_ENV=""
    if [[ "$KV_OFFLOAD" == "true" ]]; then
        info "=== Step 6d: KV-cache offload (NVMe-direct) ==="
        # The worker's store root defaults to its OWN $HOME (REMOTE_HOME is
        # resolved early in the script; $HOME inside ssh_worker is the worker's).
        [[ -n "$KV_ROOT_WORKER" ]] || KV_ROOT_WORKER="${KV_ROOT:-$HOME/fn-kv}"
        case "$EXTRA_VLLM_ARGS" in
            *--kv-transfer-config*)
                err "KV_OFFLOAD=true and an EXTRA_VLLM_ARGS --kv-transfer-config conflict - pick one." ;;
        esac
        KV_SPEC_HOST="$SCRIPT_DIR/files/kvoffload/nvme_direct2.py"
        [[ -f "$KV_SPEC_HOST" ]] || err "missing $KV_SPEC_HOST"
        KV_CAPACITY=$(( KV_CAPACITY_GIB * 1024 * 1024 * 1024 ))
        mkdir -p "$KV_ROOT" || err "cannot create $KV_ROOT"
        _fst=$(stat -f -c %T "$KV_ROOT" 2>/dev/null || echo unknown)
        case "$_fst" in
            tmpfs|ramfs) err "KV_ROOT=$KV_ROOT lives on $_fst - point it at the NVMe volume." ;;
        esac
        _av=$(df -B1 --output=avail "$KV_ROOT" | awk 'NF && $1 ~ /^[0-9]+$/ {v=$1} END{print v+0}')
        (( _av >= KV_CAPACITY )) || err "head: KV_ROOT has ${_av:-?} B free; capacity=${KV_CAPACITY} B"
        ssh_worker "mkdir -p '$KV_ROOT_WORKER'" || err "cannot create $KV_ROOT_WORKER on worker"
        _av=$(ssh_worker "df -B1 --output=avail '$KV_ROOT_WORKER'" | awk 'NF && $1 ~ /^[0-9]+$/ {v=$1} END{print v+0}')
        (( _av >= KV_CAPACITY )) || err "worker: KV_ROOT_WORKER has ${_av:-?} B free; capacity=${KV_CAPACITY} B"
        # The mount of a file creates /opt/fnkv/kvoffload/ in the container;
        # PYTHONPATH=/opt/fnkv then makes kvoffload.nvme_direct2 importable
        # as a namespace package (same shape as the GLM kit's lane). Head
        # mounts in place; worker gets a copy staged to /tmp.
        ssh_worker "rm -f /tmp/fnkv-nvme_direct2.py"
        scp -q "$KV_SPEC_HOST" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/fnkv-nvme_direct2.py" \
            || err "failed to stage nvme_direct2.py on worker"
        KV_HEAD_MOUNTS="-v $KV_SPEC_HOST:/opt/fnkv/kvoffload/nvme_direct2.py:ro -v $KV_ROOT:/mnt/fn-kv"
        KV_WORKER_MOUNTS="-v /tmp/fnkv-nvme_direct2.py:/opt/fnkv/kvoffload/nvme_direct2.py:ro -v $KV_ROOT_WORKER:/mnt/fn-kv"
        KV_ENV="-e PYTHONHASHSEED=0 -e PYTHONPATH=/opt/fnkv"
        [[ "$KV_OFFLOAD_DEBUG" == "true" ]] && KV_ENV+=" -e KV_OFFLOAD_DEBUG=1"
        # Overlay generators (house pattern: extract orig from image once,
        # regenerate from it every launch, fail loud on anchor drift).
        #  - offloading/config.py: assert -> classified receipt (insurance;
        #    our geometry is aligned, see the generator docstring).
        #  - offloading/scheduler.py: is_scratch exclusion for the QSA
        #    indexer ring (CircularBufferSpec, block 8, 13 layers) — GLM #58
        #    touchpoint port; REQUIRED, the boot assert is upstream of it.
        #  - mamba_hybrid.py + scheduler.py: the blazux align-mode block-size
        #    fix — REQUIRED for correct restore-on-hit of the hybrid state
        #    (port of patch_mamba_block_size.py, Apache-2.0).
        KVCFG="$VLLM_PKG/distributed/kv_transfer/kv_connector/v1/offloading/config.py"
        KVSGD="$VLLM_PKG/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
        MHYB="$VLLM_PKG/v1/worker/gpu/model_states/mamba_hybrid.py"
        SCHD="$VLLM_PKG/v1/core/sched/scheduler.py"
        mkdir -p "$SCRIPT_DIR/files/kvoffload/orig" "$SCRIPT_DIR/files/kv/orig"
        extract_from_image "$KVCFG" "$SCRIPT_DIR/files/kvoffload/orig/config.py"
        extract_from_image "$KVSGD" "$SCRIPT_DIR/files/kvoffload/orig/scheduler.py"
        extract_from_image "$MHYB" "$SCRIPT_DIR/files/kv/orig/mamba_hybrid.py"
        extract_from_image "$SCHD" "$SCRIPT_DIR/files/kv/orig/scheduler.py"
        if [[ "${KV_SKIP_GROUPS_PATCH:-0}" != "1" ]]; then
            python3 "$SCRIPT_DIR/files/patch_kv_offload_groups.py" || err "kv groups patch failed"
            add_overlay "$SCRIPT_DIR/files/kvoffload/config_patched.py" "$KVCFG"
            add_overlay "$SCRIPT_DIR/files/kvoffload/scheduler_patched.py" "$KVSGD"
        fi
        if [[ "${KV_SKIP_MAMBA_FIX:-0}" != "1" ]]; then
            python3 "$SCRIPT_DIR/files/patch_mamba_block_size.py" || err "mamba block-size patch failed"
            add_overlay "$SCRIPT_DIR/files/kv/mamba_hybrid.py" "$MHYB"
            add_overlay "$SCRIPT_DIR/files/kv/scheduler.py" "$SCHD"
        else
            warn "KV_SKIP_MAMBA_FIX=1 — align-mode state seeding stays BUGGY"
            warn "     (prefix hits can restore an all-zero mamba state). Only"
            warn "     for A/B experiments; never ship with it."
        fi
        _KVT_PROMPT_ONLY=true; [[ "$KV_PROMPT_ONLY" == "false" ]] && _KVT_PROMPT_ONLY=false
        # Hybrid state mode: restores need the boundary state per chunk, so
        # the arm defaults to align; none is possible but external hits
        # collapse (measured: 54/72 state files missing, hit = 0).
        [[ -n "$MAMBA_CACHE_MODE" ]] || MAMBA_CACHE_MODE="align"
        case "$MAMBA_CACHE_MODE" in align|none) ;; *) err "MAMBA_CACHE_MODE must be align or none (got '$MAMBA_CACHE_MODE')" ;; esac
        # State snapshots must align to the hash chunk (1664): the kernel's
        # native step (mamba_block_size=None -> ~5 chunks) leaves holes at
        # chunk 0, 5, 10... and every external hit collapses (measured:
        # exactly one state-file miss per request at chunk 0, five requests,
        # five boots). KV_MAMBA_BLOCK_SIZE=0 keeps the image default.
        KV_MAMBA_BLOCK_SIZE="${KV_MAMBA_BLOCK_SIZE:-1664}"
        [[ "$KV_MAMBA_BLOCK_SIZE" != "0" ]] && \
            VLLM_MAMBA_BLOCK=("--mamba-block-size" "$KV_MAMBA_BLOCK_SIZE") \
            || VLLM_MAMBA_BLOCK=()
        # MODEL_REVISION (optional) overrides the fence revision so a test or
        # recovery boot lands in its own namespace instead of adopting the
        # live one; default is the HF snapshot hash.
        KV_REVISION="${MODEL_REVISION:-$SNAP}"
        # Compact JSON (no spaces) wrapped in LITERAL single quotes: the
        # rendered launch scripts expand $VLLM_ARGS_STR as source, and the
        # quotes keep bash brace-expansion out of the commas (same pattern as
        # --speculative-config above).
        KV_XFER_JSON="'{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"NvmeDirectOffloadingSpec2\",\"spec_module_path\":\"kvoffload.nvme_direct2\",\"root_dir\":\"/mnt/fn-kv\",\"model_name\":\"${MODEL_ID}\",\"model_revision\":\"${KV_REVISION}\",\"capacity_bytes\":${KV_CAPACITY},\"n_io_threads\":${KV_IO_THREADS},\"offload_prompt_only\":${_KVT_PROMPT_ONLY},\"mamba_cache_mode\":\"${MAMBA_CACHE_MODE}\"}}'"
        ok "KV offload ON: root=$KV_ROOT (worker $KV_ROOT_WORKER), cap=${KV_CAPACITY_GIB} GiB, prompt_only=${_KVT_PROMPT_ONLY}"
        # The connector path costs ~11.5 GiB of post-capture consumption on
        # this box (boot 20260920T190603: consumed 54.86 vs 43.39 GiB without
        # the arm; the capture peak then dips host avail under the memwatch
        # floor and the watchdog kills the boot). Give the margin back out of
        # the KV pool: 0.835 - 0.055 = 0.78, the GMU the PLE disk tier already
        # measured good (11+ GiB host headroom). Set KV_GMU_DELTA=0 to keep
        # the .env GMU as-is.
        KV_GMU_DELTA="${KV_GMU_DELTA:-0.055}"
        if [[ "$GPU_MEMORY_UTILIZATION" == "0.835" && "$KV_GMU_DELTA" != "0" ]]; then
            GPU_MEMORY_UTILIZATION=$(python3 -c "print(f'{$GPU_MEMORY_UTILIZATION - $KV_GMU_DELTA:.3f}')")
            info "  KV arm: GMU -> $GPU_MEMORY_UTILIZATION (delta -$KV_GMU_DELTA; connector consumes ~11 GiB extra)"
        fi
    fi

    # ---------------------------------------------------------------------------
    # 7. Build vLLM args (shared between head and worker)
    # ---------------------------------------------------------------------------
    info "=== Step 7: Launch vLLM ==="

    VLLM_ARGS=()
    # Hybrid state mode rides with the KV arm (the flag must come after this
    # reinit; 6d resolved the knob and validated the value).
    if [[ "$KV_OFFLOAD" == "true" && -n "$MAMBA_CACHE_MODE" ]]; then
        VLLM_ARGS+=("--mamba-cache-mode" "$MAMBA_CACHE_MODE")
        VLLM_ARGS+=(${VLLM_MAMBA_BLOCK[@]+"${VLLM_MAMBA_BLOCK[@]}"})
    fi
    VLLM_ARGS+=("--enable-prompt-tokens-details")
    VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")
    VLLM_ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
    VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
    VLLM_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
    VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
    VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
    VLLM_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
    [[ -n "$MAMBA_SSM_CACHE_DTYPE" ]] && VLLM_ARGS+=("--mamba-ssm-cache-dtype" "$MAMBA_SSM_CACHE_DTYPE")
    VLLM_ARGS+=("--load-format" "safetensors")
    VLLM_ARGS+=("--safetensors-load-strategy" "lazy")
    VLLM_ARGS+=("--enable-chunked-prefill")
    VLLM_ARGS+=("--reasoning-parser" "qwen3")
    VLLM_ARGS+=("--enable-auto-tool-choice")
    if [[ -n "$CHAT_TEMPLATE" ]]; then
        [[ -r "$CHAT_TEMPLATE" ]] || err "CHAT_TEMPLATE=$CHAT_TEMPLATE is not readable"
        VLLM_ARGS+=("--chat-template" "/root/chat_template.jinja")
        # The froggeric template emits canonical XML tool calls; qwen3_coder
        # would not parse them.
        VLLM_ARGS+=("--tool-call-parser" "qwen3_xml")
    else
        VLLM_ARGS+=("--tool-call-parser" "qwen3_coder")
    fi
    VLLM_ARGS+=("--distributed-executor-backend" "mp")
    VLLM_ARGS+=("--mm-encoder-tp-mode" "$MM_ENCODER_TP_MODE")
    VLLM_ARGS+=("--nnodes" "2")
    VLLM_ARGS+=("--master-addr" "$HEAD_IP")
    VLLM_ARGS+=("--master-port" "$MASTER_PORT")

    if [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]]; then
        VLLM_ARGS+=("--enable-expert-parallel")
        VLLM_ARGS+=("--all2all-backend" "allgather_reducescatter")
    fi

    # JSON args: use printf to build properly quoted strings for the heredoc
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        # --async-scheduling with MTP > 0 silently corrupts n-grams (jschmied:
        # "no benchmark reveals it"). Match the bare flag after the word-split
        # and any "--async-scheduling=..." value.
        if [[ "$EXTRA_VLLM_ARGS" == *"--async-scheduling"* ]]; then
            err "EXTRA_VLLM_ARGS contains --async-scheduling while MTP is on: silent n-gram corruption (jschmied). Remove --async-scheduling."
        fi
        _SPEC_ARGMAX=""
        _SPEC_SCHED=""
        if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
            # get_top_tokens (added by patch_mtp_draft_vocab.py) is only reached
            # through this flag; it also cuts the draft all-gather from
            # O(vocab_size) to O(2*tp_size) per token.
            _SPEC_ARGMAX=',"use_local_argmax_reduction":true'
        fi
        if [[ -n "$MTP_K_SCHEDULE" ]]; then
            _SPEC_SCHED=",\"num_speculative_tokens_per_batch_size\":[$(
                printf '%s' "$MTP_K_SCHEDULE" | awk -F, '{
                    out=""
                    for (i = 1; i <= NF; i++) {
                        split($i, r, ":")
                        out = out (i > 1 ? "," : "") "[" r[1] "," r[2] "," r[3] "]"
                    }
                    printf "%s", out
                }')]"
        fi
        VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s%s%s}'" "$MTP_NUM_SPECULATIVE_TOKENS" "$_SPEC_SCHED" "$_SPEC_ARGMAX")")
    fi

    # CUDA graph capture sizes: "auto" derives every (1+K(S))*S width the
    # scheduler can build for S in 1..MAX_NUM_SEQS (same expression as the
    # single-Spark kit); a comma list passes through; empty keeps vLLM's default.
    _CG_SIZES="$CUDAGRAPH_CAPTURE_SIZES"
    if [[ "$_CG_SIZES" == "auto" ]]; then
        _CG_SIZES=$(
            _AUTO_MAX_SEQS="$MAX_NUM_SEQS" \
            _AUTO_K="$MTP_NUM_SPECULATIVE_TOKENS" \
            _AUTO_SCHED="$MTP_K_SCHEDULE" \
            python3 -c '
import os
max_seqs = int(os.environ["_AUTO_MAX_SEQS"])
k_default = int(os.environ["_AUTO_K"])
k_of = {}
for part in filter(None, os.environ["_AUTO_SCHED"].strip().split(",")):
    lo, hi, k = (int(x) for x in part.split(":"))
    for s in range(lo, min(hi, max_seqs) + 1):
        k_of.setdefault(s, min(k, k_default))
print(",".join(str(x) for x in sorted(
    {(1 + k_of.get(s, k_default)) * s for s in range(1, max_seqs + 1)})))
'
        )
    fi
    if [[ -n "$_CG_SIZES" ]]; then
        VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":%s,\"cudagraph_mode\":\"%s\",\"cudagraph_capture_sizes\":[%s]}'" "$COMPILATION_MODE" "$CUDAGRAPH_MODE" "$_CG_SIZES")")
    else
        VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":%s,\"cudagraph_mode\":\"%s\"}'" "$COMPILATION_MODE" "$CUDAGRAPH_MODE")")
    fi

    # hf-overrides: ONE merged payload, nested under "text_config".
    # vLLM's ModelConfig._apply_dict_overrides only recurses into keys that are
    # themselves nested configs. For qwen4_exp the parent config also exposes a
    # plain `rope_parameters` dict, so a top-level {"rope_parameters":...} is
    # setattr'd onto the parent and NEVER reaches text_config -- i.e. YaRN was
    # silently a no-op. Everything the model reads lives under text_config, so
    # nest both the rope override and the PLE dtype there.
    HF_OVERRIDES_JSON=$(
        PLE_DTYPE="$PLE_EMBEDDING_DTYPE" \
        YARN="$YARN_ENABLE" YARN_FACTOR="${YARN_FACTOR:-}" \
        python3 -c '
import json, os
tc = {}
if os.environ.get("PLE_DTYPE"):
    tc["ple_embedding_dtype"] = os.environ["PLE_DTYPE"]
if os.environ.get("YARN") == "true":
    tc["rope_parameters"] = {
        "rope_type": "yarn",
        "factor": float(os.environ["YARN_FACTOR"]),
        "original_max_position_embeddings": 262144,
    }
print(json.dumps({"text_config": tc}, separators=(",", ":")) if tc else "")
'
    )
    if [[ -n "$HF_OVERRIDES_JSON" ]]; then
        VLLM_ARGS+=("--hf-overrides" "'$HF_OVERRIDES_JSON'")
    fi

    # EXTRA_VLLM_ARGS is appended LAST and verbatim (quote JSON values with single
    # quotes exactly as you would on a shell command line).
    if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
        VLLM_ARGS+=("$EXTRA_VLLM_ARGS")
    fi
    # KV offload: the connector config rides VLLM_ARGS so both nodes' launch
    # scripts carry it (the JSON already wears its literal single quotes).
    if [[ "$KV_OFFLOAD" == "true" ]]; then
        VLLM_ARGS+=("--kv-transfer-config" "$KV_XFER_JSON")
    fi
    # One rendered string shared by the worker and head launch scripts. JSON
    # values already carry their own single quotes (see printf above).
    VLLM_ARGS_STR="${VLLM_ARGS[*]}"
    OVERLAY_ENV_STR="${OVERLAY_ENV[*]:-}"

    # Build docker run args (base, without node-specific VLLM_HOST_IP)
    DOCKER_ARGS=()
    DOCKER_ARGS+=(-d --name vllm-fn)
    DOCKER_ARGS+=(--gpus all --network host --ipc host)
    DOCKER_ARGS+=(--cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864)
    DOCKER_ARGS+=(--device /dev/infiniband:/dev/infiniband)
    # NCCL / fabric env (VLLM_HOST_IP set per-node below)
    DOCKER_ARGS+=(-e "GLOO_SOCKET_IFNAME=$IFACE")
    DOCKER_ARGS+=(-e "NCCL_SOCKET_IFNAME=$IFACE")
    DOCKER_ARGS+=(-e "TP_SOCKET_IFNAME=$IFACE")
    DOCKER_ARGS+=(-e "NCCL_IB_DISABLE=0")
    DOCKER_ARGS+=(-e "NCCL_IB_HCA=$IB_HCA")
    DOCKER_ARGS+=(-e "NCCL_IB_GID_INDEX=$IB_GID_INDEX")
    DOCKER_ARGS+=(-e "NCCL_IB_AUTO_DETECT=0")
    DOCKER_ARGS+=(-e "NCCL_DEBUG=WARN")
    # Offline mode
    DOCKER_ARGS+=(-e "HF_HUB_OFFLINE=1")
    DOCKER_ARGS+=(-e "TRANSFORMERS_OFFLINE=1")
    # YaRN: allow max_model_len > 262K
    if [[ -n "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" ]]; then
        DOCKER_ARGS+=(-e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN")
    fi
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        DOCKER_ARGS+=(-e "VLLM_PLE_CPU_OFFLOAD=1")
    fi
    # Volumes — single elements (flag + value together) for eval to parse correctly
    # NOTE: the container runs as root (HOME=/root), so cache mounts must target /root,
    # not the host user's $HOME — otherwise offline HF lookups fail.
    if [[ -n "$HEAD_PLE_MOUNT" ]]; then
        DOCKER_ARGS+=("$HEAD_PLE_MOUNT")
    fi
    if [[ -n "$HEAD_MODELOPT_MOUNT" ]]; then
        DOCKER_ARGS+=("$HEAD_MODELOPT_MOUNT")
    fi
    DOCKER_ARGS+=("-e HF_HOME=/root/.cache/huggingface")
    DOCKER_ARGS+=("-v $HF_CACHE_DIR:/root/.cache/huggingface")
    DOCKER_ARGS+=("-v $HOME/.cache/vllm:/root/.cache/vllm")
    if [[ -n "$EXTRA_DOCKER_ARGS" ]]; then
        # shellcheck disable=SC2206
        DOCKER_ARGS+=($EXTRA_DOCKER_ARGS)
    fi

    # -----------------------------------------------------------------------
    # Launch: worker (rank 1) first, then head (rank 0).
    # The head node serves the API; the worker runs headless.
    # -----------------------------------------------------------------------

    info ""
    info "Config:"
    info "  Model:      $MODEL_ID"
    if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        info "  ABLIT:      1 (gated Keys house QSA L3-47)"
    else
        info "  ABLIT:      $ABLIT"
    fi
    if [[ "$NFS_SHARE" == "true" ]]; then
        info "  Weights:    NFS from $NFS_SERVER_IP (head cache, no worker copy)"
    else
        info "  Weights:    local HF cache on each node (worker copy synced from head)"
    fi
    info "  Image:      $IMAGE"
    info "  Nodes:      $HEAD_IP (head, rank 0) + $WORKER_IP (worker, rank 1)"
    info "  TP=$TENSOR_PARALLEL_SIZE  EP=$( [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]] && echo on || echo off )  MTP=$MTP_NUM_SPECULATIVE_TOKENS"
    info "  Context:    $MAX_MODEL_LEN tokens"
    info "  GMU:        $GPU_MEMORY_UTILIZATION"
    info "  Max seqs:   $MAX_NUM_SEQS"
    info "  Port:       $PORT"
    info "  IFACE:      $IFACE"
    info "  IB_HCA:     $IB_HCA"
    info "  FP8 dense:  $FP8_DENSE   QSA profile: $QSA_PROFILE"
    info "  KV dtype:   $KV_CACHE_DTYPE   Draft vocab: ${MTP_DRAFT_VOCAB:-full}"
    info "  SSM state:  ${MAMBA_SSM_CACHE_DTYPE:-float32 (checkpoint)}"
    info "  MM encoder: $MM_ENCODER_TP_MODE tp-mode"
    [[ -n "$EXTRA_VLLM_ARGS" ]] && info "  Extra args: $EXTRA_VLLM_ARGS"
    info ""

    # ---- Worker (rank 1) ----
    info "--- Launching worker (rank 1) on $WORKER_IP ---"
    ssh_worker "docker rm -f vllm-fn >/dev/null 2>&1 || true"
    ssh_worker "mkdir -p '$REMOTE_HF' ~/.cache/vllm"
    if [[ "$NFS_SHARE" == "true" ]]; then
        nfs_ensure_worker_volume recreate
        if nfs_worker_has_model "hub/models--${ORG}--${NAME}"; then
            ok "Worker sees checkpoint over NFS"
        else
            err "WORKER cannot see hub/models--${ORG}--${NAME} over NFS. Check: docker logs $NFS_CONTAINER"
        fi
        WORKER_HF_MOUNT="-v $NFS_VOLUME:/root/.cache/huggingface:ro"
    else
        if ! ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
            err "WORKER is missing $REMOTE_HUB/models--${ORG}--${NAME}. Re-run ./start.sh without --launch to sync, or use --nfs."
        fi
        ok "Worker has a local checkpoint copy"
        WORKER_HF_MOUNT="-v $REMOTE_HF:/root/.cache/huggingface"
    fi

    # Worker can't mount head's filesystem — copy the patched file over
    if [[ -n "$WORKER_PLE_MOUNT" ]]; then
        info "  Copying PLE patch to worker..."
        PLE_DEST="/tmp/ple_layer_patched.py"
        scp -q "$PATCHED_PLE" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${PLE_DEST}"
    fi
    if [[ -n "$WORKER_MODELOPT_MOUNT" ]]; then
        info "  Copying MXFP8 fallback patch to worker..."
        scp -q "$PATCHED_MODELOPT" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/modelopt_patched.py"
    fi
    # Overlay files likewise: copy to /tmp/vllm-overlay on the worker
    WORKER_OVERLAY_MOUNTS=""
    if [[ ${#OVERLAY_FILES[@]} -gt 0 ]]; then
        info "  Copying ${#OVERLAY_FILES[@]} overlay file(s) to worker..."
        ssh_worker "mkdir -p /tmp/vllm-overlay"
        for entry in "${OVERLAY_FILES[@]}"; do
            host_file="${entry%%|*}"; container_path="${entry##*|}"
            scp -q "$host_file" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/vllm-overlay/$(basename "$host_file")"
            WORKER_OVERLAY_MOUNTS+=" -v /tmp/vllm-overlay/$(basename "$host_file"):$container_path:ro"
        done
    fi
    HEAD_OVERLAY_MOUNTS="${OVERLAY_MOUNTS[*]:-}"

    # PLE offload env flag (only set when explicitly true — avoids the ${VAR:+}
    # pitfall where "false" is non-empty and would wrongly enable the flag)
    PLE_OFFLOAD_ENV=""
    [[ "$PLE_OFFLOAD" == "true" ]] && PLE_OFFLOAD_ENV="-e VLLM_PLE_CPU_OFFLOAD=1"

    # Write worker launch script to a temp file and scp it (avoids SSH JSON quoting issues)
    WORKER_SCRIPT=$(mktemp /tmp/vllm_worker_XXXXXX.sh)
    cat > "$WORKER_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \
    -d --name vllm-fn \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -e GLOO_SOCKET_IFNAME=$WORKER_IFACE \
    -e NCCL_SOCKET_IFNAME=$WORKER_IFACE \
    -e TP_SOCKET_IFNAME=$WORKER_IFACE \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=$WORKER_IB_HCA \
    -e NCCL_IB_GID_INDEX=$IB_GID_INDEX \
    -e NCCL_IB_AUTO_DETECT=0 \
    -e NCCL_DEBUG=WARN \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=$WORKER_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    $KV_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    $WORKER_PLE_MOUNT \
    $WORKER_MODELOPT_MOUNT \
    $WORKER_OVERLAY_MOUNTS \
    $KV_WORKER_MOUNTS \
    $OVERLAY_ENV_STR \
    $WORKER_HF_MOUNT \
    -v $REMOTE_HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ID \
    $VLLM_ARGS_STR \
    --node-rank 1 \
    --headless
LAUNCH_EOF
    # No chmod here on purpose: mktemp already creates the file 0600. What used
    # to be `chmod +x` *loosened* that to 0711 under every umask below 077, and
    # the +x bit was never needed — the script is run as `bash <file>`, which
    # works fine on mode 0600. Matters as soon as EXTRA_VLLM_ARGS carries
    # something like `--api-key <key>` and the rendered script holds a secret.
    #
    # Feed it to the worker on stdin instead of scp'ing it to the fixed path
    # /tmp/vllm_worker_launch.sh: nothing is left on the worker to leak or to
    # clean up, there is no predictable /tmp name to pre-create as a symlink,
    # and ssh still reports the remote exit status, so `set -e` fails fast when
    # the worker's `docker run` fails instead of hanging in the health loop.
    info "  (starting worker container...)"
    worker_rc=0
    ssh_worker "bash -s" < "$WORKER_SCRIPT" || worker_rc=$?
    rm -f "$WORKER_SCRIPT"
    if (( worker_rc != 0 )); then
        err "Worker container failed to start (exit $worker_rc) — not launching the head."
    fi
    ok "Worker container started."
    info "  Waiting 15s for worker to initialize..."
    sleep 15

    # ---- Head (rank 0) ----
    info "--- Launching head (rank 0) on $HEAD_IP ---"
    ARCHIVE_TS=$(date '+%Y%m%dT%H%M%S')
    mkdir -p "$SCRIPT_DIR/logs/archive"
    # Keep the newest 20 sets, then drop the oldest (same rule as the single kit):
    # a set is a timestamp prefix with -container.log / -memwatch.log /
    # -timeout.log members.
    ls -1t "$SCRIPT_DIR"/logs/archive/*-container.log 2>/dev/null | tail -n +21 | while read -r f; do
        _set="${f%-container.log}"
        rm -f "${_set}-container.log" "${_set}-memwatch.log" "${_set}-timeout.log" 2>/dev/null || true
    done
    if docker inspect vllm-fn &>/dev/null; then
        docker logs --tail 3000 vllm-fn > "$SCRIPT_DIR/logs/archive/vllm-fn-${ARCHIVE_TS}-container.log" 2>&1 || true
        info "Previous container log archived: logs/archive/vllm-fn-${ARCHIVE_TS}-container.log"
    fi
    docker rm -f vllm-fn >/dev/null 2>&1 || true
    mkdir -p "$HOME/.cache/vllm"

    # Write head launch script (same approach as worker — avoids eval JSON issues)
    HEAD_SCRIPT=$(mktemp /tmp/vllm_head_XXXXXX.sh)
    cat > "$HEAD_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \
    -d --name vllm-fn \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -e GLOO_SOCKET_IFNAME=$IFACE \
    -e NCCL_SOCKET_IFNAME=$IFACE \
    -e TP_SOCKET_IFNAME=$IFACE \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=$IB_HCA \
    -e NCCL_IB_GID_INDEX=$IB_GID_INDEX \
    -e NCCL_IB_AUTO_DETECT=0 \
    -e NCCL_DEBUG=WARN \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=$HEAD_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    $KV_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    $HEAD_PLE_MOUNT \
    $HEAD_MODELOPT_MOUNT \
    $HEAD_OVERLAY_MOUNTS \
    $KV_HEAD_MOUNTS \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    -v $HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ID \
    $VLLM_ARGS_STR \
    --node-rank 0 \
    --host 0.0.0.0 \
    --port $PORT
LAUNCH_EOF
    # Same reasoning as the worker script above: mktemp's 0600 is already right,
    # so no chmod. The inspection copy below does need one — `cp` onto an
    # existing .last_head_launch.sh keeps that file's old mode, so on any tree
    # that ever ran the `chmod +x` version above it stays 0711 (verified).
    cp "$HEAD_SCRIPT" "$SCRIPT_DIR/.last_head_launch.sh"   # for inspection (gitignored)
    chmod 600 "$SCRIPT_DIR/.last_head_launch.sh"           # may contain --api-key values

    info "  (starting head container...)"
    bash "$HEAD_SCRIPT"
    rm -f "$HEAD_SCRIPT"
    ok "Head container started."

    # Host-memory watchdog (ported from the single-Spark kit): on unified
    # memory an exhausted pool hangs the kernel instead of raising an OOM, so
    # a poller stops the container when the host runs out of margin. Same
    # helper the supervisor calls, so the invocation cannot drift.
    mkdir -p "$SCRIPT_DIR/logs/archive"
    MEMWATCH_LOG="$SCRIPT_DIR/logs/memwatch-vllm-fn.log"
    if [[ -s "$MEMWATCH_LOG" ]]; then
        mv "$MEMWATCH_LOG" "$SCRIPT_DIR/logs/archive/vllm-fn-${ARCHIVE_TS}-memwatch.log" 2>/dev/null || true
    fi
    MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-2}" \
    MEMWATCH_FREE_GATE_GIB="${MEMWATCH_FREE_GATE_GIB:-10}" \
    MEMWATCH_GRACE="${MEMWATCH_GRACE:-30}" MEMWATCH_LOG="$MEMWATCH_LOG" \
        bash "$SCRIPT_DIR/scripts/start-memwatch.sh" vllm-fn "${MEMWATCH_MIN_GIB:-6}" >/dev/null
    ok "Watchdog running (stops the container when host margin collapses): logs/memwatch-vllm-fn.log"
    info ""
    info "vLLM is loading (~6-7 min). Following logs until ready..."
    info ""

    # Follow logs in background, poll /health until 200, then return to shell
    docker logs -f vllm-fn &
    LOGPID=$!

    info "Waiting for /health to return 200 (timeout ${READY_TIMEOUT_S}s)..."
    WAIT_START=$(date +%s)
    _last_hb=0
    while true; do
        sleep 10
        NOW=$(date +%s)
        ELAPSED=$((NOW - WAIT_START))
        if [[ "$ELAPSED" -gt "$READY_TIMEOUT_S" ]]; then
            kill $LOGPID 2>/dev/null || true
            echo ""
            err "Readiness timed out after ${ELAPSED}s (>READY_TIMEOUT_S=${READY_TIMEOUT_S})."
            err "Container was wedged before /health; archiving, removing, and exiting non-zero."
            docker logs vllm-fn > "$SCRIPT_DIR/logs/archive/vllm-fn-${ARCHIVE_TS}-timeout.log" 2>&1 || true
            docker rm -f vllm-fn >/dev/null 2>&1 || true
            exit 1
        fi
        # Check if container is still running
        if ! docker ps --format '{{.Names}}' | grep -q '^vllm-fn$'; then
            kill $LOGPID 2>/dev/null || true
            echo ""
            REASON=$(docker logs vllm-fn 2>&1 \
                     | grep -oE "(ValueError|RuntimeError|TimeoutError|torch\.[A-Za-z]*Error): .*" \
                     | grep -viE "min_frames|max_frames" | tail -1 | cut -c1-400)
            [[ -n "$REASON" ]] && { echo "  vLLM reported:"; echo "    $REASON"; }
            err "Container vllm-fn exited unexpectedly. Check: docker logs vllm-fn"
        fi
        # Check health endpoint
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            kill $LOGPID 2>/dev/null || true
            echo ""
            ok "vLLM is ready and serving on port $PORT (after ${ELAPSED}s)!"
            docker logs vllm-fn 2>&1 | grep -iE "GPU KV cache size|Available KV cache|Maximum concurrency" | tail -3 || true
            # Resuming after a manual stop clears the manual stopping flag: the
            # operator's own relaunch IS the resume (stop.sh's header promise).
            # A non-manual flag belongs to a maintenance window — leave it.
            if [[ -f "$SCRIPT_DIR/logs/stopping" && "$(head -n 1 "$SCRIPT_DIR/logs/stopping" 2>/dev/null)" == "manual" ]]; then
                rm -f "$SCRIPT_DIR/logs/stopping"
                info "manual stop flag cleared — supervisor resumes full supervision."
            fi
            info ""
            info "Test with:"
            info "  curl http://localhost:$PORT/v1/chat/completions \\"
            info "    -H 'Content-Type: application/json' \\"
            info "    -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
            info ""
            info "View logs: docker logs -f vllm-fn"
            info "Stop:      ./stop.sh"
            break
        fi
        if (( NOW - _last_hb >= 60 )); then
            _last_hb=$NOW
            echo "  ...waiting for readiness: ${ELAPSED}s elapsed, last /health code $HTTP_CODE"
        fi
    done
fi

ok "Done."
