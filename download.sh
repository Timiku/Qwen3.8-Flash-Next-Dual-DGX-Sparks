#!/usr/bin/env bash
# ============================================================================
# download.sh — Fetch HuggingFace weights onto the HEAD node only.
#
# The worker never gets a local copy. After this, ./start.sh --launch (or
# ./start-fp8.sh --launch) exports the head cache over NFS on ConnectX, or
# start.sh rsyncs the worker copy (default).
#
# Usage:
#   ./download.sh                 # MODEL_ID from .env (stock NVFP4, ABLIT=0)
#   ABLIT=1 ./download.sh         # gated Keys house QSA L3-47 (~same size as stock)
#                                 # Set HF_TOKEN, then accept the terms on
#                                 # https://huggingface.co/drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47
#   ./download.sh --fp8           # Qwen/Qwen3.8-Flash-Next-FP8
#   ./download.sh org/repo        # explicit HuggingFace repo
#   HF_TOKEN=hf_... ./download.sh # for a gated repo
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

ABLIT_MODEL_ID="drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47"
ABLIT_PAGE="https://huggingface.co/${ABLIT_MODEL_ID}"
FP8_MODEL_ID="Qwen/Qwen3.8-Flash-Next-FP8"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            sed -n '3,16p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    echo "  cp .env.sample .env"
    exit 1
fi

# Environment wins over .env for ABLIT / HF_TOKEN (same rule as start.sh).
_CLI_HF_TOKEN="${HF_TOKEN:-}"
_CLI_ABLIT="${ABLIT:-}"
_CLI_VERIFY_SHA256="${VERIFY_SHA256:-}"
# shellcheck source=.env
source .env
[[ -n "$_CLI_HF_TOKEN" ]] && HF_TOKEN="$_CLI_HF_TOKEN"
HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN
[[ -n "$_CLI_ABLIT" ]] && ABLIT="$_CLI_ABLIT"
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" == "0" || "$ABLIT" == "1" ]] || err "ABLIT must be 0 or 1 (got: '$ABLIT')"
VERIFY_SHA256="${_CLI_VERIFY_SHA256:-${VERIFY_SHA256:-1}}"
[[ "$VERIFY_SHA256" == "0" || "$VERIFY_SHA256" == "1" ]] || err "VERIFY_SHA256 must be 0 or 1 (got: '$VERIFY_SHA256')"

MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
EXPLICIT_MODEL=""
for arg in "$@"; do
    case "$arg" in
        --fp8) EXPLICIT_MODEL="$FP8_MODEL_ID" ;;
        -*) err "Unknown argument: $arg (try --help)" ;;
        *)  EXPLICIT_MODEL="$arg" ;;
    esac
done

if [[ -n "$EXPLICIT_MODEL" ]]; then
    MODEL_ID="$EXPLICIT_MODEL"
    if [[ "$ABLIT" == "1" && "$MODEL_ID" != "$ABLIT_MODEL_ID" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: MODEL_ID=$MODEL_ID"
    fi
elif [[ "$ABLIT" == "1" ]]; then
    MODEL_ID="$ABLIT_MODEL_ID"
fi

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
HUB_PATH="$HF_CACHE_DIR/hub"
export HF_HUB_CACHE="$HUB_PATH"
mkdir -p "$HUB_PATH"

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"

gated_fail() {
    err "403: this repo is gated.
  Accept the terms on that page:
    $ABLIT_PAGE
  Then set HF_TOKEN and retry:
    HF_TOKEN=hf_... ABLIT=1 ./download.sh"
}

require_ablit_token() {
    err "ABLIT=1 requires HF_TOKEN — this repo is gated.
  1. Uncomment HF_TOKEN in .env (or: export HF_TOKEN=hf_...)
  2. Open $ABLIT_PAGE
  3. Accept the terms on that page
  4. ABLIT=1 ./download.sh"
}

next_start_hint() {
    if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        info "Next: ABLIT=1 ./start.sh (or set ABLIT=1 in .env and run ./start.sh)"
    else
        info "Next: ./start.sh"
    fi
}

# Prints a snapshot hash. Exit 0 = complete, 1 = incomplete, 2 = none.
resolve_snapshot() {
    python3 "$SCRIPT_DIR/files/resolve_snapshot.py" "$1"
}

info "Downloading $MODEL_ID"
info "Head cache: $HF_CACHE_DIR"
info "Worker:     not updated (NFS from head at launch, or rsync via start.sh)"

# Already complete? Require every shard named by the safetensors index. A
# config.json appears early in a partial download and is not sufficient.
# A complete cache still runs the sha256 guard below: "downloaded" and
# "verified" are different claims, and a blob corrupted on disk after the
# download is only caught by re-hashing it.
DO_DL=true
if [[ -d "$MODEL_PATH" ]]; then
    SNAP=""
    SNAP_RC=0
    SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
    if [[ "$SNAP_RC" -eq 0 && -n "$SNAP" ]]; then
        ok "Already in cache: $MODEL_PATH ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
        DO_DL=false
    else
        if [[ -n "$SNAP" ]]; then
            warn "Partial download found (snapshot $SNAP); resuming."
        fi
    fi
fi

if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    info "This checkpoint is gated: $ABLIT_PAGE"
    if [[ -z "$HF_TOKEN" ]]; then
        require_ablit_token
    fi
    info "Using HF_TOKEN for the gated download. A 403 means the terms are not accepted yet."
fi

DL_PY='
import os, sys
from huggingface_hub import snapshot_download
cache_dir = sys.argv[3] if len(sys.argv) > 3 else None
try:
    p = snapshot_download(
        repo_id=sys.argv[1],
        token=(os.environ.get("HF_TOKEN") or None),
        cache_dir=cache_dir,
        max_workers=4,
    )
except Exception as e:
    name = type(e).__name__
    text = str(e)
    code = getattr(getattr(e, "response", None), "status_code", None)
    gated = (
        name == "GatedRepoError"
        or code in (401, 403)
        or "gated" in text.lower()
        or " 403" in f" {text}"
    )
    if gated:
        retry = sys.argv[2] if len(sys.argv) > 2 else "HF_TOKEN=hf_... ./download.sh"
        print(
            "403: this repo is gated.\n"
            "Accept the terms on that page:\n"
            f"  https://huggingface.co/{sys.argv[1]}\n"
            "Then set HF_TOKEN and retry:\n"
            f"  {retry}",
            file=sys.stderr,
        )
        sys.exit(1)
    raise
print(p)
'

if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    RETRY_HINT="HF_TOKEN=hf_... ABLIT=1 ./download.sh"
else
    RETRY_HINT="HF_TOKEN=hf_... ./download.sh $MODEL_ID"
fi

cli_download_or_die() {
    local rc=0 log
    log=$(mktemp)
    # shellcheck disable=SC2064
    trap 'rm -f "$log"' RETURN
    set +e
    if command -v uvx &>/dev/null; then
        info "Using uvx..."
        HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HUB_PATH" \
            uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    elif command -v huggingface-cli &>/dev/null; then
        info "Using huggingface-cli..."
        HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HUB_PATH" \
            huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    elif command -v hf &>/dev/null; then
        info "Using hf CLI..."
        HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HUB_PATH" \
            hf download "$MODEL_ID" --cache-dir "$HUB_PATH" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    else
        set -e
        err "No HuggingFace download tool found. Install one of:\n  pip install huggingface_hub\n  pip install uv"
    fi
    set -e
    if [[ "$rc" -ne 0 ]]; then
        if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]] && \
           grep -qiE 'gated|403|401|cannot access|GatedRepoError' "$log"; then
            gated_fail
        fi
        err "Download failed (exit $rc)"
    fi
}
if [[ "$DO_DL" == "true" ]]; then
    info "Downloading (resumable; interrupt and rerun to continue)..."
    if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]] && python3 -c "import huggingface_hub" 2>/dev/null; then
        HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HUB_PATH" HF_TOKEN="$HF_TOKEN" \
            python3 -c "$DL_PY" "$MODEL_ID" "$RETRY_HINT" "$HUB_PATH"
    else
        cli_download_or_die
    fi
fi

[[ -d "$MODEL_PATH" ]] || err "Download finished but $MODEL_PATH was not found"
SNAP=""
SNAP_RC=0
SNAP="$(resolve_snapshot "$MODEL_PATH")" && SNAP_RC=0 || SNAP_RC=$?
[[ -n "$SNAP" ]] || err "No snapshot directory under $MODEL_PATH/snapshots"
if [[ "$SNAP_RC" -ne 0 ]]; then
    if [[ "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
        err "Snapshot is missing one or more indexed weight shards — the download is incomplete.
  Resume with:  ABLIT=1 ./download.sh"
    fi
    err "Snapshot is missing one or more indexed weight shards — the download is incomplete.
  Resume with:  ./download.sh $MODEL_ID"
fi

# ---------------------------------------------------------------------------
# sha256 verification of LFS blobs (ported from the single-Spark kit). aria2
# and the hub CLIs preallocate to final size, so size checks pass on corrupt
# content; this is the only catch. VERIFY_SHA256=0 documents a skip but is not
# the default.
# ---------------------------------------------------------------------------
if [[ "$VERIFY_SHA256" == "1" ]]; then
    SNAP_DIR="$MODEL_PATH/snapshots/$SNAP"
    STATE_FILE="$MODEL_PATH/.sha256state"
    auth=(); [[ -n "$HF_TOKEN" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    # Fetches the full LFS metadata manifest, walking the HF tree API with
    # pagination (50 entries/page via the Link: rel="next" header). The
    # manifest is <size>\t<lfs.oid>\t<path> per line, so a truncated fetch
    # is visible in the entry count printed below.
    fetch_manifest() {  # <cache-file>
        local cache="$1"
        rm -f "$cache"
        local page=0 total=0
        local url="https://huggingface.co/api/models/$MODEL_ID/tree/main?recursive=true&expand=true&limit=50"
        while [[ -n "$url" ]]; do
            local headers="$cache.headers.$page"
            local body; body=$(curl -s -D "$headers" -m 60 "${auth[@]}" \
                -H "Accept: application/json" "$url" || true)
            [[ -n "$body" ]] || { rm -f "$cache" "$headers"; return 1; }
            local bodyf; bodyf=$(mktemp)
            printf '%s' "$body" > "$bodyf"
            python3 - "$cache" "$bodyf" <<'PY'
import json, sys
cache = sys.argv[1]
with open(cache, "a") as mf:
    for e in json.loads(open(sys.argv[2]).read()):
        if e.get("type") != "file":
            continue
        lfs = e.get("lfs", {})
        # lfs.oid IS the sha256 (matches the blob name HF writes into the
        # cache); the tree API has no lfs.sha256 key.
        print(f"{e.get('size', 0)}\t{lfs.get('oid', '')}\t{e.get('path', '')}", file=mf)
PY
            rm -f "$bodyf"
            total=$(grep -c . "$cache" 2>/dev/null || echo 0)
            page=$((page + 1))
            url=""
            if [[ -f "$headers" ]]; then
                url=$(tr -d '\r' < "$headers" | grep -i '^Link:' \
                    | sed -nE 's/.*<([^>]*)>; rel="next".*/\1/p' || true)
                rm -f "$headers"
            fi
            [[ -n "$url" ]] && url=$(printf '%s' "$url" | tr -d '[:space:]')
        done
        [[ -s "$cache" ]] || return 1
        echo "$total"
        return 0
    }

    MANIFEST=$(mktemp)
    ENTRY_COUNT=0
    ENTRY_COUNT=$(fetch_manifest "$MANIFEST" 2>/dev/null || echo "")
    if [[ -z "$ENTRY_COUNT" ]]; then
        warn "sha256 verification: could not fetch the LFS manifest (offline?); skipping verification."
        warn "     Rerun with network to get the corrupt-blob guard."
    else
        info "verifying sha256: ${ENTRY_COUNT} files in the remote tree manifest"
        {
            # Fewer remote entries than local LFS-sized files means the
            # pagination walk came up short (jschmied: 50 of 144 "verified"
            # cleanly).
            _LOCAL_LFS=$(find "$SNAP_DIR" -type f -size +1M 2>/dev/null | wc -l)
            if (( ENTRY_COUNT < _LOCAL_LFS )); then
                err "tree manifest records ${ENTRY_COUNT} files but the snapshot has ${_LOCAL_LFS} LFS-sized files — the manifest is truncated (pagination regression). Aborting verification."
            fi
            _resume=1
            if grep -q '^complete ' "$STATE_FILE" 2>/dev/null; then
                _resume=0
                { : > "$STATE_FILE"; } 2>/dev/null || true
            fi
            while IFS=$'\t' read -r _sz sha path; do
                # Manifest-controlled path must stay inside the snapshot: reject
                # traversal and anything that is not a plain relative path.
                case "$path" in
                    *".."*|/*|*[[:space:]]*|*[^[:print:]]*|"")
                        err "sha256 verify: manifest path '$path' is not a safe relative path; aborting."
                        ;;
                esac
                f="$SNAP_DIR/$path"
                [[ -f "$f" ]] || err "sha256 verify: $path missing from snapshot"
                _done=0
                if [[ "$_resume" == "1" && -f "$STATE_FILE" ]]; then
                    grep -qxF "$sha  $path" "$STATE_FILE" 2>/dev/null && _done=1
                fi
                if [[ "$_done" != "1" ]]; then
                    _have=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1 || echo "")
                    if [[ "$_have" != "$sha" ]]; then
                        err "sha256 mismatch on $path (got ${_have:-no-file}, want $sha). The checkpoint is corrupt or incomplete; remove $(readlink -f "$f") and rerun ./download.sh $MODEL_ID to fetch it again."
                    fi
                    if ! { printf '%s  %s\n' "$sha" "$path" >> "$STATE_FILE"; } 2>/dev/null; then
                        if [[ "${_state_warned:-0}" != "1" ]]; then
                            warn "cannot write the sha256 resume state ($STATE_FILE); verification still complete, but the next run will re-hash every blob."
                            _state_warned=1
                        fi
                    fi
                fi
            done < <(awk -F'\t' 'NF >= 3 && length($2) == 64 { print }' "$MANIFEST")
            { printf 'complete %s\n' "$SNAP" >> "$STATE_FILE"; } 2>/dev/null || true
            ok "sha256 verified all LFS blobs in snapshot $SNAP."
        }
    fi
    rm -f "$MANIFEST"
fi

ok "Cache ready: $MODEL_PATH ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
next_start_hint
