#!/usr/bin/env bash
# stop.sh — Stop the vLLM container on both head and worker nodes.
#
# Stops gracefully by default (ported from the single-Spark kit): vLLM gets
# SIGTERM and a chance to unlink the POSIX shared-memory segments the PLE
# offload handshake allocates. The container runs with --ipc host, so anything
# it leaves behind leaks onto the host's /dev/shm and survives until reboot.
# Use --force to skip the wait.
#
# Touches logs/stopping while the stop is in progress so the supervisor waits
# instead of relaunching a container the human deliberately stopped. The flag
# carries a "manual" first line: the supervisor NEVER reclaims it (unlike a
# maintenance-window flag, which is reclaimed after STOPPING_MAX_AGE_S), so a
# manual stop stays down until you relaunch (start.sh, maintenance-relaunch.sh),
# remove logs/stopping, or reboot (which clears it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found."
    exit 1
fi

source .env

WORKER_USER="${WORKER_USER:-}"
WORKER_IP="${WORKER_IP:?WORKER_IP not set in .env}"
CONTAINER_NAME="vllm-fn"
NFS_CONTAINER="${NFS_CONTAINER:-vllm-fn-nfs}"
NFS_VOLUME="${NFS_VOLUME:-vllm-fn-hf}"
STOP_NFS=false
FORCE=false
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"      # seconds before docker escalates to SIGKILL
# Validate before use: docker stop -t rejects a bad value with exit 125, which
# the `|| true` below would swallow — the unconditional forced removal would
# then SIGKILL the container while the output still reads "stopped", silently
# downgrading the graceful path.
if ! [[ "$STOP_TIMEOUT" =~ ^[0-9]+$ ]]; then
    echo "STOP_TIMEOUT must be a non-negative integer (got: '$STOP_TIMEOUT')" >&2
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        --nfs|--all) STOP_NFS=true ;;
        -f|--force) FORCE=true ;;
        -h|--help)
            echo "Usage: $0 [--nfs] [--force]"
            echo "  (default)  Stop vLLM on worker then head (graceful SIGTERM)"
            echo "  --force    Skip the SIGTERM wait (immediate docker rm -f)"
            echo "  --nfs      Also stop the head NFS share and remove the worker volume"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (try --help)"
            exit 1
            ;;
    esac
done

ssh_cmd() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    ssh -o StrictHostKeyChecking=no "${user_prefix}$WORKER_IP" "$@"
}

# Stop the watchdog first so it cannot race a slow, graceful shutdown and turn
# it into a kill. pkill -f never matches its own process.
if pkill -f "[f]iles/memwatch.sh $CONTAINER_NAME" 2>/dev/null; then
    echo "watchdog stopped"
fi

# Signal the supervisor not to fight us: while this flag exists the supervisor
# holds off relaunching. A flag that ALREADY exists belongs to a maintenance
# window (or a concurrent stop): keep it — overwriting it with a manual marker
# would make an abandoned maintenance window un-reclaimable forever.
mkdir -p "$SCRIPT_DIR/logs"
if [[ ! -f "$SCRIPT_DIR/logs/stopping" ]]; then
    printf 'manual\n%s\n' "$(date -Is)" > "$SCRIPT_DIR/logs/stopping"
fi

# Worker first: the head is the API server; killing it first would let the
# worker sit idle-held on the NCCL group for its whole timeout.
echo "Stopping $CONTAINER_NAME on worker ($WORKER_IP)..."
if [[ -n "$(ssh_cmd "docker ps -aq -f 'name=^${CONTAINER_NAME}\$'")" ]]; then
    if [[ "$FORCE" == false ]]; then
        echo "  worker: stopping (SIGTERM, up to ${STOP_TIMEOUT}s)..."
        ssh_cmd "docker stop -t $STOP_TIMEOUT $CONTAINER_NAME >/dev/null 2>&1 || true"
    fi
    ssh_cmd "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true"
    echo "  Worker: stopped."
else
    echo "  Worker: not running."
fi

echo "Stopping $CONTAINER_NAME on head..."
if [[ -z "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]]; then
    echo "  Head: not running."
else
    # docker rm below discards the container's log; keep it for the post-mortem,
    # next to the watchdog's, the way start.sh and memwatch.sh do.
    ARCHIVE_DIR="$SCRIPT_DIR/logs/archive"; TS=$(date '+%Y%m%dT%H%M%S')
    mkdir -p "$ARCHIVE_DIR"
    docker logs --tail 3000 "$CONTAINER_NAME" > "$ARCHIVE_DIR/${CONTAINER_NAME}-${TS}-container.log" 2>&1 || true
    [[ -s "$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log" ]] \
        && cp -f "$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log" "$ARCHIVE_DIR/${CONTAINER_NAME}-${TS}-memwatch.log"
    echo "  archived logs to logs/archive/${CONTAINER_NAME}-${TS}-{container,memwatch}.log"
    # Keep newest 20 archive sets (same rule as start.sh and the supervisor).
    ls -1t "$ARCHIVE_DIR"/*-container.log 2>/dev/null | tail -n +21 | while read -r f; do
        _set="${f%-container.log}"
        rm -f "${_set}-container.log" "${_set}-memwatch.log" "${_set}-timeout.log" 2>/dev/null || true
    done
    if [[ "$FORCE" == false ]]; then
        echo "  head: stopping (SIGTERM, up to ${STOP_TIMEOUT}s)..."
        docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "  Head: stopped."
fi

# Report, but never delete: other containers on this host also run --ipc host,
# so their segments live here too and are not ours to remove.
leaked=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) 2>/dev/null | wc -l)
if (( leaked > 0 )); then
    bytes=$(find /dev/shm -maxdepth 1 \( -name 'psm_*' -o -name 'sem.mp-*' \) -printf '%s\n' 2>/dev/null \
            | awk '{s+=$1} END {print s+0}')
    echo "note: $leaked multiprocessing segment(s) in /dev/shm ($((bytes/1048576)) MiB allocated)."
    echo "      Inspect with: ls -la /dev/shm"
    echo "      Only remove them once no vLLM/sglang container is running."
fi

if $STOP_NFS; then
    echo "Stopping NFS share ($NFS_CONTAINER) on head..."
    echo "  (kernel NFS in Docker can ignore SIGKILL if rpcbind is in D-state; Ctrl-C and reboot if this hangs)"
    if timeout 15 docker rm -f "$NFS_CONTAINER" >/dev/null 2>&1; then
        echo "  NFS server: stopped."
    else
        echo "  NFS server: still running (could not kill). Leave it — start.sh will reuse it."
    fi
    echo "Removing worker NFS volume ($NFS_VOLUME)..."
    ssh_cmd "docker volume rm $NFS_VOLUME 2>/dev/null && echo '  Worker volume: removed.' || echo '  Worker volume: not present.'"
fi

echo "Done."
