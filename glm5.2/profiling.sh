#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CACHE_MODE="host"
PREFETCH_MODE="offload"
MODEL_TYPE="w4"
BATCH_SIZE="12"

active_npu_processes() {
    npu-smi info | awk -F '|' '
        $2 ~ /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*$/ &&
        $3 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            gsub(/[[:space:]]+/, "/", $2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4)
            printf "NPU/chip=%s pid=%s process=%s\n", $2, $3, $4
        }
    '
}

ensure_npus_are_idle() {
    local active_processes
    active_processes="$(active_npu_processes)"
    if [[ -n "${active_processes}" ]]; then
        echo "ERROR: GLM-5.2 profiling requires exclusive access to all 16 NPU dies." >&2
        echo "The following NPU processes are already running:" >&2
        printf '%s\n' "${active_processes}" >&2
        echo "Stop or coordinate the other workload, then rerun this script." >&2
        return 1
    fi
}

usage() {
    echo "Usage: $0 [full-hbm|host] [w4|w8] [batch-size]" >&2
    echo "       $0 [full-hbm|host] [offload|prefetch] [w4|w8] [batch-size]" >&2
    echo "The second form explicitly sets PREFETCH_MODE (default is offload)." >&2
}

case "$#" in
    0)
        ;;
    1)
        CACHE_MODE="$1"
        ;;
    2)
        usage
        exit 2
        ;;
    3)
        CACHE_MODE="$1"
        MODEL_TYPE="$2"
        BATCH_SIZE="$3"
        ;;
    4)
        CACHE_MODE="$1"
        PREFETCH_MODE="$2"
        MODEL_TYPE="$3"
        BATCH_SIZE="$4"
        ;;
    *)
        usage
        exit 2
        ;;
esac

case "${CACHE_MODE}" in
    full-hbm|host) ;;
    *)
        usage
        exit 2
        ;;
esac

case "${MODEL_TYPE}" in
    w4) MODEL_PATH="${GLM52_W4_MODEL:-/data/model/GLM-5.2-w4a8}" ;;
    w8) MODEL_PATH="${GLM52_W8_MODEL:-/data/model/GLM-5.2-w8a8}" ;;
    *)
        usage
        exit 2
        ;;
esac

case "${PREFETCH_MODE}" in
    offload|prefetch) ;;
    *)
        usage
        exit 2
        ;;
esac

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "batch-size must be a positive integer" >&2
    exit 2
fi
if [[ "${CACHE_MODE}" == "full-hbm" && "${BATCH_SIZE}" -ne 1 ]]; then
    echo "full-hbm mode only supports batch-size 1" >&2
    exit 2
fi

OUTPUT_DIR="${SCRIPT_DIR}/profiling-${MODEL_TYPE}/${CACHE_MODE}-${PREFETCH_MODE}-bs${BATCH_SIZE}"

ensure_npus_are_idle

echo "Profiling model=${MODEL_TYPE}, cache=${CACHE_MODE}, prefetch=${PREFETCH_MODE}, batch_size=${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
APPLICATION="bash ${SCRIPT_DIR}/prof_glm.sh"
APPLICATION+=" ${OUTPUT_DIR} ${SCRIPT_DIR}/offline_glm52.py"
APPLICATION+=" ${MODEL_PATH} ${CACHE_MODE} ${PREFETCH_MODE} ${BATCH_SIZE}"
msprof \
    --application="${APPLICATION}" \
    --output="${OUTPUT_DIR}"
