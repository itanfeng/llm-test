#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_MODE="${1:-host}"
MODEL_TYPE="${2:-w4}"
BATCH_SIZE="${3:-12}"

usage() {
    echo "Usage: $0 [full-hbm|host] [w4|w8] [batch-size]" >&2
}

case "${PROFILE_MODE}" in
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

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "batch-size must be a positive integer" >&2
    exit 2
fi
if [[ "${PROFILE_MODE}" == "full-hbm" && "${BATCH_SIZE}" -ne 1 ]]; then
    echo "full-hbm mode only supports batch-size 1" >&2
    exit 2
fi

OUTPUT_DIR="${SCRIPT_DIR}/profiling-${MODEL_TYPE}/${PROFILE_MODE}-bs${BATCH_SIZE}"

echo "Profiling model=${MODEL_TYPE}, mode=${PROFILE_MODE}, batch_size=${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
APPLICATION="bash ${SCRIPT_DIR}/prof_glm.sh"
APPLICATION+=" ${OUTPUT_DIR} ${SCRIPT_DIR}/offline_glm52.py"
APPLICATION+=" ${MODEL_PATH} ${PROFILE_MODE} ${BATCH_SIZE}"
msprof \
    --application="${APPLICATION}" \
    --output="${OUTPUT_DIR}"
