#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_MODE="${1:-prefetch}"
MODEL_TYPE="${2:-w4}"
BATCH_SIZE="${3:-12}"

case "${PROFILE_MODE}" in
    base|prefetch) ;;
    *)
        echo "Usage: $0 [base|prefetch] [w4|w8|w8-reduced] [batch-size]" >&2
        exit 2
        ;;
esac

case "${MODEL_TYPE}" in
    w4) MODEL_PATH="/data/model/GLM-5.1-w4a8" ;;
    w8) MODEL_PATH="/data/model/GLM-5.1-w8a8" ;;
    w8-reduced) MODEL_PATH="/data/model/GLM-5.1-w8a8-reduced" ;;
    *)
        echo "Usage: $0 [base|prefetch] [w4|w8|w8-reduced] [batch-size]" >&2
        exit 2
        ;;
esac

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "batch-size must be a positive integer" >&2
    exit 2
fi

OUTPUT_DIR="${SCRIPT_DIR}/profiling-${MODEL_TYPE}/${PROFILE_MODE}-bs${BATCH_SIZE}"

echo "Profiling model=${MODEL_TYPE}, mode=${PROFILE_MODE}, batch_size=${BATCH_SIZE}, output=${OUTPUT_DIR}"
msprof \
    --application="bash ${SCRIPT_DIR}/prof_glm.sh ${OUTPUT_DIR} ${SCRIPT_DIR}/offline_glm51.py ${MODEL_PATH} ${PROFILE_MODE} ${BATCH_SIZE}" \
    --output="${OUTPUT_DIR}"
