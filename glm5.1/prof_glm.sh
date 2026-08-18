#!/bin/bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <output-dir> <offline-script> <model-path> <offload|prefetch> <batch-size>" >&2
    exit 2
fi

OUTPUT_DIR="$1"
OFFLINE_SCRIPT="$2"
MODEL_PATH="$3"
PREFETCH_MODE="$4"
NUM_SAMPLES="$5"

case "${PREFETCH_MODE}" in
    offload) PREFETCH_FLAG="--no-enable-prefetch-with-hidden-states" ;;
    prefetch) PREFETCH_FLAG="--enable-prefetch-with-hidden-states" ;;
    base) PREFETCH_FLAG="--no-enable-prefetch-with-hidden-states" ;;
    *) exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python "${OFFLINE_SCRIPT}" \
    --model "${MODEL_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --max-num-seqs "${NUM_SAMPLES}" \
    --max-model-len 68000 \
    --max-tokens 64 \
    --input-len 0 \
    --output-dir "${OUTPUT_DIR}" \
    --tp 16 \
    --gpu-memory-utilization 0.94 \
    --enable-segment-sfa \
    "${PREFETCH_FLAG}" \
    2>&1 | tee "${OUTPUT_DIR}/glm_bs${NUM_SAMPLES}.log"
