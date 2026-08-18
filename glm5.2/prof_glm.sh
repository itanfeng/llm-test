#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <output-dir> <offline-script> <model-path> <full-hbm|host> <batch-size>" >&2
    exit 2
fi

OUTPUT_DIR="$1"
OFFLINE_SCRIPT="$2"
MODEL_PATH="$3"
PROFILE_MODE="$4"
BATCH_SIZE="$5"

case "${PROFILE_MODE}" in
    full-hbm) DSA_MODE="full_hbm" ;;
    host) DSA_MODE="host_memory" ;;
    *)
        echo "profile mode must be full-hbm or host" >&2
        exit 2
        ;;
esac

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "batch-size must be a positive integer" >&2
    exit 2
fi
if [[ "${DSA_MODE}" == "full_hbm" && "${BATCH_SIZE}" -ne 1 ]]; then
    echo "full-hbm mode only supports batch-size 1" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT="600"
export HCCL_EXEC_TIMEOUT="3600"
export HCCL_CONNECT_TIMEOUT="3600"
export HCCL_BUFFSIZE="200"
export OMP_PROC_BIND="false"
export OMP_NUM_THREADS="1"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

python "${OFFLINE_SCRIPT}" \
    --mode "${DSA_MODE}" \
    --model "${MODEL_PATH}" \
    --dataset "${GLM52_DATASET:-/data/datasets/longbench/data/longbench_narrativeqa_64k.jsonl}" \
    --output-dir "${OUTPUT_DIR}" \
    --result "${OUTPUT_DIR}/result.json" \
    --batch-size "${BATCH_SIZE}" \
    --max-num-seqs "${BATCH_SIZE}" \
    --max-model-len "${GLM52_MAX_MODEL_LEN:-65536}" \
    --warmup-steps "${GLM52_WARMUP_STEPS:-5}" \
    --timed-steps "${GLM52_TIMED_STEPS:-20}" \
    --tp "${GLM52_TP:-16}" \
    --gpu-memory-utilization "${GLM52_GPU_MEMORY_UTILIZATION:-0.92}" \
    --max-num-batched-tokens "${GLM52_MAX_NUM_BATCHED_TOKENS:-8192}" \
    2>&1 | tee "${OUTPUT_DIR}/glm_bs${BATCH_SIZE}.log"
