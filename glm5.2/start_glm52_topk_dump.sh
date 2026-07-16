#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-/data/model/GLM-5.2-w8a8}"
PORT="${PORT:-8077}"
TOPK_DIR="${GLM52_TOPK_DIR:-data}"

export VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"

export GLM52_TOPK_DIR="$TOPK_DIR"
export GLM52_TOPK_TOKENS="${GLM52_TOPK_TOKENS:-1}"
export GLM52_TOPK_HEADS="${GLM52_TOPK_HEADS:-4}"
export GLM52_TOPK_K="${GLM52_TOPK_K:-2048}"
export GLM52_TOPK_TP_RANK="${GLM52_TOPK_TP_RANK:-0}"
export GLM52_TOPK_PRINT="${GLM52_TOPK_PRINT:-0}"

mkdir -p "$TOPK_DIR"

echo "Starting GLM-5.2; dump directory: $TOPK_DIR"
exec vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name glm-52 \
    --max-num-seqs 48 \
    --enforce-eager \
    --max-model-len 70000 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --additional-config '{"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
    "$@"
