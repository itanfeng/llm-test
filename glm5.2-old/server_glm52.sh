#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基础配置
# =========================

MODEL_PATH="${MODEL_PATH:-/data/model/GLM-5.2-w8a8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8077}"

DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-8}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-72000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10240}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-52}"

# =========================
# Ascend / vLLM 环境变量
# =========================

export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-1}"
export VLLM_ASCEND_ENABLE_MLAPO="${VLLM_ASCEND_ENABLE_MLAPO:-1}"

export VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# =========================
# GLM-5.2 Top-K dump 配置
# =========================

TOPK_DIR="${GLM52_TOPK_DIR:-${SCRIPT_DIR}/pt}"

export GLM52_TOPK_DIR="$TOPK_DIR"

# 默认只让 TP rank 0 写文件，避免所有 rank 同时 dump
export GLM52_TOPK_TP_RANK="${GLM52_TOPK_TP_RANK:-0}"

mkdir -p "$TOPK_DIR"

# =========================
# 日志配置
# =========================

LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/online_glm52_server.log}"
mkdir -p "$(dirname -- "$LOG_FILE")"

echo "========================================"
echo "Starting GLM-5.2 service"
echo "Model:                 $MODEL_PATH"
echo "Address:               $HOST:$PORT"
echo "DP size:               $DP_SIZE"
echo "TP size:               $TP_SIZE"
echo "Expected NPU count:    $((DP_SIZE * TP_SIZE))"
echo "Max model length:      $MAX_MODEL_LEN"
echo "Max sequences:         $MAX_NUM_SEQS"
echo "Max batched tokens:    $MAX_NUM_BATCHED_TOKENS"
echo "Execution mode:        eager"
echo "Top-K dump directory:  $TOPK_DIR"
echo "Top-K TP rank:         $GLM52_TOPK_TP_RANK"
echo "Log file:              $LOG_FILE"
echo "========================================"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --data-parallel-size "$DP_SIZE" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enforce-eager \
    --additional-config '{
        "fuse_muls_add": true,
        "multistream_overlap_shared_expert": true
    }' \
    "$@" \
    2>&1 | tee "$LOG_FILE"
