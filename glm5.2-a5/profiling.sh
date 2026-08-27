#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_DIR="$(cd -- "${SCRIPT_DIR}/../../vllm-ascend" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/g00809126/repos/glm-moe-dsa}"
PREFETCH_MODE="${PREFETCH_MODE:-offload}"
PREFETCH_TOP_K="${PREFETCH_TOP_K:-2048}"

usage() {
    echo "Usage: $0 [offload|prefetch]" >&2
    echo "Both modes enable offload; prefetch additionally enables hidden-state prefetch." >&2
    echo "MODEL_PATH selects the model checkpoint; default: /home/g00809126/repos/glm-moe-dsa." >&2
    echo "PREFETCH_MODE may also be set through the environment; default: offload." >&2
    echo "PREFETCH_TOP_K controls the predicted Top-K width; default: 2048." >&2
}

case "$#" in
    0) ;;
    1)
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            *) PREFETCH_MODE="$1" ;;
        esac
        ;;
    *)
        usage
        exit 2
        ;;
esac

case "${PREFETCH_MODE}" in
    offload)
        RUN_MODE="offload"
        PREFETCH_ENABLED="false"
        PREFETCH_ARGS=()
        ;;
    prefetch)
        RUN_MODE="prefetch"
        PREFETCH_ENABLED="true"
        PREFETCH_ARGS=(
            --enable-prefetch-with-hidden-states
            --prefetch-top-k "${PREFETCH_TOP_K}"
        )
        ;;
    *)
        usage
        exit 2
        ;;
esac

RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/profiling-w4/${RUN_MODE}-${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}"

unset http_proxy
unset https_proxy

echo "Profiling offload=enabled, hidden_state_prefetch=${PREFETCH_ENABLED}, prefetch_top_k=${PREFETCH_TOP_K}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"

cd "${VLLM_ASCEND_DIR}"
VLLM_ASCEND_ENABLE_MLAPO=0 \
bash examples/dsa_sparse_pd_mock_probe.sh \
    --model "${MODEL_PATH}" \
    --host-ip 90.90.93.29 \
    --prefill-device 4 \
    --decode-device 5 \
    --max-tokens 4 \
    --ifname ens6f1 \
    --local-shm \
    --mtp-speculative-tokens 2 \
    --verify-path \
    --log-dir "${OUTPUT_DIR}" \
    "${PREFETCH_ARGS[@]}"
