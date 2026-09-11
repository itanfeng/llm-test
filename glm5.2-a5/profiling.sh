#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-${SCRIPT_DIR}/../../vllm-ascend}"
MODEL_PATH="${MODEL_PATH:-/data/model/GLM-5.2-w4a4c8-mxfp4-l10}"
PREFETCH_MODE="${PREFETCH_MODE:-prefetch}"
PREFETCH_TOP_K="${PREFETCH_TOP_K:-2048}"
# HiCached coarse block Top-M, passed alongside --prefetch-top-k.
PREFETCH_HI_BLOCK_NUM="${PREFETCH_HI_BLOCK_NUM:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PREFILL_DEVICE="${PREFILL_DEVICE:-4}"
DECODE_DEVICE="${DECODE_DEVICE:-3}"
HOST_IP="${HOST_IP:-90.90.93.29}"
IFNAME="${IFNAME:-ens6f1}"
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-3}"
# Default workload: 12 concurrent requests using the 64K JSONL prompt.
BENCH="${BENCH:-1}"
BENCH_BATCH="${BENCH_BATCH:-12}"
# Keep requests alive while the other P/D handoffs reach the Decode engine.
# MTP width (4 queries) is independent of the response length (256 tokens).
BENCH_OUTPUT_TOKENS="${BENCH_OUTPUT_TOKENS:-16}"
BENCH_JSONL="${BENCH_JSONL:-examples/longbench_narrativeqa_64k.jsonl}"
# Must clear the default JSONL's 66068-token context plus the Decode tail.
BENCH_MAX_MODEL_LEN="${BENCH_MAX_MODEL_LEN:-66560}"
BENCH_PROFILE="${BENCH_PROFILE:-1}"
# Default to HiCached; select prefetch_li to use the single-stage indexer.
PREFETCH_INDEXER="${PREFETCH_INDEXER:-lightning_indexer_hi_cached}"
COHORT_KVGATHER="${COHORT_KVGATHER:-0}"
# This checkout implements the JSONL workload in the hhm probe.
if [[ "${BENCH}" == "1" ]]; then
    PROBE_SCRIPT="${PROBE_SCRIPT:-examples/dsa_offload_probe_hhm.sh}"
else
    PROBE_SCRIPT="${PROBE_SCRIPT:-examples/dsa_offload_probe.sh}"
fi

usage() {
    echo "Usage: $0 [offload|prefetch]" >&2
    echo "Both modes enable offload; prefetch additionally enables hidden-state prefetch." >&2
    echo "MODEL_PATH selects the model checkpoint; default: /data/model/GLM-5.2-w4a4c8-mxfp4-l10." >&2
    echo "PREFETCH_MODE may also be set through the environment; default: prefetch." >&2
    echo "PREFETCH_TOP_K controls the predicted Top-K width; default: 2048." >&2
    echo "PREFETCH_HI_BLOCK_NUM controls HiCached hi_block_num/topm only; default: 64." >&2
    echo "GPU_MEMORY_UTILIZATION controls the per-engine memory fraction; default: 0.90." >&2
    echo "MTP_SPECULATIVE_TOKENS defaults to 3 (1 main token + 3 draft tokens per request per step)." >&2
    echo "PREFILL_DEVICE=4, DECODE_DEVICE=3, HOST_IP=90.90.93.29, IFNAME=ens6f1 are overridable." >&2
    echo "VLLM_ASCEND_DIR defaults to the sibling vllm-ascend checkout." >&2
    echo "PROBE_SCRIPT is relative to VLLM_ASCEND_DIR, or an absolute path." >&2
    echo "  Default: examples/dsa_offload_probe_hhm.sh; BENCH=0: examples/dsa_offload_probe.sh." >&2
    echo "BENCH defaults to 1: long-prompt JSONL workload (batch=12, output=256)." >&2
    echo "  BENCH_JSONL selects the input; the first entry is duplicated BENCH_BATCH times." >&2
    echo "  The default 64K JSONL has 66068 prompt tokens; its input is used without truncation." >&2
    echo "  BENCH_MAX_MODEL_LEN defaults to 66560; BENCH_OUTPUT_TOKENS defaults to 256." >&2
    echo "  Concurrency is not a fixed scheduler batch; collected indexer shapes are checked after profiling." >&2
    echo "  BENCH_PROFILE defaults to 1 (runtime profiling and verification); set to 0 for throughput only." >&2
    echo "  BENCH=0 selects the short synthetic probe and always enables profiling and verification." >&2
    echo "PREFETCH_INDEXER=prefetch_li|lightning_indexer_hi_cached (default: lightning_indexer_hi_cached)." >&2
    echo "  plain and hicached/hi_cached are aliases; maps to prefetch_plain_indexer=true/false." >&2
    echo "  Applies when prefetch is enabled and the Indexer is non-C8." >&2
    echo "COHORT_KVGATHER defaults to 0; set to 1 to enable cohort KV Gather." >&2
    echo "OUTPUT_DIR defaults to profiling-w4/<mode>-<RUN_ID> next to this script." >&2
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

for name in BENCH BENCH_PROFILE COHORT_KVGATHER; do
    case "${!name}" in
        0|1) ;;
        *) echo "${name} must be 0 or 1, got ${!name}" >&2; exit 2 ;;
    esac
done
if ! [[ "${PREFETCH_HI_BLOCK_NUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREFETCH_HI_BLOCK_NUM must be a positive integer." >&2
    exit 2
fi
case "${PREFETCH_INDEXER}" in
    prefetch_li|plain) PREFETCH_INDEXER="prefetch_li" ;;
    lightning_indexer_hi_cached|hicached|hi_cached) PREFETCH_INDEXER="lightning_indexer_hi_cached" ;;
    *)
        echo "PREFETCH_INDEXER must be prefetch_li or lightning_indexer_hi_cached (aliases: plain/hicached/hi_cached), got ${PREFETCH_INDEXER}" >&2
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
            --prefetch-hi-block-num "${PREFETCH_HI_BLOCK_NUM}"
            --prefetch-indexer "${PREFETCH_INDEXER}"
        )
        ;;
    *)
        usage
        exit 2
        ;;
esac

# Keep cohort KV Gather independent of prefetch for offload/prefetch comparisons.
if [[ "${COHORT_KVGATHER}" == "1" ]]; then
    PREFETCH_ARGS+=(--enable-cohort-kvgather)
fi

VLLM_ASCEND_DIR="$(cd -- "${VLLM_ASCEND_DIR}" && pwd)"
if [[ "${PROBE_SCRIPT}" != /* ]]; then
    PROBE_SCRIPT="${VLLM_ASCEND_DIR}/${PROBE_SCRIPT}"
fi
if [[ ! -r "${PROBE_SCRIPT}" ]]; then
    echo "Probe script is not readable: ${PROBE_SCRIPT}" >&2
    exit 2
fi

# Check version-dependent switches before starting either model server.
PROBE_HELP="$(cd -- "${VLLM_ASCEND_DIR}" && bash "${PROBE_SCRIPT}" --help)"
require_probe_flag() {
    if [[ "${PROBE_HELP}" != *"$1"* ]]; then
        echo "${PROBE_SCRIPT} does not support $1; select a compatible PROBE_SCRIPT." >&2
        exit 2
    fi
}
if [[ "${BENCH}" == "1" ]]; then
    require_probe_flag --input-jsonl
    require_probe_flag --batch-size
fi
if [[ "${PREFETCH_MODE}" == "prefetch" ]]; then
    require_probe_flag --prefetch-indexer
    require_probe_flag --prefetch-hi-block-num
fi
if [[ "${COHORT_KVGATHER}" == "1" ]]; then
    require_probe_flag --enable-cohort-kvgather
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/profiling-w4/${RUN_MODE}-${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}"
# Preserve relative OUTPUT_DIR overrides when changing to the framework checkout.
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"

COMMON_ARGS=(
    --model "${MODEL_PATH}"
    --scenario pd
    --connector local-shm
    --io-backend kvgather_sim
    --cudagraph-mode FULL_DECODE_ONLY
    --host-ip "${HOST_IP}"
    --prefill-device "${PREFILL_DEVICE}"
    --decode-device "${DECODE_DEVICE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --ifname "${IFNAME}"
    --mtp-speculative-tokens "${MTP_SPECULATIVE_TOKENS}"
    --log-dir "${OUTPUT_DIR}"
)
PROFILE_ENABLED=1
if [[ "${BENCH}" == "1" ]]; then
    # The JSONL replaces the synthetic prompts; its first entry is repeated
    # BENCH_BATCH times and shares prefix KV through prefix caching.
    COMMON_ARGS+=(
        --input-jsonl "${BENCH_JSONL}"
        --batch-size "${BENCH_BATCH}"
        --max-model-len "${BENCH_MAX_MODEL_LEN}"
        --max-tokens "${BENCH_OUTPUT_TOKENS}"
    )
    PROFILE_ENABLED="${BENCH_PROFILE}"
else
    COMMON_ARGS+=(--max-tokens 4)
fi
if [[ "${PROFILE_ENABLED}" == "1" ]]; then
    COMMON_ARGS+=(--verify-path)
fi
# Bash 3 treats an empty array as unset under nounset.
if ((${#PREFETCH_ARGS[@]} > 0)); then
    COMMON_ARGS+=("${PREFETCH_ARGS[@]}")
fi

unset http_proxy
unset https_proxy

echo "Profiling offload=enabled, hidden_state_prefetch=${PREFETCH_ENABLED}, prefetch_top_k=${PREFETCH_TOP_K}"
echo "Decode mode: io_backend=kvgather_sim, cudagraph_mode=FULL_DECODE_ONLY, mtp_speculative_tokens=${MTP_SPECULATIVE_TOKENS}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Model: ${MODEL_PATH}"
echo "Framework: ${VLLM_ASCEND_DIR}"
echo "Probe script: ${PROBE_SCRIPT}"
echo "Devices: prefill=${PREFILL_DEVICE}, decode=${DECODE_DEVICE}; host_ip=${HOST_IP}, ifname=${IFNAME}"
echo "Prefetch indexer: ${PREFETCH_INDEXER}; cohort KV Gather: ${COHORT_KVGATHER}"
echo "HiCached hi_block_num/topm: ${PREFETCH_HI_BLOCK_NUM} (HiCached only)"
echo "Benchmark: ${BENCH}; runtime profiling and verification: ${PROFILE_ENABLED}"
if [[ "${BENCH}" == "1" ]]; then
    echo "Workload: input_jsonl=${BENCH_JSONL}, batch_size=${BENCH_BATCH}, max_model_len=${BENCH_MAX_MODEL_LEN}, max_tokens=${BENCH_OUTPUT_TOKENS}"
    echo "Target: ${BENCH_BATCH} requests x (1 + ${MTP_SPECULATIVE_TOKENS}) queries; max-tokens is the response length, not the MTP width."
fi
echo "Output: ${OUTPUT_DIR}"

cd "${VLLM_ASCEND_DIR}"
TMPDIR="${TMPDIR:-/data/pip-tmp}" \
VLLM_ASCEND_ENABLE_MLAPO=0 \
bash "${PROBE_SCRIPT}" \
    "${COMMON_ARGS[@]}"
