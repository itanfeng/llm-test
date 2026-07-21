#!/bin/bash
# Run HiSparse synthetic GLM-5.1 profiling for multiple batch sizes and save
# each run's output to a separate log file.

# 修改位置：1. MAX_TOKENS由2048改为20， 相应MAX_MODEL_LEN改为66000
# 2. 修改num_samples的循环范围为1（原为1 2 4 8）
# 3. 进入容器 docker exec -it huanghongming bash
# 4. 在当前目录运行 bash profiling.sh
# profiling.sh 会将 msprof 数据和运行日志的共用输出目录作为第一个参数传入。


set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <profiling-output-dir> <offline-script>" >&2
    exit 2
fi

MODEL="/data/model/GLM-5.1-w8a8-reduced/"
#MODEL="/data/model/GLM-5.1-w8a8/"
INPUT_LEN=65536
# MAX_MODEL_LEN=68000
# MAX_TOKENS=2048
MAX_MODEL_LEN=66000
MAX_TOKENS=200
TP=4
OUTPUT_DIR="$1"
PYTHON_SCRIPT="$2"

# 只使用后8张NPU（设备编号8-15），与TP=8保持一致
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

mkdir -p "${OUTPUT_DIR}"

failed_runs=()

# for num_samples in 1 2 4 8; do
for num_samples in 4; do
    log_file="${OUTPUT_DIR}/glm_ns${num_samples}.log"
    echo "Running num_samples=${num_samples}, max_num_seqs=${num_samples}, output=${log_file}"
    if python "${PYTHON_SCRIPT}" \
        --model "${MODEL}" \
        --num-samples "${num_samples}" \
        --max-num-seqs "${num_samples}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-tokens "${MAX_TOKENS}" \
        --output-dir "${OUTPUT_DIR}" \
        --tp "${TP}" \
        --gpu-memory-utilization 0.94 \
        2>&1 | tee "${log_file}"; then
        # --input-len "${INPUT_LEN}" \
        # > "${log_file}" 2>&1; then
        echo "Finished num_samples=${num_samples}"
    else
        echo "FAILED num_samples=${num_samples}, see ${log_file}"
        failed_runs+=("${num_samples}")
    fi
done

if [ ${#failed_runs[@]} -eq 0 ]; then
    echo "All runs completed. Logs are in ${OUTPUT_DIR}/"
else
    echo "Some runs failed: ${failed_runs[*]}"
    exit 1
fi
