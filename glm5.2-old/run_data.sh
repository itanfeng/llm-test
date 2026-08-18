#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/../data}"

python3 hf_data.py \
    --count "${SAMPLE_COUNT:-10}" \
    --seed "${SEED:-42}" \
    --output "${PROMPTS_FILE:-${SCRIPT_DIR}/swe_prompts.jsonl}" \
    "$@"
