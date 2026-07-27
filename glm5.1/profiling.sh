#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_SCRIPT="${SCRIPT_DIR}/offline_glm51.py"
PROFILE_MODE="${1:-prefetch}"

case "${PROFILE_MODE}" in
    base)
        PROFILING_OUTPUT_DIR="${SCRIPT_DIR}/profiling/base-bs16"
        PROFILE_SCRIPT="${SCRIPT_DIR}/prof_glm_base.sh"
        ;;
    prefetch)
        PROFILING_OUTPUT_DIR="${SCRIPT_DIR}/profiling/prefetch-bs16"
        PROFILE_SCRIPT="${SCRIPT_DIR}/prof_glm_prefetch.sh"
        ;;
    *)
        echo "Usage: $0 [base|prefetch]" >&2
        exit 2
        ;;
esac

msprof \
    --application="bash ${PROFILE_SCRIPT} ${PROFILING_OUTPUT_DIR} ${OFFLINE_SCRIPT}" \
    --output="${PROFILING_OUTPUT_DIR}"
