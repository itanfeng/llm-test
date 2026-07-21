#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILING_OUTPUT_DIR="${SCRIPT_DIR}/profiling/glm-base"
OFFLINE_SCRIPT="${SCRIPT_DIR}/offline_glm51.py"

msprof \
    --application="bash ${SCRIPT_DIR}/prof_glm.sh ${PROFILING_OUTPUT_DIR} ${OFFLINE_SCRIPT}" \
    --output="${PROFILING_OUTPUT_DIR}"
