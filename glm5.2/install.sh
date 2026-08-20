#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${SCRIPT_DIR}/../../vllm-ascend}"

cd "${VLLM_ASCEND_ROOT}"
PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi \
PIP_TRUSTED_HOST=mirrors.huaweicloud.com \
python -m pip install -v -e . 2>&1 | tee vllm-ascend-v0.23.0-install.log
