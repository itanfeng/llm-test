#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ascend_remote_common.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/dev/ascend_remote_collect.sh [--run-id <id>]

Options:
  --run-id <id>    Collect specific run logs.
  -h, --help       Show this help.

Notes:
  1) If --run-id is not set, the latest remote run will be collected.
  2) Logs are copied to ${ASCEND_LOCAL_LOG_ROOT}/<run-id>.
EOF
}

run_id=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            run_id="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

require_cmd ssh
require_cmd rsync

if [[ -z "${run_id}" ]]; then
    run_id=$(ssh "${ASCEND_REMOTE_HOST}" bash -s -- "${ASCEND_REMOTE_LOG_ROOT}" <<'REMOTE_SCRIPT'
#!/bin/bash
set -euo pipefail

log_root="$1"
if [[ "${log_root}" != /* ]]; then
    log_root="${HOME}/${log_root}"
fi

if [[ ! -d "${log_root}" ]]; then
    exit 0
fi

latest_dir=$(ls -1dt "${log_root}"/* 2>/dev/null | head -n1 || true)
if [[ -n "${latest_dir}" ]]; then
    basename "${latest_dir}"
fi
REMOTE_SCRIPT
)
fi

if [[ -z "${run_id}" ]]; then
    echo "[ERROR] no remote logs found under ${ASCEND_REMOTE_LOG_ROOT}" >&2
    exit 1
fi

mkdir -p "${ASCEND_LOCAL_LOG_ROOT}"
local_run_dir="${ASCEND_LOCAL_LOG_ROOT}/${run_id}"
mkdir -p "${local_run_dir}"

remote_log_root="${ASCEND_REMOTE_LOG_ROOT}"
if [[ "${remote_log_root}" != /* && "${remote_log_root}" != "~/"* ]]; then
    remote_log_root="~/${remote_log_root#./}"
fi

echo "[INFO] Collecting remote logs: ${ASCEND_REMOTE_HOST}:${remote_log_root}/${run_id}/"
rsync -az -e ssh "${ASCEND_REMOTE_HOST}:${remote_log_root}/${run_id}/" "${local_run_dir}/"
echo "[INFO] Logs saved to ${local_run_dir}"
