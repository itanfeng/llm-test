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
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

ASCEND_REMOTE_CONFIG_DEFAULT="${SCRIPT_DIR}/ascend_remote.env"
ASCEND_REMOTE_CONFIG="${ASCEND_REMOTE_CONFIG:-${ASCEND_REMOTE_CONFIG_DEFAULT}}"

if [[ -f "${ASCEND_REMOTE_CONFIG}" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_REMOTE_CONFIG}"
fi

ASCEND_REMOTE_HOST="${ASCEND_REMOTE_HOST:-ascend-dev}"
ASCEND_REMOTE_REPO="${ASCEND_REMOTE_REPO:-/docker/tf/ops-transformer}"
ASCEND_REMOTE_CONTAINER="${ASCEND_REMOTE_CONTAINER:-tf}"
ASCEND_REMOTE_CONTAINER_REPO="${ASCEND_REMOTE_CONTAINER_REPO:-${ASCEND_REMOTE_REPO}}"
ASCEND_REMOTE_LOG_ROOT="${ASCEND_REMOTE_LOG_ROOT:-${ASCEND_REMOTE_REPO}/.codex-remote-logs}"
ASCEND_LOCAL_LOG_ROOT="${ASCEND_LOCAL_LOG_ROOT:-${REPO_ROOT}/.codex-remote-logs}"
ASCEND_CONTAINER_ENV_CMD="${ASCEND_CONTAINER_ENV_CMD:-}"
ASCEND_BUILD_CMD="${ASCEND_BUILD_CMD:-}"
ASCEND_TEST_CMD="${ASCEND_TEST_CMD:-}"

if [[ "${ASCEND_LOCAL_LOG_ROOT}" != /* ]]; then
    ASCEND_LOCAL_LOG_ROOT="${REPO_ROOT}/${ASCEND_LOCAL_LOG_ROOT#./}"
fi

timestamp_id() {
    date "+%Y%m%d-%H%M%S"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "[ERROR] command not found: ${cmd}" >&2
        exit 1
    fi
}

remote_bash() {
    local cmd="$1"
    ssh "${ASCEND_REMOTE_HOST}" "bash -lc $(printf '%q' "${cmd}")"
}

ensure_remote_repo_dir() {
    remote_bash "mkdir -p $(printf '%q' "${ASCEND_REMOTE_REPO}")"
}

ensure_remote_log_root() {
    remote_bash "mkdir -p $(printf '%q' "${ASCEND_REMOTE_LOG_ROOT}")"
}

print_runtime_config() {
    cat <<EOF
ASCEND_REMOTE_HOST=${ASCEND_REMOTE_HOST}
ASCEND_REMOTE_REPO=${ASCEND_REMOTE_REPO}
ASCEND_REMOTE_CONTAINER=${ASCEND_REMOTE_CONTAINER}
ASCEND_REMOTE_CONTAINER_REPO=${ASCEND_REMOTE_CONTAINER_REPO}
ASCEND_REMOTE_LOG_ROOT=${ASCEND_REMOTE_LOG_ROOT}
ASCEND_LOCAL_LOG_ROOT=${ASCEND_LOCAL_LOG_ROOT}
ASCEND_CONTAINER_ENV_CMD=${ASCEND_CONTAINER_ENV_CMD}
EOF
}
