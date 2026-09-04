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
  bash scripts/dev/ascend_remote_sync.sh [--no-delete]

Options:
  --no-delete     Do not delete files that were removed locally.
  -h, --help      Show this help.
EOF
}

DELETE_ENABLED=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-delete)
            DELETE_ENABLED=0
            shift
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

ensure_remote_repo_dir

rsync_args=(
    -az
    --exclude=/.git/
    --exclude=/.gitcode/
    --exclude=/.codex-remote-logs/
    --exclude=/build/
    --exclude=/build_out/
    --exclude=/output/
    --exclude=/pa-custom-op/
    --exclude=/.pytest_cache/
    --exclude=/.mypy_cache/
    --exclude=/.venv/
    --exclude=__pycache__/
    --exclude=.DS_Store
)

if [[ ${DELETE_ENABLED} -eq 1 ]]; then
    rsync_args+=(--delete)
fi

echo "[INFO] Syncing local repository to ${ASCEND_REMOTE_HOST}:${ASCEND_REMOTE_REPO}"
rsync "${rsync_args[@]}" -e ssh "${REPO_ROOT}/" "${ASCEND_REMOTE_HOST}:${ASCEND_REMOTE_REPO}/"
echo "[INFO] Sync finished"
