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
  bash scripts/dev/ascend_remote_cycle.sh [options]

Options:
  --build-cmd <cmd>   Build command running inside container.
  --test-cmd <cmd>    Test command running inside container.
  --skip-build        Skip build step.
  --skip-test         Skip test step.
  --run-id <id>       Custom run id. Default: timestamp.
  --no-delete         Pass to sync step (do not delete remote removed files).
  -h, --help          Show this help.

Behavior:
  1) sync local repo to remote
  2) run remote build/test in container
  3) collect run logs back to local
EOF
}

run_id="$(timestamp_id)"
build_cmd=""
test_cmd=""
skip_build=0
skip_test=0
sync_no_delete=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-cmd)
            build_cmd="$2"
            shift 2
            ;;
        --test-cmd)
            test_cmd="$2"
            shift 2
            ;;
        --skip-build)
            skip_build=1
            shift
            ;;
        --skip-test)
            skip_test=1
            shift
            ;;
        --run-id)
            run_id="$2"
            shift 2
            ;;
        --no-delete)
            sync_no_delete=1
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

sync_args=()
if [[ ${sync_no_delete} -eq 1 ]]; then
    sync_args+=(--no-delete)
fi

run_args=(--run-id "${run_id}")
if [[ -n "${build_cmd}" ]]; then
    run_args+=(--build-cmd "${build_cmd}")
fi
if [[ -n "${test_cmd}" ]]; then
    run_args+=(--test-cmd "${test_cmd}")
fi
if [[ ${skip_build} -eq 1 ]]; then
    run_args+=(--skip-build)
fi
if [[ ${skip_test} -eq 1 ]]; then
    run_args+=(--skip-test)
fi

echo "[INFO] Step 1/3 sync"
if [[ ${#sync_args[@]} -gt 0 ]]; then
    bash "${SCRIPT_DIR}/ascend_remote_sync.sh" "${sync_args[@]}"
else
    bash "${SCRIPT_DIR}/ascend_remote_sync.sh"
fi

echo "[INFO] Step 2/3 run"
set +e
bash "${SCRIPT_DIR}/ascend_remote_run.sh" "${run_args[@]}"
run_rc=$?
set -e

echo "[INFO] Step 3/3 collect"
set +e
bash "${SCRIPT_DIR}/ascend_remote_collect.sh" --run-id "${run_id}"
collect_rc=$?
set -e

if [[ ${collect_rc} -ne 0 ]]; then
    echo "[WARN] collect step failed (rc=${collect_rc})"
fi

echo "[INFO] Cycle finished: run_id=${run_id}, run_rc=${run_rc}"
exit "${run_rc}"
