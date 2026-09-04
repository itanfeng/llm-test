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
  bash scripts/dev/ascend_remote_run.sh [options]

Options:
  --build-cmd <cmd>   Build command running inside container.
  --test-cmd <cmd>    Test command running inside container.
  --skip-build        Skip build step.
  --skip-test         Skip test step.
  --run-id <id>       Custom run id. Default: timestamp.
  -h, --help          Show this help.

Notes:
  1) Build/test command priority: CLI args > ascend_remote.env > empty.
  2) At least one of build/test must be provided and not skipped.
  3) Logs are stored in ${ASCEND_REMOTE_LOG_ROOT}/<run-id> on remote host.
EOF
}

build_cmd="${ASCEND_BUILD_CMD}"
test_cmd="${ASCEND_TEST_CMD}"
run_id="$(timestamp_id)"

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
            build_cmd=""
            shift
            ;;
        --skip-test)
            test_cmd=""
            shift
            ;;
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

if [[ -z "${build_cmd}" && -z "${test_cmd}" ]]; then
    echo "[ERROR] build/test command are both empty" >&2
    echo "        set ASCEND_BUILD_CMD / ASCEND_TEST_CMD in ascend_remote.env," >&2
    echo "        or pass --build-cmd / --test-cmd on command line." >&2
    exit 1
fi

ensure_remote_log_root

echo "[INFO] Running on remote host ${ASCEND_REMOTE_HOST}, run_id=${run_id}"

host_repo_escaped=$(printf '%q' "${ASCEND_REMOTE_REPO}")
container_name_escaped=$(printf '%q' "${ASCEND_REMOTE_CONTAINER}")
container_repo_escaped=$(printf '%q' "${ASCEND_REMOTE_CONTAINER_REPO}")
log_root_escaped=$(printf '%q' "${ASCEND_REMOTE_LOG_ROOT}")
run_id_escaped=$(printf '%q' "${run_id}")
build_cmd_escaped=$(printf '%q' "${build_cmd}")
test_cmd_escaped=$(printf '%q' "${test_cmd}")
env_cmd_escaped=$(printf '%q' "${ASCEND_CONTAINER_ENV_CMD}")

remote_env_prefix="HOST_REPO=${host_repo_escaped} "
remote_env_prefix+="CONTAINER_NAME=${container_name_escaped} "
remote_env_prefix+="CONTAINER_REPO=${container_repo_escaped} "
remote_env_prefix+="LOG_ROOT=${log_root_escaped} "
remote_env_prefix+="RUN_ID=${run_id_escaped} "
remote_env_prefix+="BUILD_CMD=${build_cmd_escaped} "
remote_env_prefix+="TEST_CMD=${test_cmd_escaped} "
remote_env_prefix+="ENV_CMD=${env_cmd_escaped}"

set +e
ssh "${ASCEND_REMOTE_HOST}" "${remote_env_prefix} bash -s" <<'REMOTE_SCRIPT'
#!/bin/bash
set -euo pipefail

host_repo="${HOST_REPO}"
container_name="${CONTAINER_NAME}"
container_repo="${CONTAINER_REPO}"
log_root="${LOG_ROOT}"
run_id="${RUN_ID}"
build_cmd="${BUILD_CMD}"
test_cmd="${TEST_CMD}"
env_cmd="${ENV_CMD}"

resolve_path() {
    local raw="$1"
    if [[ "${raw}" == /* ]]; then
        printf '%s' "${raw}"
    else
        printf '%s/%s' "${HOME}" "${raw}"
    fi
}

host_repo_abs=$(resolve_path "${host_repo}")
log_root_abs=$(resolve_path "${log_root}")
container_repo_home_abs=$(resolve_path "${container_repo}")
container_repo_root_abs="${container_repo}"
if [[ "${container_repo_root_abs}" != /* ]]; then
    container_repo_root_abs="/${container_repo_root_abs}"
fi

log_dir="${log_root_abs}/${run_id}"
mkdir -p "${log_dir}"

status_file="${log_dir}/status.env"
: > "${status_file}"

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

run_step() {
    local step="$1"
    local step_cmd="$2"
    local log_file="${log_dir}/${step}.log"

    {
        echo "[INFO] $(timestamp) step=${step} begin"
        echo "[INFO] host_repo=${host_repo_abs}"
        echo "[INFO] container_repo_candidate_1=${container_repo_home_abs}"
        echo "[INFO] container_repo_candidate_2=${container_repo_root_abs}"
        echo "[INFO] container_repo_candidate_3=${container_repo}"
        echo "[INFO] command=${step_cmd}"
    } | tee -a "${log_file}"

    set +e
    docker exec \
        -e OPS_REPO_PRIMARY="${container_repo_home_abs}" \
        -e OPS_REPO_SECONDARY="${container_repo_root_abs}" \
        -e OPS_REPO_RAW="${container_repo}" \
        -e OPS_ENV_CMD="${env_cmd}" \
        -e OPS_CMD="${step_cmd}" \
        "${container_name}" \
        bash -lc '
            set -euo pipefail
            if [[ -d "${OPS_REPO_PRIMARY}" ]]; then
                workdir="${OPS_REPO_PRIMARY}"
            elif [[ -d "${OPS_REPO_SECONDARY}" ]]; then
                workdir="${OPS_REPO_SECONDARY}"
            elif [[ -d "${OPS_REPO_RAW}" ]]; then
                workdir="${OPS_REPO_RAW}"
            else
                echo "[ERROR] repository path not found in container"
                echo "[ERROR] tried: ${OPS_REPO_PRIMARY}"
                echo "[ERROR] tried: ${OPS_REPO_SECONDARY}"
                echo "[ERROR] tried: ${OPS_REPO_RAW}"
                exit 2
            fi
            cd "${workdir}"
            if [[ -n "${OPS_ENV_CMD}" ]]; then
                eval "${OPS_ENV_CMD}"
            fi
            eval "${OPS_CMD}"
        ' \
        2>&1 | tee -a "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e

    echo "[INFO] $(timestamp) step=${step} end rc=${rc}" | tee -a "${log_file}"
    printf '%s_RC=%s\n' "$(echo "${step}" | tr '[:lower:]' '[:upper:]')" "${rc}" >> "${status_file}"
    return "${rc}"
}

overall_rc=0
build_rc=0
test_rc=0

if [[ -n "${build_cmd}" ]]; then
    set +e
    run_step "build" "${build_cmd}"
    build_rc=$?
    set -e
    if [[ ${build_rc} -ne 0 ]]; then
        overall_rc=${build_rc}
    fi
else
    echo "BUILD_RC=SKIPPED" >> "${status_file}"
fi

if [[ ${overall_rc} -eq 0 && -n "${test_cmd}" ]]; then
    set +e
    run_step "test" "${test_cmd}"
    test_rc=$?
    set -e
    if [[ ${test_rc} -ne 0 ]]; then
        overall_rc=${test_rc}
    fi
elif [[ -z "${test_cmd}" ]]; then
    echo "TEST_RC=SKIPPED" >> "${status_file}"
else
    echo "TEST_RC=SKIPPED_BUILD_FAILED" >> "${status_file}"
fi

{
    echo "RUN_ID=${run_id}"
    echo "REMOTE_LOG_DIR=${log_dir}"
    echo "OVERALL_RC=${overall_rc}"
} >> "${status_file}"

echo "[INFO] run_id=${run_id}"
echo "[INFO] remote_log_dir=${log_dir}"
echo "[INFO] overall_rc=${overall_rc}"

exit "${overall_rc}"
REMOTE_SCRIPT
run_rc=$?
set -e

echo "[INFO] Remote run finished with rc=${run_rc}, run_id=${run_id}"
exit "${run_rc}"
