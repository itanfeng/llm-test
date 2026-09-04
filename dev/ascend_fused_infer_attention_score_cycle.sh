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

usage() {
    cat <<'EOF'
Usage:
  bash scripts/dev/ascend_fused_infer_attention_score_cycle.sh [options]

Options:
  --soc <soc>                Default: ascend910b
  --vendor-name <name>       Default: custom
  --op <op_name>             Default: fused_infer_attention_score
  --install-path <path>      Default: $(pwd)/pa-custom-op (evaluated in container)
  --run-id <id>              Pass through to ascend_remote_cycle.sh
  --no-delete                Pass through to sync step
  -h, --help                 Show this help.

Behavior:
  1) source /usr/local/Ascend/cann/set_env.sh
  2) bash build.sh --pkg --soc=<soc> --vendor_name=<vendor> --ops=<op>
  3) ./build/cann-ops-transformer-<vendor>_linux-aarch64.run --install-path=<path>
  4) execute with sync -> run -> collect
EOF
}

soc="ascend910b"
vendor_name="custom"
op_name="fused_infer_attention_score"
install_path='$(pwd)/pa-custom-op'

cycle_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc)
            soc="$2"
            shift 2
            ;;
        --vendor-name)
            vendor_name="$2"
            shift 2
            ;;
        --op)
            op_name="$2"
            shift 2
            ;;
        --install-path)
            install_path="$2"
            shift 2
            ;;
        --run-id)
            cycle_args+=(--run-id "$2")
            shift 2
            ;;
        --no-delete)
            cycle_args+=(--no-delete)
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

build_cmd="source /usr/local/Ascend/cann/set_env.sh && "
build_cmd+="bash build.sh --pkg --soc=${soc} --vendor_name=${vendor_name} --ops=${op_name} && "
build_cmd+="./build/cann-ops-transformer-${vendor_name}_linux-aarch64.run --install-path=${install_path}"

if [[ ${#cycle_args[@]} -gt 0 ]]; then
    bash "${SCRIPT_DIR}/ascend_remote_cycle.sh" "${cycle_args[@]}" --skip-test --build-cmd "${build_cmd}"
else
    bash "${SCRIPT_DIR}/ascend_remote_cycle.sh" --skip-test --build-cmd "${build_cmd}"
fi
