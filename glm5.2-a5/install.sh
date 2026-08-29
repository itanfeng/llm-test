Ip_vpn='90.254.44.238'

export http_proxy="http://p_atlas:proxy%40123@${Ip_vpn}:6688"
export https_proxy="${http_proxy}"
export NO_PROXY=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${SCRIPT_DIR}/../../vllm-ascend}"

cd "${VLLM_ASCEND_ROOT}" || exit 1

pip install -v -e . \
    --no-build-isolation \
    --progress-bar on \
    2>&1 | tee "${SCRIPT_DIR}/vllm-ascend-v0.23.0-install.log"