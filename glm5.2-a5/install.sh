export http_proxy="http://p_atlas:proxy%40123@90.255.20.111:8888"
export https_proxy="$http_proxy"
export NO_PROXY=""  

# 保存当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${SCRIPT_DIR}/../../vllm-ascend}"
cd "${VLLM_ASCEND_ROOT}"
# 安装整个 vllm-ascend，日志保存到当前脚本目录
PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi \
PIP_TRUSTED_HOST=mirrors.huaweicloud.com \
pip install -v -e . --no-build-isolation 2>&1 | tee "${SCRIPT_DIR}/vllm-ascend-v0.23.0-install.log"
