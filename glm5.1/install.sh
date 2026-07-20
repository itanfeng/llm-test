cd ../../vllm-ascend
# 安装整个vllm-ascend
PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple \
PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi \
PIP_TRUSTED_HOST=mirrors.huaweicloud.com \
pip install -v -e .