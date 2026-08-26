unset http_proxy
unset https_proxy

VLLM_ASCEND_ENABLE_MLAPO=0 \
bash examples/dsa_sparse_pd_mock_probe.sh \
    --model /home/g00809126/repos/glm-moe-dsa \
    --host-ip 90.90.93.29 \
    --prefill-device 4 \
    --decode-device 5 \
    --max-tokens 4 \
    --ifname ens6f1 \
    --local-shm \
    --mtp-speculative-tokens 2 \
    --verify-path \
    --log-dir /data/autotriton/sekd/dsa-sparse-023-eager-mtp/vllm-ascend/examples/log