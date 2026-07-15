docker run -itd \
  --name tf-vllm-ascend-8.5.1 \
  --network host \
  --ipc=host \
  --privileged \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  --device=/dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /usr/local/openmpi:/usr/local/openmpi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
  -v /docker:/docker \
  -v /data:/data \
  -w /data/tf \
  quay.io/ascend/vllm-ascend:v0.18.0-a3 \
  sleep infinity

