# GLM-5.2 DSA 测试脚本

本目录基于 `vllm-ascend/examples/dsa_sparse` 的 GLM-5.2 测试配置，沿用
`glm5.1` 目录的离线脚本与 msprof 包装方式。

## 性能测试

默认测试 W4A8、Host Memory、BS=12：

```bash
bash profiling.sh
```

也可以显式选择模式、模型和 batch size：

```bash
bash profiling.sh host prefetch w4 12
bash profiling.sh host offload w4 12
bash profiling.sh full-hbm prefetch w4 1
bash profiling.sh host w4 12
```

`full-hbm` 仅支持 BS=1。结果保存在
`profiling-<model>/<mode>-bs<batch-size>/`，其中 `result.json` 包含 decode
吞吐、平均耗时、P50/P95/P99 和输出 token。

常用环境变量：

```bash
GLM52_DATASET=/data/datasets/longbench/data/longbench_narrativeqa_64k.jsonl
GLM52_W4_MODEL=/data/model/GLM-5.2-w4a8
GLM52_W8_MODEL=/data/model/GLM-5.2-w8a8
GLM52_MAX_MODEL_LEN=65536
GLM52_WARMUP_STEPS=5
GLM52_TIMED_STEPS=20
```

如需采集指定 decode 区间的 torch-npu profiling，可直接运行：

```bash
python offline_glm52.py \
  --mode host_memory \
  --output-dir /data/results/glm52 \
  --batch-size 12 \
  --profile-dir /data/results/glm52/torch-profiler \
  --profile-steps 20
```

## EP 精度回归

```bash
python validate_glm52_ep_accuracy.py --model /data/model/GLM-5.2-w4a8
```

## 安装当前 vllm-ascend checkout

```bash
bash install.sh
```

如果 `vllm-ascend` 不在 `llm-test` 同级目录，可通过
`VLLM_ASCEND_ROOT=/path/to/vllm-ascend` 覆盖。


## 切换vllm-ascend分支后要同步切换third_party的commit
```bash
git switch dsa-sparse-0.23-graph-prefetch
git submodule sync --recursive
git submodule update --init --recursive --checkout
git submodule status csrc/third_party/catlass
git -C csrc/third_party/catlass rev-parse HEAD
```