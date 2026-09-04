# Ascend Remote Dev Loop

This folder provides a local-edit + remote-verify loop for Ascend NPU operator development.

## 1) Prepare config

```bash
cp scripts/dev/ascend_remote.env.example scripts/dev/ascend_remote.env
```

Defaults already match your current setup:
- `ASCEND_REMOTE_HOST=ascend-dev`
- `ASCEND_REMOTE_REPO=/docker/tf/ops-transformer`
- `ASCEND_REMOTE_CONTAINER=tf`
- `ASCEND_REMOTE_CONTAINER_REPO=/docker/tf/ops-transformer`

Set your build/test commands in `scripts/dev/ascend_remote.env`, or pass them on CLI.

## 2) Scripts

- `ascend_remote_sync.sh`
  - Sync local repository to remote with `rsync`.
- `ascend_remote_run.sh`
  - Run build/test on remote server inside container (`docker exec tf-ops bash -lc ...`).
  - Save remote logs to `${ASCEND_REMOTE_LOG_ROOT}/<run-id>/`.
- `ascend_remote_collect.sh`
  - Pull logs from remote to local `${ASCEND_LOCAL_LOG_ROOT}/<run-id>/`.
- `ascend_remote_cycle.sh`
  - One command loop: `sync -> run -> collect`.

## 3) Common commands

Dedicated loop for `fused_infer_attention_score` (recommended for repeated iteration):

```bash
bash scripts/dev/ascend_fused_infer_attention_score_cycle.sh
```

This command fixes your repeated workflow into one entry:
- `source /usr/local/Ascend/cann/set_env.sh`
- `bash build.sh --pkg --soc=ascend910b --vendor_name=custom --ops=fused_infer_attention_score`
- `./build/cann-ops-transformer-custom_linux-aarch64.run --install-path=$(pwd)/pa-custom-op`

Dedicated compile + decode test loop for `fused_infer_attention_score`:

```bash
bash scripts/dev/ascend_fused_infer_attention_score_decode_cycle.sh
```

This command runs:
- compile + install package
- source CANN env
- source `./pa-custom-op/vendors/custom_transformer/bin/set_env.bash`
- `python ./scripts/dev/fia_mla_decode.py`

Run a full loop with one command:

```bash
bash scripts/dev/ascend_remote_cycle.sh \
  --build-cmd "bash build.sh --ophost --opkernel --soc=ascend910b -j16" \
  --test-cmd "bash build.sh --ophost_test --soc=ascend910b"
```

Only sync:

```bash
bash scripts/dev/ascend_remote_sync.sh
```

Run remote commands only:

```bash
bash scripts/dev/ascend_remote_run.sh \
  --build-cmd "bash build.sh --ophost --opkernel --soc=ascend910b -j16" \
  --test-cmd "bash build.sh --ophost_test --soc=ascend910b"
```

Collect the latest run logs:

```bash
bash scripts/dev/ascend_remote_collect.sh
```

## 4) Log layout

Remote:
- `${ASCEND_REMOTE_LOG_ROOT}/<run-id>/build.log`
- `${ASCEND_REMOTE_LOG_ROOT}/<run-id>/test.log`
- `${ASCEND_REMOTE_LOG_ROOT}/<run-id>/status.env`

Local:
- `${ASCEND_LOCAL_LOG_ROOT}/<run-id>/...`
