# GLM-5.2 SFA Top-K tensor dump

These files add a diagnostic dump at the actual lightning indexer calls in
`vllm_ascend/device/device_op.py` on the `glm52-hi` branch. After the patch is
applied, each `.pt` captures one complete `DecodeOnly` operator invocation. Only
TP rank 0 writes by default.

## Use

```bash
./apply_glm52_sfa_topk_dump.sh check
./apply_glm52_sfa_topk_dump.sh apply
./server_glm52.sh
```

In another shell/container, run:

```bash
./client_glm52.sh
```

The client reads the `prompt` field from `swe_prompts.jsonl` and sends requests
sequentially to the running online service. `COUNT` selects the number of prompt
records, not the number of decode steps. By default it sends one prompt and requests
65 output tokens per prompt with EOS ignored, producing 64 calls through the patched
`DecodeOnly` path. Existing dumps are preserved by default; set
`CLEAR_TOPK_DUMPS=1` to remove old `.pt` and `.tmp` files first. Override generation
with `DECODE_STEPS`, `MAX_TOKENS`, or `IGNORE_EOS`.
For example, select the third prompt with `START_LINE=3 COUNT=1 ./client_glm52.sh`.
It waits for every non-streaming response and writes results to
`online_glm52_responses.jsonl`. The online logs are
`online_glm52_server.log` and `online_glm52_client.log`.
The HTTP test succeeds both with and without the dump patch. When the patch is
active, the sender also reports how many new dump files were produced. Set
`REQUIRE_TOPK_DUMP=1` when the test must fail unless a new dump is generated.
Tensor files are retained in the script-relative directory `pt` by default, which
resolves to `/data/tf/llm-test/glm5.2/pt` on the server. Dumps are limited to
0-based layers `0, 1, 2, 6, 10, 14, 30, 34, 38, 66, 70, 74`, matching the actual
online lightning indexer calls observed for this checkpoint. With the default 64
decode steps, this produces `12 * 64 = 768` files. Each `.pt` contains `operator`,
complete `inputs`, complete `outputs`, scalar `attrs`, `layer_name`, `dump_index`,
`pid`, and `tp_rank`, so it can be used for standalone operator replay. The file
index keeps increasing during the worker lifetime. Neither online nor offline runs
delete `.pt` files by default.

The patched worker prints its loaded source path once, reports the first hook call
for each layer/state pair, and prints every successfully saved `.pt` path. These
messages use the `[GLM52_DUMP]` prefix in the server or offline log.

Useful overrides include `MODEL_PATH`, `PORT`, `MAX_TOKENS`, and `GLM52_TOPK_DIR`.

Patch operations are idempotent: applying an already-applied patch or reversing an
already-reversed patch succeeds without changing files. To remove the source change:

```bash
./apply_glm52_sfa_topk_dump.sh reverse
```

## Offline run

The offline runner creates one vLLM engine, reads `swe_prompts.jsonl`, runs the
selected requests, writes `offline_glm52_responses.jsonl`, and then releases the
engine. The wrapper log is `offline_glm52.log`:

```bash
./run_offline_glm52.sh
```

It uses TP=16 and DP=1 because vLLM's single-process offline `LLM` API does not
support DP greater than one and TP=8 cannot fit the model weights on each NPU.
Other model length, batching, sampling, Ascend, and Top-K defaults match the online
scripts. Use `--start-line` and
`--count` to select records, `--clear-dumps` to remove existing tensors, and
`--require-dump` to fail when the patch does not produce a new tensor.
## dataset
https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_40K
