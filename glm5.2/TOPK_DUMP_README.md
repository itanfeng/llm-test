# GLM-5.2 SFA Top-K tensor dump

These files add a diagnostic dump to `vllm_ascend/attention/sfa_v1.py` on the
`glm52-hi` branch. After the patch is applied, the hook always saves freshly computed
Top-K indices from ordinary `DecodeOnly` calls. Only TP rank 0 writes by default.

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
records, not the number of decode steps. By default it sends all prompts and requests
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
Tensor files are retained in the relative directory `data` by default. The scripts
change to their own directory first, so this resolves to
`lightning_indexer_certified_cached/data`. Each `.pt` contains `topk_indices`, `layer_name`,
`dump_index`, `pid`, and `tp_rank`. The file index keeps increasing during the
worker lifetime. Neither the service script nor the sender deletes `.pt` files.

Useful overrides include `MODEL_PATH`, `PORT`, `PROMPT`, `MAX_TOKENS`,
`GLM52_TOPK_DIR`, `GLM52_TOPK_HEADS=0` (all heads), and
`GLM52_TOPK_LAYER=<layer-name-substring>`.

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

It uses the same DP/TP, model length, batching, sampling, Ascend environment, and
Top-K defaults as `server_glm52.sh` and `client_glm52.sh`. Use `--start-line` and
`--count` to select records, `--clear-dumps` to remove existing tensors, and
`--require-dump` to fail when the patch does not produce a new tensor.
## dataset
https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_40K
