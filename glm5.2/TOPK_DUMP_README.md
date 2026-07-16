# GLM-5.2 SFA Top-K tensor dump

These files add a diagnostic dump to `vllm_ascend/attention/sfa_v1.py` on the
`glm52-hi` branch. After the patch is applied, the hook always saves freshly computed
Top-K indices from ordinary `DecodeOnly` calls. Only TP rank 0 writes by default.

## Use

```bash
./apply_glm52_sfa_topk_dump.sh check
./apply_glm52_sfa_topk_dump.sh apply
./start_glm52.sh
```

In another shell/container, run:

```bash
./send_glm52_test.sh
```

The sender preserves all existing dumps and waits for the non-streaming response.
The HTTP test succeeds both with and without the dump patch. When the patch is
active, the sender also reports how many new dump files were produced. Set
`REQUIRE_TOPK_DUMP=1` when the test must fail unless a new dump is generated.
Tensor files are retained in the relative directory `data` by default. The scripts
change to their own directory first, so this resolves to
`lightning_indexer_certified_cached/data`. Each `.pt` contains `topk_indices`, `layer_name`,
`dump_index`, `pid`, and `tp_rank`. The file index keeps increasing during the
worker lifetime. Neither the service script nor the sender deletes `.pt` files.

Useful overrides include `MODEL_PATH`, `PORT`, `PROMPT`, `MAX_TOKENS`,
`GLM52_TOPK_DIR`, `GLM52_TOPK_HEADS=0` (all heads), `GLM52_TOPK_K=0` (all K), and
`GLM52_TOPK_LAYER=<layer-name-substring>`.

Patch operations are idempotent: applying an already-applied patch or reversing an
already-reversed patch succeeds without changing files. To remove the source change:

```bash
./apply_glm52_sfa_topk_dump.sh reverse
```
## dataset
https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_40K

