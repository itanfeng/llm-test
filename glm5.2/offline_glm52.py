#!/usr/bin/env python3
"""Run GLM-5.2 offline and dump patched SFA Top-K tensors."""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start one offline GLM-5.2 engine, run prompts, and dump Top-K tensors."
    )
    parser.add_argument("--model", default=env("MODEL_PATH", "/data/model/GLM-5.2-w8a8"))
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path(env("PROMPTS_FILE", str(SCRIPT_DIR / "swe_prompts.jsonl"))),
    )
    parser.add_argument(
        "--responses-file",
        "--output-file",
        dest="responses_file",
        type=Path,
        default=Path(
            env("RESP_FILE", str(SCRIPT_DIR / "offline_glm52_responses.jsonl"))
        ),
    )
    parser.add_argument(
        "--topk-dir",
        type=Path,
        default=Path(env("GLM52_TOPK_DIR", str(SCRIPT_DIR / "pt"))),
    )
    parser.add_argument("--start-line", type=int, default=int(env("START_LINE", "1")))
    parser.add_argument("--count", type=int, default=int(env("COUNT", "1")))
    parser.add_argument(
        "--decode-steps", type=int, default=int(env("DECODE_STEPS", "64"))
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ["MAX_TOKENS"]) if "MAX_TOKENS" in os.environ else None,
    )
    parser.add_argument("--temperature", type=float, default=float(env("TEMPERATURE", "0")))
    parser.add_argument(
        "--ignore-eos", type=int, choices=(0, 1), default=int(env("IGNORE_EOS", "1"))
    )
    parser.add_argument(
        "--clear-dumps",
        action="store_true",
        default=env("CLEAR_TOPK_DUMPS", "0") == "1",
    )
    parser.add_argument(
        "--require-dump",
        action="store_true",
        default=env("REQUIRE_TOPK_DUMP", "0") == "1",
    )
    return parser.parse_args()


def configure_environment(topk_dir: Path) -> None:
    defaults = {
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "1",
        "HCCL_BUFFSIZE": "200",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ASCEND_BALANCE_SCHEDULING": "1",
        "VLLM_ASCEND_ENABLE_MLAPO": "1",
        "VLLM_VERSION": "0.21.0",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "GLM52_TOPK_TP_RANK": "0",
        "GLM52_TOPK_PRINT": "0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    os.environ["GLM52_TOPK_DIR"] = str(topk_dir)


def load_records(path: Path, start_line: int, count: int) -> list[dict[str, Any]]:
    if start_line < 1:
        raise ValueError("--start-line must be at least 1")
    if count < 0:
        raise ValueError("--count must be non-negative")
    if not path.is_file():
        raise FileNotFoundError(f"prompts file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number < start_line:
                continue
            if count and len(records) >= count:
                break
            record = json.loads(line)
            if not isinstance(record.get("prompt"), str):
                raise ValueError(f"line {line_number} has no string 'prompt' field")
            record["_line"] = line_number
            records.append(record)

    if not records:
        raise ValueError("no prompts selected")
    return records


def clear_dumps(topk_dir: Path) -> None:
    topk_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.pt", "*.tmp"):
        for path in topk_dir.glob(pattern):
            path.unlink()


def main() -> None:
    args = parse_args()
    args.topk_dir = args.topk_dir.expanduser().resolve()
    args.prompts_file = args.prompts_file.expanduser().resolve()
    args.responses_file = args.responses_file.expanduser().resolve()

    if args.decode_steps < 1:
        raise ValueError("--decode-steps must be positive")
    if args.max_tokens is None:
        # The first generated token comes from prefill.
        args.max_tokens = args.decode_steps + 1
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")

    configure_environment(args.topk_dir)
    records = load_records(args.prompts_file, args.start_line, args.count)
    if args.clear_dumps:
        clear_dumps(args.topk_dir)
    else:
        args.topk_dir.mkdir(parents=True, exist_ok=True)
    before_dumps = set(args.topk_dir.glob("*.pt"))
    args.responses_file.parent.mkdir(parents=True, exist_ok=True)

    # Import only after Ascend/vLLM environment variables have been configured.
    from vllm import LLM, SamplingParams

    dp_size = int(env("OFFLINE_DP_SIZE", "1"))
    tp_size = int(env("TP_SIZE", "16"))
    max_model_len = int(env("MAX_MODEL_LEN", "72000"))
    max_num_seqs = int(env("MAX_NUM_SEQS", "16"))
    max_num_batched_tokens = int(env("MAX_NUM_BATCHED_TOKENS", "10240"))
    if dp_size != 1:
        raise ValueError(
            "offline LLM single-process mode requires OFFLINE_DP_SIZE=1; "
            "use server_glm52.sh for data parallel inference"
        )

    print("=" * 64)
    print("Starting GLM-5.2 offline engine")
    print(f"Model:                {args.model}")
    print(f"Prompts:              {args.prompts_file}")
    print(f"Selected requests:    {len(records)}")
    print(f"Decode steps:         {args.decode_steps}")
    print(f"Max output tokens:    {args.max_tokens}")
    print(f"DP / TP:              {dp_size} / {tp_size}")
    print(f"Expected NPU count:   {dp_size * tp_size}")
    print(f"Max model length:     {max_model_len}")
    print(f"Max sequences:        {max_num_seqs}")
    print(f"Max batched tokens:   {max_num_batched_tokens}")
    print(f"Top-K dump directory: {args.topk_dir}")
    print(f"Responses:            {args.responses_file}")
    print("=" * 64)

    llm = LLM(
        model=args.model,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=True,
        seed=1024,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=0.95,
        quantization="ascend",
        enforce_eager=True,
        trust_remote_code=True,
        distributed_executor_backend="mp",
        additional_config={
            "fuse_muls_add": True,
            "multistream_overlap_shared_expert": True,
        },
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        seed=1024,
        ignore_eos=bool(args.ignore_eos),
    )
    conversations = [[{"role": "user", "content": record["prompt"]}] for record in records]

    started = time.monotonic()
    try:
        outputs = llm.chat(conversations, sampling_params=sampling_params)
        with args.responses_file.open("w", encoding="utf-8") as target:
            for record, output in zip(records, outputs, strict=True):
                result = {
                    "line": record["_line"],
                    "instance_id": record.get("instance_id", ""),
                    "repo": record.get("repo", ""),
                    "response": output.outputs[0].text,
                    "finish_reason": output.outputs[0].finish_reason,
                }
                target.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"line={result['line']} repo={result['repo']} "
                    f"instance_id={result['instance_id']}"
                )
    finally:
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del llm
        gc.collect()

    after_dumps = set(args.topk_dir.glob("*.pt"))
    new_dumps = after_dumps - before_dumps
    print(f"Completed {len(records)} requests in {time.monotonic() - started:.2f}s")
    print(f"New Top-K dumps: {len(new_dumps)}")
    print(f"Total Top-K dumps: {len(after_dumps)}")
    for path in sorted(new_dumps)[:20]:
        print(path)

    if args.require_dump and not new_dumps:
        raise RuntimeError("inference succeeded but no Top-K tensor was dumped")


if __name__ == "__main__":
    main()
