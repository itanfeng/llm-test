#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark GLM-5.2 Full-HBM and Host-Memory DSA decode paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MASTER_PORT", str(29500 + random.randint(0, 2000)))
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

_BLOCK_SIZE = 128
_DEFAULT_DATASET = "/data/datasets/longbench/data/longbench_narrativeqa_64k.jsonl"
_DEFAULT_MAX_MODEL_LEN = 64 * 1024
_DEFAULT_MODEL = "/data/model/GLM-5.2-w4a8"
_DSA_SPARSE_THRESHOLD = 8 * 1024 + 2 * 1024 + _BLOCK_SIZE
_PROMPT_TEMPLATE = (
    "Answer the question using only the supplied text. Be concise.\n\n"
    "Text:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible GLM-5.2 DSA decode benchmark and write "
            "machine-readable results."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("full_hbm", "host_memory"),
        default="host_memory",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("GLM52_MODEL", _DEFAULT_MODEL)),
    )
    parser.add_argument(
        "--dataset",
        "--jsonl",
        dest="dataset",
        type=Path,
        default=Path(os.environ.get("GLM52_DATASET", _DEFAULT_DATASET)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for result JSON and runtime-generated files",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="result JSON path; defaults to <output-dir>/result.json",
    )
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--sample-index", "--sample-id", dest="sample_index", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        "--num-samples",
        dest="batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="defaults to --batch-size and must match it for this benchmark",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=_DEFAULT_MAX_MODEL_LEN,
    )
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lookup-backend",
        choices=("asu_hbm_index_lookup", "asu_hbm_index_lookup_opt"),
        default="asu_hbm_index_lookup_opt",
    )
    parser.add_argument(
        "--segmented-sfa",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-prefetch-with-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-steps", type=int, default=0)
    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.model = args.model.expanduser().resolve()
    args.dataset = args.dataset.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.result is None:
        args.result = args.output_dir / "result.json"
    else:
        args.result = args.result.expanduser().resolve()
    if args.reference_result is not None:
        args.reference_result = args.reference_result.expanduser().resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.expanduser().resolve()
    if args.max_num_seqs is None:
        args.max_num_seqs = args.batch_size
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise ValueError(f"Model directory does not exist: {args.model}")
    if not args.dataset.is_file():
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    if args.sample_index < 0:
        raise ValueError("--sample-index must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_num_seqs != args.batch_size:
        raise ValueError("--max-num-seqs must equal --batch-size")
    if args.warmup_steps < 0 or args.timed_steps <= 0:
        raise ValueError("Warmup must be non-negative and timed steps positive")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.tp <= 0:
        raise ValueError("--tp must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.max_num_batched_tokens <= 0:
        raise ValueError("--max-num-batched-tokens must be positive")
    if args.profile_steps < 0:
        raise ValueError("--profile-steps must be non-negative")
    if bool(args.profile_dir) != (args.profile_steps > 0):
        raise ValueError(
            "--profile-dir and a positive --profile-steps must be used together"
        )
    if args.mode == "full_hbm" and args.batch_size != 1:
        raise ValueError(
            "The Full-HBM baseline is intentionally limited to BS=1; "
            "larger batches exceed the verified HBM budget."
        )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _read_sample(path: Path, sample_index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as dataset_file:
        for index, line in enumerate(dataset_file):
            if index == sample_index:
                sample = json.loads(line)
                if isinstance(sample, dict):
                    return sample
                break
    raise ValueError(f"Dataset has no object sample at index {sample_index}")


def _prepare_prompt(
    model: Path,
    dataset: Path,
    sample_index: int,
) -> tuple[dict[str, Any], list[int]]:
    from vllm.tokenizers.registry import get_tokenizer

    sample = _read_sample(dataset, sample_index)
    context = str(sample.get("context", "")).strip()
    question = str(sample.get("input", "")).strip()
    if not context or not question:
        raise ValueError(
            f"Dataset sample {sample_index} must contain non-empty context and input"
        )

    tokenizer = get_tokenizer(str(model), trust_remote_code=True)
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return sample, list(
        tokenizer.encode(
            chat_prompt,
            add_special_tokens=False,
        )
    )


def _generated_tokens(output: Any) -> list[int]:
    return [int(token_id) for token_id in output.outputs[0].token_ids]


def _generated_text(output: Any) -> str:
    return str(output.outputs[0].text)


def _make_additional_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "enable_dsa_cp": False,
        "enable_balance_scheduling": True,
        "enable_flashcomm1": True,
        # The single-node reference configuration keeps fused MC2 disabled.
        "enable_fused_mc2": False,
        "fuse_muls_add": False,
        "recompute_scheduler_enable": False,
        "multistream_overlap_shared_expert": True,
    }
    if args.mode == "host_memory":
        config["dsa_sparse_config"] = {
            "io_backend": "host_memory",
            "lookup_backend": args.lookup_backend,
            "segmented_sfa": args.segmented_sfa,
            "enable_prefetch_with_hidden_states": args.enable_prefetch_with_hidden_states,
        }
    return config


def _verify_reference(
    result: dict[str, Any],
    reference_path: Path | None,
) -> bool | None:
    if reference_path is None:
        result["reference_match"] = None
        return None
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if result["batch_size"] != reference.get("batch_size"):
        raise ValueError("Reference result must use the same batch size")
    matches = (
        result["prompt_tokens"] == reference.get("prompt_tokens")
        and result["answer_token_ids"] == reference.get("answer_token_ids")
    )
    result["reference_match"] = matches
    return matches


def main() -> int:
    args = _normalize_args(_parse_args())
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    sample, prompt_token_ids = _prepare_prompt(
        args.model,
        args.dataset,
        args.sample_index,
    )
    decode_steps = args.warmup_steps + args.timed_steps
    profile_stop_step = args.warmup_steps + args.profile_steps
    if args.profile_steps > 0 and profile_stop_step > decode_steps:
        print(
            f"BENCH_NOTE profile_steps={args.profile_steps} exceeds "
            f"timed_steps={args.timed_steps}; profiling will stop at "
            f"decode step {decode_steps}.",
            flush=True,
        )
        profile_stop_step = decode_steps
    if len(prompt_token_ids) <= _DSA_SPARSE_THRESHOLD:
        raise ValueError(
            f"Prompt has {len(prompt_token_ids)} tokens; sparse decode requires "
            f"more than {_DSA_SPARSE_THRESHOLD}."
        )
    if len(prompt_token_ids) + decode_steps > args.max_model_len:
        raise ValueError(
            "Prompt and generated tokens exceed --max-model-len: "
            f"{len(prompt_token_ids)} + {decode_steps} > {args.max_model_len}."
        )

    blocks_per_request = math.ceil(args.max_model_len / _BLOCK_SIZE)
    num_gpu_blocks_override = blocks_per_request * args.batch_size
    compilation_config: dict[str, Any] = {
        "cudagraph_mode": "NONE" if args.enforce_eager else "FULL_DECODE_ONLY",
    }
    if not args.enforce_eager:
        compilation_config["cudagraph_capture_sizes"] = [16]
    input_summary = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "sample_index": args.sample_index,
        "prompt_tokens": len(prompt_token_ids),
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "question": sample.get("input"),
        "reference_answers": sample.get("answers"),
        "enforce_eager": args.enforce_eager,
        "lookup_backend": args.lookup_backend,
        "segmented_sfa": args.segmented_sfa,
        "prefetch_mode": (
            "prefetch" if args.enable_prefetch_with_hidden_states else "offload"
        ),
        "profile_steps": args.profile_steps,
    }
    print(
        "BENCH_INPUT " + json.dumps(input_summary, ensure_ascii=False),
        flush=True,
    )

    profiler_config = None
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir),
            "torch_profiler_with_stack": False,
        }

    model_started = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        trust_remote_code=True,
        seed=1024,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        data_parallel_size=1,
        enable_expert_parallel=True,
        block_size=_BLOCK_SIZE,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True,
        quantization="ascend",
        num_gpu_blocks_override=num_gpu_blocks_override,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        compilation_config=compilation_config,
        additional_config=_make_additional_config(args),
        async_scheduling=False,
        profiler_config=profiler_config,
    )
    model_ready_seconds = time.perf_counter() - model_started
    print(f"BENCH_READY seconds={model_ready_seconds:.3f}", flush=True)

    sampling_params = SamplingParams(
        max_tokens=decode_steps,
        temperature=0.0,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    prompt = {"prompt_token_ids": prompt_token_ids}
    for request_index in range(args.batch_size):
        llm.llm_engine.add_request(
            str(request_index),
            prompt,
            sampling_params,
        )

    latest_outputs: dict[str, Any] = {}
    timed_step_ms: list[float] = []
    previous_decode_step = 0
    engine_steps = 0
    request_started = time.perf_counter()
    first_token_seconds = 0.0
    profile_started = False
    profile_finished = False
    while llm.llm_engine.has_unfinished_requests():
        if (
            not profile_started
            and args.profile_steps > 0
            and previous_decode_step == args.warmup_steps
        ):
            llm.start_profile("dsa_sparse_decode")
            profile_started = True
        step_started = time.perf_counter()
        step_outputs = llm.llm_engine.step()
        step_ms = (time.perf_counter() - step_started) * 1000
        engine_steps += 1
        for output in step_outputs:
            latest_outputs[str(output.request_id)] = output
        if len(latest_outputs) != args.batch_size:
            continue

        decode_step = min(
            len(_generated_tokens(output)) for output in latest_outputs.values()
        )
        if decode_step <= previous_decode_step:
            continue
        if decode_step != previous_decode_step + 1:
            raise RuntimeError(
                "Decode step advanced unexpectedly: "
                f"{previous_decode_step} -> {decode_step}"
            )
        previous_decode_step = decode_step
        if decode_step == 1:
            first_token_seconds = time.perf_counter() - request_started
            print(
                f"BENCH_DECODE_START seconds={first_token_seconds:.3f}",
                flush=True,
            )
        if args.warmup_steps < decode_step <= decode_steps:
            timed_step_ms.append(step_ms)
        if profile_started and decode_step == profile_stop_step:
            llm.stop_profile()
            profile_started = False
            profile_finished = True

    if len(timed_step_ms) != args.timed_steps:
        raise RuntimeError(
            f"Expected {args.timed_steps} timed steps, got {len(timed_step_ms)}"
        )
    if profile_finished:
        from torch_npu.profiler.profiler import analyse as npu_profiler_analyse

        analyse_started = time.perf_counter()
        npu_profiler_analyse(str(args.profile_dir))
        print(
            "BENCH_PROFILE_ANALYSE "
            f"seconds={time.perf_counter() - analyse_started:.3f}",
            flush=True,
        )

    outputs = [
        latest_outputs[str(request_index)]
        for request_index in range(args.batch_size)
    ]
    timed_seconds = sum(timed_step_ms) / 1000
    result = {
        **input_summary,
        "model": str(args.model),
        "max_model_len": args.max_model_len,
        "block_size": _BLOCK_SIZE,
        "num_gpu_blocks_override": num_gpu_blocks_override,
        "graph_mode": compilation_config["cudagraph_mode"],
        "model_ready_seconds": model_ready_seconds,
        "first_token_seconds": first_token_seconds,
        "request_seconds": time.perf_counter() - request_started,
        "engine_steps": engine_steps,
        "decode_throughput_tokens_per_second": (
            args.batch_size * args.timed_steps / timed_seconds
        ),
        "decode_step_mean_ms": sum(timed_step_ms) / len(timed_step_ms),
        "decode_step_p50_ms": _percentile(timed_step_ms, 0.50),
        "decode_step_p95_ms": _percentile(timed_step_ms, 0.95),
        "decode_step_p99_ms": _percentile(timed_step_ms, 0.99),
        "decode_step_times_ms": timed_step_ms,
        "answer_token_ids": [_generated_tokens(output) for output in outputs],
        "answers": [_generated_text(output) for output in outputs],
    }
    reference_match = _verify_reference(result, args.reference_result)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "BENCH_RESULT " + json.dumps(result, ensure_ascii=False),
        flush=True,
    )
    if reference_match is False:
        raise RuntimeError("Generated token IDs do not match the reference result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
