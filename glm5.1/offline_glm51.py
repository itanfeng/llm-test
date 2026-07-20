#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V3.2 / GLM-5.1 HiSparse FULL_DECODE_ONLY graph offline validation.

Supports both LongBench prompts and synthetic prompts generated with a fixed
input length (``--input-len``). The shared helpers are kept compatible with
``profiling.py``. Keep profiling-specific msprof logic out of this file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["USE_MULTI_GROUPS_KV_CACHE"] = "0"
os.environ.setdefault("MASTER_PORT", str(29500 + random.randint(0, 2000)))
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"

_DEFAULT_JSONL = "/data/datasets/longbench/data/multifieldqa_zh.jsonl"
_DEFAULT_MODEL = "/data/model/GLM-5.1-W8A8/"
_BLOCK_SIZE = 128
_USER_BODY = (
    "你是一个只基于给定文本回答问题的助手。\n"
    "要求：直接写出对问题的回答内容；不要写答题要求、不要评价对错、不要复述问题。\n"
    "如果文本中找不到答案，只输出：文本中未找到明确答案。\n\n"
    "【文本】\n{text}\n\n"
    "【问题】\n{question}\n\n"
    "回答："
)
_PROMPT_PADDING_CANDIDATES = ("\n.\n", "\nA\n", "\n0\n", "。", ".", " a", "\n", " ")


# Sparse-attention models that can run with HiSparse. The runtime does not care
# about the exact architecture, but the validation/prompt helpers need to know
# whether we are driving a DeepSeek-V3.2-style model or a GLM-5.1-style model.
_DEEPSEEK_V32_MODEL_TYPES = ("deepseek_v32",)
_GLM_DSA_MODEL_TYPES = ("glm_moe_dsa",)


@dataclass(frozen=True)
class PromptCase:
    prompt: Any
    sample_id: int
    sample: dict[str, Any]
    question: str
    gold_answers: str
    prompt_tokens: int
    aligned_tokens: int
    source: str


@dataclass(frozen=True)
class PromptBatch:
    cases: list[PromptCase]

    @property
    def prompts(self) -> list[Any]:
        return [case.prompt for case in self.cases]

    @property
    def prompt_tokens(self) -> list[int]:
        return [case.prompt_tokens for case in self.cases]

    @property
    def samples(self) -> list[dict[str, Any]]:
        return [case.sample for case in self.cases]

    @property
    def sample_ids(self) -> list[int]:
        return [case.sample_id for case in self.cases]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_longbench_samples(
    jsonl_path: str,
    *,
    num_samples: int,
    sample_ids: list[int] | None,
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if sample_ids is not None:
        return [rows[index] for index in sample_ids]
    return rows[:num_samples]


def _detect_model_family(model_path: str) -> str | None:
    """Read model_type from config.json if available."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = str(raw.get("model_type", "")).lower()
        if model_type in _DEEPSEEK_V32_MODEL_TYPES:
            return "deepseek_v32"
        if model_type in _GLM_DSA_MODEL_TYPES:
            return "glm_moe_dsa"
    except Exception:
        pass
    return None


def _tokenizer_mode_for_model(model_path: str) -> str:
    """Return the vLLM tokenizer_mode for the model family.

    DeepSeek-V3.2 needs the custom ``deepseek_v32`` tokenizer. GLM-5.1 uses the
    standard Hugging Face tokenizer path.
    """
    family = _detect_model_family(model_path)
    if family == "deepseek_v32":
        return "deepseek_v32"
    return os.environ.get("HISPARSE_TOKENIZER_MODE", "auto")


def _chat_prompt(tokenizer: Any, user_text: str, model_path: str) -> str:
    template = getattr(tokenizer, "apply_chat_template", None)
    if template is None:
        raise RuntimeError("tokenizer.apply_chat_template required")
    family = _detect_model_family(model_path)
    if family == "deepseek_v32":
        prompt = template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            thinking=False,
        )
    else:
        prompt = template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("apply_chat_template returned empty prompt")
    return prompt


def _prompt_token_ids(tokenizer: Any, prompt: Any) -> list[int]:
    if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
        return list(prompt["prompt_token_ids"])
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def _clone_prompt(prompt: Any) -> Any:
    if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
        cloned = dict(prompt)
        cloned["prompt_token_ids"] = list(prompt["prompt_token_ids"])
        return cloned
    return prompt


def _sample_ids(args: argparse.Namespace) -> list[int]:
    if args.sample_ids:
        ids = _parse_int_list(args.sample_ids)
        if not args.same_prompt_batch and len(ids) != args.num_samples:
            raise ValueError("--sample-ids length must equal --num-samples")
        return ids[:1] if args.same_prompt_batch else ids

    sample_count = 1 if args.same_prompt_batch else args.num_samples
    return list(range(args.sample_id, args.sample_id + sample_count))


def _question(sample: dict[str, Any]) -> str:
    return str(sample.get("input", "")).strip()


def _context(sample: dict[str, Any], args: argparse.Namespace) -> str:
    context = str(sample.get("context", "")).strip()
    max_context_chars = int(getattr(args, "max_context_chars", 0) or 0)
    return context[:max_context_chars] if max_context_chars > 0 else context


def _prompt_from_context(
    tokenizer: Any, context: str, question: str, model_path: str
) -> str:
    return _chat_prompt(
        tokenizer,
        _USER_BODY.format(text=context, question=question),
        model_path,
    )


def _padding_token_ids(tokenizer: Any, token_count: int) -> list[int]:
    if token_count <= 0:
        return []
    for pad in _PROMPT_PADDING_CANDIDATES:
        unit_ids = _prompt_token_ids(tokenizer, pad)
        if unit_ids:
            repeat = (token_count + len(unit_ids) - 1) // len(unit_ids)
            return (unit_ids * repeat)[:token_count]
    raise ValueError("failed to build prompt padding token ids")


def _exact_prompt_token_ids(
    tokenizer: Any,
    *,
    context: str,
    question: str,
    target_tokens: int,
    sample_id: int,
    model_path: str,
) -> tuple[Any, str]:
    sentinel = "<HISPARSE_CONTEXT_SENTINEL>"
    prompt = _prompt_from_context(tokenizer, sentinel, question, model_path)
    if prompt.count(sentinel) != 1:
        raise ValueError("failed to locate context sentinel in chat prompt")

    prefix, suffix = prompt.split(sentinel, 1)
    prefix_ids = _prompt_token_ids(tokenizer, prefix)
    suffix_ids = _prompt_token_ids(tokenizer, suffix)
    context_budget = target_tokens - len(prefix_ids) - len(suffix_ids)
    if context_budget <= 0:
        raise ValueError(
            f"sample {sample_id} cannot fit target prompt length {target_tokens}; "
            f"chat/question wrapper uses {len(prefix_ids) + len(suffix_ids)} tokens"
        )

    context_ids = _prompt_token_ids(tokenizer, context)
    selected_context_ids = context_ids[:context_budget]
    prompt_token_ids = (
        prefix_ids
        + selected_context_ids
        + _padding_token_ids(tokenizer, context_budget - len(selected_context_ids))
        + suffix_ids
    )
    if len(prompt_token_ids) != target_tokens:
        raise AssertionError(
            f"failed to build exact {target_tokens}-token prompt for sample "
            f"{sample_id}; got {len(prompt_token_ids)}"
        )
    return {"prompt_token_ids": prompt_token_ids}, tokenizer.decode(selected_context_ids)


def _build_prompt_case(
    tokenizer: Any,
    args: argparse.Namespace,
    sample: dict[str, Any],
    sample_id: int,
    *,
    copy_index: int | None = None,
) -> PromptCase:
    question = _question(sample)
    context = _context(sample, args)
    if not question or not context:
        raise ValueError(f"LongBench sample {sample_id} has empty question/context")

    target_tokens = int(getattr(args, "prompt_token_target", 0) or 0)
    if target_tokens > 0:
        prompt, context = _exact_prompt_token_ids(
            tokenizer,
            context=context,
            question=question,
            target_tokens=target_tokens,
            sample_id=sample_id,
            model_path=args.model,
        )
        prompt_tokens = target_tokens
    else:
        prompt = _prompt_from_context(tokenizer, context, question, args.model)
        prompt_tokens = len(_prompt_token_ids(tokenizer, prompt))

    source = (
        f"longbench:{args.jsonl}#sample={sample_id} "
        f"id={sample.get('_id')} dataset={sample.get('dataset')} "
        f"context_chars={len(context)}/{len(str(sample.get('context', '')).strip())}"
    )
    if target_tokens > 0:
        source = f"{source} prompt_token_target={target_tokens}"
    if copy_index is not None:
        source = f"{source} same_prompt_copy_index={copy_index}"

    return PromptCase(
        prompt=prompt,
        sample_id=sample_id,
        sample=sample,
        question=question,
        gold_answers=repr(sample.get("answers")),
        prompt_tokens=prompt_tokens,
        aligned_tokens=(prompt_tokens // _BLOCK_SIZE) * _BLOCK_SIZE,
        source=source,
    )


def _synthetic_token_ids(tokenizer: Any, input_len: int) -> list[int]:
    """Build a deterministic sequence of real token ids of length ``input_len``."""
    base_text = (
        "Hello world, this is a synthetic prompt used for throughput testing. "
    )
    base_ids = _prompt_token_ids(tokenizer, base_text)
    if not base_ids:
        print('[ERROR] Base id is NULL!!! Cannot create same req!======================')
        vocab_size = max(1, int(getattr(tokenizer, "vocab_size", 100000)))
        return [random.randint(0, vocab_size - 1) for _ in range(input_len)]
    repeats = (input_len + len(base_ids) - 1) // len(base_ids)
    return (base_ids * repeats)[:input_len]


def _build_synthetic_prompt_batch(
    tokenizer: Any, args: argparse.Namespace
) -> PromptBatch:
    """Generate ``num_samples`` prompts with exactly ``input_len`` tokens each."""
    input_len = int(args.input_len)
    if input_len <= 0:
        raise ValueError("--input-len must be a positive integer")
    token_ids = _synthetic_token_ids(tokenizer, input_len)
    cases: list[PromptCase] = []
    for sample_id in range(args.num_samples):
        source = f"synthetic:sample={sample_id} input_len={input_len}"
        cases.append(
            PromptCase(
                prompt={"prompt_token_ids": list(token_ids)},
                sample_id=sample_id,
                sample={"_id": source, "dataset": "synthetic"},
                question="",
                gold_answers="",
                prompt_tokens=input_len,
                aligned_tokens=(input_len // _BLOCK_SIZE) * _BLOCK_SIZE,
                source=source,
            )
        )
    return PromptBatch(cases=cases)


def _build_prompt_batch(tokenizer: Any, args: argparse.Namespace) -> PromptBatch:
    if int(getattr(args, "input_len", 0) or 0) > 0:
        return _build_synthetic_prompt_batch(tokenizer, args)

    ids = _sample_ids(args)
    samples = _load_longbench_samples(
        args.jsonl,
        num_samples=len(ids),
        sample_ids=ids,
    )
    if args.same_prompt_batch:
        base = _build_prompt_case(tokenizer, args, samples[0], ids[0], copy_index=0)
        cases = [
            PromptCase(
                prompt=_clone_prompt(base.prompt),
                sample_id=base.sample_id,
                sample=base.sample,
                question=base.question,
                gold_answers=base.gold_answers,
                prompt_tokens=base.prompt_tokens,
                aligned_tokens=base.aligned_tokens,
                source=base.source.replace(
                    "same_prompt_copy_index=0",
                    f"same_prompt_copy_index={copy_index}",
                ),
            )
            for copy_index in range(args.num_samples)
        ]
    else:
        cases = [
            _build_prompt_case(tokenizer, args, sample, sample_id)
            for sample, sample_id in zip(samples, ids)
        ]

    if len(cases) != args.num_samples:
        raise AssertionError(
            f"resolved prompt count {len(cases)} does not match "
            f"--num-samples={args.num_samples}"
        )
    _validate_prompt_lengths(cases, args)
    return PromptBatch(cases=cases)


def _build_prompts(
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[Any], list[int], list[dict[str, Any]], list[int]]:
    batch = _build_prompt_batch(tokenizer, args)
    return batch.prompts, batch.prompt_tokens, batch.samples, batch.sample_ids


def _decode_token_count(args: argparse.Namespace) -> int:
    return int(getattr(args, "max_tokens", getattr(args, "decode_tokens", 0)))


def _auto_host_cache_tokens(
    prompt_tokens: list[int],
    max_tokens: int,
    *,
    same_prompt_batch: bool,
) -> int:
    token_lengths = [max(prompt_tokens)] if same_prompt_batch else prompt_tokens
    return sum(
        ((tokens + max(1, max_tokens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
        * _BLOCK_SIZE
        for tokens in token_lengths
    )


def _host_cache_tokens(args: argparse.Namespace, prompt_tokens: list[int]) -> int:
    if args.hisparse_full_backend != "host_memory":
        return 0
    if args.host_cache_tokens > 0:
        return int(args.host_cache_tokens)
    return _auto_host_cache_tokens(
        prompt_tokens,
        _decode_token_count(args),
        same_prompt_batch=args.same_prompt_batch,
    )


def _hisparse_config(args: argparse.Namespace, host_cache_tokens: int) -> str:
    env_config = os.environ.get("HISPARSE_CONFIG")
    if env_config:
        return env_config

    config: dict[str, Any] = {"full_backend": args.hisparse_full_backend}
    if args.hisparse_full_backend == "host_memory" and host_cache_tokens > 0:
        config["host_cache_tokens"] = host_cache_tokens
    return json.dumps(config, separators=(",", ":"))


def _llm_kwargs(
    args: argparse.Namespace,
    *,
    host_cache_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer_mode": _tokenizer_mode_for_model(args.model),
        "trust_remote_code": True,
        "tensor_parallel_size": args.tp,
        "enable_expert_parallel": args.tp > 1,
        "quantization": os.environ.get("HISPARSE_SMOKE_QUANTIZATION", "ascend"),
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": True,
        "block_size": _BLOCK_SIZE,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "async_scheduling": False,
        "additional_config": {
            "recompute_scheduler_enable": False,
            "multistream_overlap_shared_expert": False,
            "enable_hisparse": True,
            "hisparse_config": _hisparse_config(args, host_cache_tokens),
            "enable_prefetch_with_hidden_states": args.enable_prefetch_with_hidden_states,
            "use_lightning_indexer_hi_cached": args.use_lightning_indexer_hi_cached,
        },
        "compilation_config": {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
        },
    }
    if args.max_num_batched_tokens > 0:
        kwargs["max_num_batched_tokens"] = int(args.max_num_batched_tokens)
    if args.max_layers is not None:
        kwargs["hf_overrides"] = {"num_hidden_layers": int(args.max_layers)}
    return kwargs


def _sampling_params(
    max_tokens: int,
    count: int | None = None,
    *,
    temperature: float = 0.1,
    top_p: float = 1.0,
    seed_base: int | None = None,
) -> Any:
    from vllm import SamplingParams

    def _make(index: int) -> Any:
        kwargs: dict[str, Any] = {}
        if seed_base is not None:
            kwargs["seed"] = int(seed_base) + index
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            ignore_eos=True,
            **kwargs,
        )

    return [_make(index) for index in range(count)] if count is not None else _make(0)


def _greedy_sampling_params(max_tokens: int, count: int | None = None) -> Any:
    return _sampling_params(max_tokens, count)


def _validate_prompt_lengths(cases: list[PromptCase], args: argparse.Namespace) -> None:
    too_long = [
        (index, case.prompt_tokens)
        for index, case in enumerate(cases)
        if case.prompt_tokens + _decode_token_count(args) > args.max_model_len
    ]
    if too_long:
        raise ValueError(
            "prompt + decode exceeds max_model_len: "
            + ", ".join(f"request={idx} tokens={tokens}" for idx, tokens in too_long)
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.tp < 1:
        raise ValueError("--tp must be >= 1")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.max_num_seqs < args.num_samples:
        raise ValueError("--max-num-seqs must be >= --num-samples")
    if _decode_token_count(args) < 1:
        raise ValueError("--max-tokens/--decode-tokens must be >= 1")
    if args.max_num_batched_tokens < 0:
        raise ValueError("--max-num-batched-tokens must be >= 0")
    if args.cudagraph_capture_sizes != [args.max_num_seqs]:
        raise ValueError("--cudagraph-capture-sizes must equal --max-num-seqs")
    if args.host_cache_tokens < 0:
        raise ValueError("--host-cache-tokens must be >= 0")
    if args.max_layers is not None and args.max_layers <= 0:
        raise ValueError("--max-layers must be > 0")
    input_len = int(getattr(args, "input_len", 0) or 0)
    prompt_target = int(getattr(args, "prompt_token_target", 0) or 0)
    if input_len > 0 and prompt_target > 0:
        raise ValueError("--input-len and --prompt-token-target are mutually exclusive")
    if input_len < 0:
        raise ValueError("--input-len must be >= 0")
    if input_len == 0 and prompt_target < 0:
        raise ValueError("--prompt-token-target must be 0 or a block-size multiple")
    if input_len == 0 and prompt_target % _BLOCK_SIZE != 0:
        raise ValueError("--prompt-token-target must be 0 or a block-size multiple")


def _add_common_args(parser: argparse.ArgumentParser, *, profiling: bool = False) -> None:
    parser.add_argument("--jsonl", default=os.environ.get("PROF_LONGBENCH_JSONL", _DEFAULT_JSONL))
    parser.add_argument(
        "--model",
        default=os.environ.get("HISPARSE_SMOKE_MODEL", _DEFAULT_MODEL),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for runtime-generated files; supplied by profiling.sh",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=int(os.environ.get("HISPARSE_SMOKE_TP", "16")),
    )
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--sample-ids", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--same-prompt-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="copy one LongBench sample into all request slots",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=int(os.environ.get("PROF_MAX_CONTEXT_CHARS", "0")),
        help="0 keeps full context",
    )
    parser.add_argument(
        "--prompt-token-target",
        type=int,
        default=0,
        help="0 uses natural prompt length; otherwise must be block-size aligned",
    )
    parser.add_argument(
        "--max-tokens" if not profiling else "--decode-tokens",
        dest="max_tokens" if not profiling else "decode_tokens",
        type=int,
        default=64 if not profiling else int(os.environ.get("PROF_DECODE_TOKENS", "128")),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("PROF_MAX_NUM_BATCHED_TOKENS", "0")),
        help="0 keeps the vLLM default",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        type=_parse_int_list,
        default=None,
        help="defaults to --max-num-seqs",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("PROF_GPU_MEMORY_UTILIZATION", "0.85")),
    )
    parser.add_argument(
        "--host-cache-tokens",
        type=int,
        default=0,
        help="0 auto-sizes host full-KV capacity from selected prompts",
    )
    parser.add_argument(
        "--hisparse-full-backend",
        choices=("host_memory", "device"),
        default="host_memory",
    )
    parser.add_argument(
        "--enable-prefetch-with-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable prefetch with hidden states for HiSparse",
    )
    parser.add_argument(
        "--use-lightning-indexer-hi-cached",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LightningIndexerHiCached for HiSparse",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=0,
        help="generate synthetic prompts with exactly this many input tokens; "
        "0 loads prompts from --jsonl (LongBench)",
    )


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_num_seqs is None:
        args.max_num_seqs = args.num_samples
    if args.cudagraph_capture_sizes is None:
        args.cudagraph_capture_sizes = [args.max_num_seqs]
    return args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HiSparse FULL_DECODE_ONLY graph decode on LongBench prompts."
    )
    _add_common_args(parser)
    return _normalize_args(parser.parse_args())


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    from vllm import LLM
    from vllm.tokenizers.registry import get_tokenizer

    tokenizer = get_tokenizer(
        args.model,
        tokenizer_mode=_tokenizer_mode_for_model(args.model),
        trust_remote_code=True,
    )
    batch = _build_prompt_batch(tokenizer, args)
    host_cache_tokens = _host_cache_tokens(args, batch.prompt_tokens)

    print(
        "=== HiSparse graph offline "
        f"tp={args.tp} num_samples={args.num_samples} "
        f"max_num_seqs={args.max_num_seqs} same_prompt_batch={args.same_prompt_batch} "
        f"capture_sizes={args.cudagraph_capture_sizes} "
        f"prefix_cache=True host_cache_tokens={host_cache_tokens} ==="
    )
    for idx, case in enumerate(batch.cases):
        print(
            f"[{idx}] sample_id={case.sample_id} prompt_tokens={case.prompt_tokens} "
            f"gold={case.gold_answers}"
        )

    llm = LLM(**_llm_kwargs(args, host_cache_tokens=host_cache_tokens))
    outputs = llm.generate(
        batch.prompts,
        sampling_params=_greedy_sampling_params(_decode_token_count(args)),
    )
    for idx, output in enumerate(outputs):
        text = (output.outputs[0].text if output.outputs else "").strip()
        cached = getattr(output, "num_cached_tokens", None)
        print(f"[{idx}] cached={cached}/{batch.prompt_tokens[idx]} answer={text!r}")

    print("=== PASSED: HiSparse graph batch generation completed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
