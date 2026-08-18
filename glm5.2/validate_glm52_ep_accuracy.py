#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a basic GLM-5.2 accuracy regression with the documented EP path."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

_SERVER_TIMEOUT_SECONDS = 3600
_REQUEST_TIMEOUT_SECONDS = 120
_SHUTDOWN_TIMEOUT_SECONDS = 90
_TEST_CASES = (
    ("What is 2+2? Answer with only the number.", "4"),
    ("法国的首都是哪里？只回答城市名。", "巴黎"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GLM-5.2 accuracy with a single-node TP16 EP deployment."
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GLM52_MODEL", "/data/model/GLM-5.2-w4a8"),
        help="local GLM-5.2 model path",
    )
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--enable-sparse-li-c8",
        action="store_true",
        help="enable the sparse LI C8 path for W4A8C8 weights",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=_SERVER_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--log-file",
        help="server log path; a temporary file is used when omitted",
    )
    return parser.parse_args()


def _request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    ) as response:
        return json.load(response)


def _wait_until_ready(
    process: subprocess.Popen,
    base_url: str,
    timeout: int,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}")
        try:
            _request_json(f"{base_url}/v1/models")
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(2)
    raise TimeoutError(f"vLLM server was not ready within {timeout} seconds")


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _tail(path: Path, line_count: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-line_count:])
    except OSError:
        return "<server log unavailable>"


def main() -> int:
    args = _parse_args()
    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(
            prefix="glm52_ep_accuracy_",
            suffix=".log",
            delete=False,
        ) as log_file:
            log_path = Path(log_file.name)

    env = os.environ.copy()
    env.update(
        {
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_TRANSFER_TIMEOUT": "600",
            "HCCL_EXEC_TIMEOUT": "3600",
            "HCCL_CONNECT_TIMEOUT": "3600",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "1",
            "HCCL_BUFFSIZE": "200",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
            "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
        }
    )

    additional_config = {
        "enable_dsa_cp": True,
        "enable_balance_scheduling": True,
    }
    if args.enable_sparse_li_c8:
        additional_config["enable_sparse_li_c8"] = True

    command = [
        "vllm",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--api-server-count",
        "1",
        "--served-model-name",
        "glm-5",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--enable-auto-tool-choice",
        "--data-parallel-size",
        "1",
        "--tensor-parallel-size",
        "16",
        "--enable-expert-parallel",
        "--seed",
        "1024",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "12",
        "--max-num-batched-tokens",
        "8192",
        "--trust-remote-code",
        "--quantization",
        "ascend",
        "--gpu-memory-utilization",
        "0.92",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--additional-config",
        json.dumps(additional_config, separators=(",", ":")),
        "--speculative-config",
        '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}',
    ]

    print(f"Server log: {log_path}")
    with log_path.open("w") as server_log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    base_url = f"http://127.0.0.1:{args.port}"
    try:
        _wait_until_ready(process, base_url, args.server_timeout)
        for prompt, expected in _TEST_CASES:
            response = _request_json(
                f"{base_url}/v1/chat/completions",
                {
                    "model": "glm-5",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 16,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            actual = response["choices"][0]["message"]["content"].strip()
            status = "PASS" if actual == expected else "FAIL"
            print(
                f"[{status}] prompt={prompt!r} expected={expected!r} "
                f"actual={actual!r}"
            )
            if actual != expected:
                return 1
        print("Basic GLM-5.2 EP accuracy regression passed.")
        return 0
    except Exception:
        print(_tail(log_path))
        raise
    finally:
        _stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
