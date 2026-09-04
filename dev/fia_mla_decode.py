# -*- coding: utf-8 -*-
import torch
import torch_npu
import numpy as np
import time
import os
import random


def rand_bf16(shape):
    return torch.randn(shape, dtype=torch.bfloat16).npu()


def set_global_seed():
    seed = int(os.getenv("FIA_SEED", "20260301"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(seed)
    print(f"FIA_SEED={seed}")


def test_npu_fused_infer_attention_decode():
    set_global_seed()
    # ===== shapes =====
    B = int(os.getenv("FIA_B", "5"))
    H = int(os.getenv("FIA_H", "16"))
    KVH = int(os.getenv("FIA_KVH", "1"))
    S_q = int(os.getenv("FIA_SQ", "1"))
    D = int(os.getenv("FIA_D", "512"))
    D_rope = int(os.getenv("FIA_D_ROPE", "64"))

    KV_TOTAL_BLOCKS = int(os.getenv("FIA_KV_TOTAL_BLOCKS", "1365"))
    KV_BLOCK = int(os.getenv("FIA_KV_BLOCK", "128"))

    # ===== 构造随机tensor =====
    q_nope = rand_bf16((B, H, S_q, D))
    k_nope = rand_bf16((KV_TOTAL_BLOCKS, KVH, KV_BLOCK, D))

    query_rope = rand_bf16((B, H, S_q, D_rope))
    key_rope = rand_bf16((KV_TOTAL_BLOCKS, KVH, KV_BLOCK, D_rope))

    # ===== 其他参数 =====
    num_heads = H
    num_key_value_heads = KVH
    input_layout = "BNSD_NBSD"

    atten_mask = None
    sparse_mode = 0

    scale = 0.1352337788608801

    antiquant_mode = 0
    antiquant_scale = None

    block_table_len = int(os.getenv("FIA_BLOCK_TABLE_LEN", "243"))
    # block table
    block_table = torch.randint(
        low=0,
        high=KV_TOTAL_BLOCKS,
        size=(B, block_table_len),
        dtype=torch.int32
    ).npu()

    block_size = KV_BLOCK

    actual_seq_lengths = None

    # 可调的 kv seq len：默认使用 block_table_len * block_size 附近的值
    kv_base = block_table_len * block_size
    actual_seq_lengths_kv = [kv_base - i * 3 for i in range(B)]

    # ===== run =====
    run_times_ms = []
    loop_count = int(os.getenv("FIA_LOOP_COUNT", "20"))

    for i in range(loop_count):
        torch_npu.npu.synchronize()
        start_time = time.perf_counter()
        attn_output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
            q_nope,
            k_nope,
            k_nope,
            query_rope=query_rope,
            key_rope=key_rope,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            input_layout=input_layout,
            atten_mask=atten_mask,
            sparse_mode=sparse_mode,
            scale=scale,
            antiquant_mode=antiquant_mode,
            antiquant_scale=antiquant_scale,
            block_table=block_table,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        torch_npu.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        run_times_ms.append(elapsed_ms)

        print("---- run", i)
        print("attn_output:", attn_output.shape, attn_output.dtype)
        print("softmax_lse:", softmax_lse.shape, softmax_lse.dtype)
        print(f"elapsed_ms: {elapsed_ms:.3f}")

    avg_time_ms = sum(run_times_ms) / len(run_times_ms)
    if len(run_times_ms) > 1:
        steady_avg_ms = sum(run_times_ms[1:]) / (len(run_times_ms) - 1)
        print(f"average_elapsed_ms_steady({loop_count - 1} runs, drop first): {steady_avg_ms:.3f}")
    print(f"average_elapsed_ms_all({loop_count} runs): {avg_time_ms:.3f}")


if __name__ == "__main__":
    test_npu_fused_infer_attention_decode()
