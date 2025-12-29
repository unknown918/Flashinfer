"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch

import flashinfer
from benchmarks.bench_trtllm_fmha import page_size
from flashinfer.testing.utils import (
    bench_gpu_time,
    attention_tflops_per_sec_with_actual_seq_lens,
)


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def bench_variable_block_sparse_attention(
        num_qo_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        num_blocks_row,
        num_blocks_col,
        block_density,
):
    if num_qo_heads % num_kv_heads != 0:
        return
    if seq_len // num_blocks_row < 1:
        return
    if seq_len // num_blocks_col < 1:
        return

    block_row_sz = torch.tensor([[1]], dtype=torch.int32, device="cuda").repeat(num_kv_heads, 1)

    block_col_sz = torch.ones(num_blocks_col, dtype=torch.int32) * (
            seq_len // num_blocks_col
    )
    block_col_sz[-1] = seq_len - (seq_len // num_blocks_col) * (num_blocks_col - 1)
    block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    block_mask_map = (
            torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) < block_density
    )

    q = torch.randn(num_qo_heads, 1, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.half, device="cuda")

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )
    sparse_wrapper_fa2 = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="fa2"
    )

    sparse_wrapper_fa2.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=torch.half,
    )

    # Benchmark sparse attention with FA2
    sparse_sm80_ms = np.median(
        bench_gpu_time(
            lambda: sparse_wrapper_fa2.run(q, k, v),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    q = torch.randn(num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    dense_sm80_ms = (
        np.median(
            bench_gpu_time(
                lambda: flashinfer.single_decode_with_kv_cache(
                    q, k, v
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
    )

    def flops(ms):
        return attention_tflops_per_sec_with_actual_seq_lens(
            torch.tensor([seq_len]),
            torch.tensor([seq_len]),
            head_dim,
            head_dim,
            num_qo_heads,
            False,
            ms,
        )

    print(
        f"bench_variable_block_sparse_attention\n"
        f"num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, seq_len={seq_len}\n"
        f"num_blocks_row={num_blocks_row}, num_blocks_col={num_blocks_col}, block_density={block_density}\n"
        # f"sparse fa2-template: {flops(sparse_ms_fa2):.3f} TFLOPs/s\n"
        # f"dense fa2-template: {flops(dense_sm80_ms):.3f} TFLOPs/s\n"
        f"Sparse Attention: {sparse_sm80_ms:.3f} ms\n"
        f"Dense Attention: {dense_sm80_ms:.3f} ms"
    )


if __name__ == "__main__":
    for num_qo_heads in [32]:
        for num_kv_heads in [8]:
            for head_dim in [128]:
                for seq_len in [32768]:
                    for num_blocks_row in [1]:
                        for num_blocks_col in [int(32768 / 32)]:
                            for block_density in [4096 / 32768]:
                                bench_variable_block_sparse_attention(
                                    num_qo_heads,
                                    num_kv_heads,
                                    head_dim,
                                    seq_len,
                                    num_blocks_row,
                                    num_blocks_col,
                                    block_density,
                                )
