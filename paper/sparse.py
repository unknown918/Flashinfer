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
from flashinfer.testing.utils import (
    bench_gpu_time,
    attention_tflops_per_sec_with_actual_seq_lens,
)

torch.manual_seed(42)

page_size = 32
head_dim = 128
num_kv_heads = 8
num_qo_heads = 32

budget = 4096
seq_len = 1024
max_length = 131072

num_blocks_row = 1
num_blocks_col = int(seq_len / page_size)
block_density = 1


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


block_row_sz = torch.tensor([[1]], dtype=torch.int32, device="cuda").repeat(num_kv_heads, 1)

block_col_sz = torch.ones(num_blocks_col, dtype=torch.int32) * (
        seq_len // num_blocks_col
)
block_col_sz[-1] = seq_len - (seq_len // num_blocks_col) * (num_blocks_col - 1)
block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_heads, 1)

block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col) < block_density
)

num_pages = (max_length + page_size - 1) // page_size

paged_k_cache = torch.empty(num_kv_heads, num_pages, page_size, head_dim).half().to(0).contiguous()
paged_v_cache = torch.empty(num_kv_heads, num_pages, page_size, head_dim).half().to(0).contiguous()
paged_kv_cache = (paged_k_cache, paged_v_cache)

pooling_cache = torch.empty(num_pages, num_kv_heads, head_dim).half().to(0).contiguous()

# same layout as in estimate.py
q = torch.randn(num_qo_heads, head_dim).half().to(0)
k = torch.randn(num_kv_heads, seq_len, head_dim).half().to(0)
v = torch.randn(num_kv_heads, seq_len, head_dim).half().to(0)

current_pages = (seq_len + page_size - 1) // page_size
last_page_len = seq_len % page_size
kv_page_indptr = torch.cat(
    [torch.zeros(1).int().to(0), torch.tensor([current_pages]).int().to(0)], dim=0
).int()
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")
kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda:0")

append_page_prefill_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.append_paged_kv_cache_prefill(
            k,
            v,
            paged_kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            pooling_cache,
            kv_layout="HNND"
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

float_workspace_buffer = torch.empty(
    128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
)

wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
    float_workspace_buffer, backend="fa2"
)

wrapper.plan(
    block_mask_map=block_mask_map,
    block_row_sz=block_row_sz,
    block_col_sz=block_col_sz,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    q_data_type=torch.half,
)

out = wrapper.run(
    q.unsqueeze(1),
    paged_kv_cache[0],
    paged_kv_cache[1]
)

ref = flashinfer.single_decode_with_kv_cache(
    q, k, v, kv_layout="HND"
)

assert_close(ref, out)

print(f"Append KV Cache Overhead: {append_page_prefill_overhead:.3f}")
