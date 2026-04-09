import torch
import einops
import flashinfer

"""
KV Cache Layout: 
 - NHD: [num_pages, page_size, num_kv_heads, head_dim]
 - HND: [num_pages, num_kv_heads, page_size, head_dim]
"""


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


seq_len = 1029
head_dim = 128
page_size = 32
num_kv_heads = 8
max_length = 131072
dtype = torch.bfloat16

num_pages = (max_length + page_size - 1) // page_size

paged_k_cache = torch.empty(
    num_pages, page_size, num_kv_heads, head_dim,
    dtype=dtype,
    device="cuda:0"
).contiguous()
paged_v_cache = torch.empty(
    num_pages, page_size, num_kv_heads, head_dim,
    dtype=torch.bfloat16,
    device="cuda:0"
).contiguous()
paged_kv_cache = (paged_k_cache, paged_v_cache)

# same layout as in estimate.py
pooling_cache = torch.empty(
    num_pages, num_kv_heads, head_dim,
    dtype=torch.bfloat16,
    device="cuda:0"
).contiguous()

current_pages = (seq_len + page_size - 1) // page_size
if seq_len % page_size == 0:
    last_page_len = page_size
else:
    last_page_len = seq_len % page_size
kv_page_indptr = torch.cat(
    [torch.zeros(1).int().to(0), torch.tensor([current_pages]).int().to(0)], dim=0
).int()
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")
kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda:0")

k_append = torch.randn(
    seq_len, num_kv_heads, head_dim,
    dtype=torch.bfloat16,
    device="cuda:0"
).contiguous()
v_append = torch.randn(
    seq_len, num_kv_heads, head_dim,
    dtype=torch.bfloat16,
    device="cuda:0"
).contiguous()

flashinfer.append_paged_kv_cache_prefill(
    k_append,
    v_append,
    paged_kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    pooling_cache,
    kv_layout="NHD"
)

paged_k = einops.rearrange(
    paged_kv_cache[0],
    "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
)
paged_v = einops.rearrange(
    paged_kv_cache[1],
    "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
)

assert_close(paged_k[:seq_len, :], k_append)
assert_close(paged_v[:seq_len, :], v_append)
assert paged_v[seq_len:, :].sum() == 0

k_pooled = torch.stack(
    [
        k_append[i * page_size + 4: min((i + 1) * page_size + 4, seq_len), :, :].mean(dim=0)
        for i in range(current_pages - 1)
    ], dim=0
)

assert_close(k_pooled, pooling_cache[:current_pages - 1, :] / page_size)
