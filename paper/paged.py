import torch
import einops
import flashinfer

seq_len = 8196
head_dim = 128
page_size = 32
num_kv_heads = 8
max_length = 131072


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


num_pages = (max_length + page_size - 1) // page_size
paged_kv_cache = torch.empty(2, num_kv_heads, num_pages, page_size, head_dim).half().to(0).contiguous()
pooling_cache = torch.empty(num_kv_heads, num_pages, head_dim).half().to(0).contiguous()

k_append = torch.randn(seq_len, num_kv_heads, head_dim).half().to(0)
v_append = torch.randn(seq_len, num_kv_heads, head_dim).half().to(0)

current_pages = (seq_len + page_size - 1) // page_size
last_page_len = seq_len % page_size
kv_page_indptr = torch.cat(
    [torch.zeros(1).int().to(0), torch.tensor([current_pages]).int().to(0)], dim=0
).int()
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")
kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda:0")

flashinfer.append_paged_kv_cache_prefill(
    k_append,
    v_append,
    paged_kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    pooling_cache,
    kv_layout="HND"
)

paged_k = einops.rearrange(
    paged_kv_cache[:, 0].transpose(1, 2),
    "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
)

assert_close(paged_k[:seq_len], k_append)

k_pooled = torch.stack([
    k_append[i * page_size + 4: min((i + 1) * page_size + 4, seq_len), :, :].mean(dim=0)
    for i in range(current_pages - 1)
], dim=0)

assert_close(k_pooled, pooling_cache[:current_pages - 1] / page_size)

# decode:
# 1. update kv_last_page_len, kv_page_indices;
# 2. move data
kv_last_page_len = torch.tensor([last_page_len + 1], dtype=torch.int32, device="cuda:0")
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")

q = torch.randn(num_kv_heads, head_dim).half().to(0)
k = torch.randn(num_kv_heads, head_dim).half().to(0)
v = torch.randn(num_kv_heads, head_dim).half().to(0)

flashinfer.append_paged_kv_cache_decode(
    k,
    v,
    paged_kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    pooling_cache,
    kv_layout="HND"
)

assert_close(k, pooling_cache[current_pages - 1])
