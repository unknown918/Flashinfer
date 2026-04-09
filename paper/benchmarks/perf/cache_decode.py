import torch
import einops
import flashinfer


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


seq_len = 1028
head_dim = 128
page_size = 32
num_kv_heads = 8
max_length = 131072

num_pages = (max_length + page_size - 1) // page_size

# NHD: [num_pages, page_size, num_kv_heads, head_dim]
# HND: [num_pages, num_kv_heads, page_size, head_dim]
# HNND: [num_kv_heads, num_pages, page_size, head_dim]
paged_k_cache = torch.empty(num_pages, page_size, num_kv_heads, head_dim).half().to(0).contiguous()
paged_v_cache = torch.empty(num_pages, page_size, num_kv_heads, head_dim).half().to(0).contiguous()
paged_kv_cache = (paged_k_cache, paged_v_cache)

# same layout as in estimate.py
pooling_cache = torch.empty(num_pages, num_kv_heads, head_dim).half().to(0).contiguous()

# decode
k = torch.randn(2, num_kv_heads, head_dim).half().to(0)
v = torch.randn(2, num_kv_heads, head_dim).half().to(0)

# append first token
seq_len += 1
current_pages = (seq_len + page_size - 1) // page_size
if seq_len % page_size == 0:
    last_page_len = page_size
else:
    last_page_len = seq_len % page_size

kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda:0")
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")

flashinfer.append_paged_kv_cache_decode(
    k[0],
    v[0],
    paged_kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    pooling_cache,
    kv_layout="HNND"
)

assert_close(k[0], pooling_cache[current_pages - 1])

paged_k = einops.rearrange(
    paged_kv_cache[0],
    "num_kv_heads num_pages page_size head_dim -> num_kv_heads (num_pages page_size) head_dim"
)
paged_v = einops.rearrange(
    paged_kv_cache[1],
    "num_kv_heads num_pages page_size head_dim -> num_kv_heads (num_pages page_size) head_dim"
)
assert_close(paged_k[:, seq_len - 1], k[0])
assert_close(paged_v[:, seq_len - 1], v[0])

# append second token
seq_len += 1
current_pages = (seq_len + page_size - 1) // page_size
if seq_len % page_size == 0:
    last_page_len = page_size
else:
    last_page_len = seq_len % page_size

kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda:0")
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")

flashinfer.append_paged_kv_cache_decode(
    k[1],
    v[1],
    paged_kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    pooling_cache,
    kv_layout="HNND"
)

assert_close(k.sum(dim=0), pooling_cache[current_pages - 1])

paged_k = einops.rearrange(
    paged_kv_cache[0],
    "num_kv_heads num_pages page_size head_dim -> num_kv_heads (num_pages page_size) head_dim"
)
paged_v = einops.rearrange(
    paged_kv_cache[1],
    "num_kv_heads num_pages page_size head_dim -> num_kv_heads (num_pages page_size) head_dim"
)

assert_close(paged_k[:, seq_len - 1], k[1])
assert_close(paged_v[:, seq_len - 1], v[1])