import torch
import einops
import flashinfer

"""
KV Cache Layout: 
 - NHD: [num_pages, page_size, num_kv_heads, head_dim]
 - HND: [num_pages, num_kv_heads, page_size, head_dim]
"""
torch.manual_seed(42)


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


sink = 4
seq_len = 1976
head_dim = 128
page_size = 32
num_kv_heads = 8
max_length = 32768
device = torch.device("cuda:0")
dtype = torch.bfloat16

num_pages = (max_length + page_size - 1) // page_size

paged_k_cache = torch.empty(
    num_pages, page_size, num_kv_heads, head_dim,
    dtype=dtype,
    device=device
).contiguous()

paged_v_cache = torch.empty(
    num_pages, page_size, num_kv_heads, head_dim,
    dtype=dtype,
    device=device
).contiguous()

pooling_cache = torch.empty(
    num_pages, num_kv_heads, head_dim,  # same layout as in estimate.py
    dtype=dtype,
    device=device
).contiguous()

current_pages = (seq_len + page_size - 1) // page_size
last_page_len = seq_len - (current_pages - 1) * page_size
kv_page_indptr = torch.tensor([0, current_pages], dtype=torch.int32, device=device)
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device=device)
kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

k_append = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
v_append = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

flashinfer.append_paged_kv_cache_prefill(
    sink,
    k_append,
    v_append,
    (paged_k_cache, paged_v_cache),
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len,
    pooling_cache,
    kv_layout="NHD"
)

paged_k = einops.rearrange(
    paged_k_cache,
    "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
)
paged_v = einops.rearrange(
    paged_v_cache,
    "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
)

torch.testing.assert_close(paged_k[:seq_len, :], k_append)
torch.testing.assert_close(paged_v[:seq_len, :], v_append)

assert paged_v[seq_len:, :].sum() == 0

k_pooled = torch.stack(
    [
        k_append[i * page_size + sink: min((i + 1) * page_size + sink, seq_len), :, :].mean(dim=0)
        for i in range(current_pages - 1)
    ], dim=0
)

assert_close(k_pooled, pooling_cache[:current_pages - 1, :] / 32)

num_decode_iters = 10
for _ in range(num_decode_iters):
    seq_len += 1
    current_pages = (seq_len + page_size - 1) // page_size
    last_page_len = seq_len - (current_pages - 1) * page_size
    kv_page_indptr[1] = current_pages
    kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    k = torch.randn(num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_kv_heads, head_dim, dtype=dtype, device=device)
    flashinfer.append_paged_kv_cache_decode(
        k,
        v,
        (paged_k_cache, paged_v_cache),
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
        pooling_cache,
        kv_layout="NHD"
    )
    k_append = torch.cat([k_append, k.unsqueeze(0)], dim=0)
    v_append = torch.cat([v_append, v.unsqueeze(0)], dim=0)
    paged_k = einops.rearrange(
        paged_k_cache,
        "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
    )
    paged_v = einops.rearrange(
        paged_v_cache,
        "num_pages page_size num_kv_heads head_dim -> (num_pages page_size) num_kv_heads head_dim"
    )

    torch.testing.assert_close(paged_k[:seq_len, :], k_append)
    torch.testing.assert_close(paged_v[:seq_len, :], v_append)
    assert paged_v[seq_len:, :].sum() == 0

    k_pooled = torch.stack(
        [
            k_append[i * page_size + sink: min((i + 1) * page_size + sink, seq_len), :, :].mean(dim=0)
            for i in range(current_pages - 1)
        ], dim=0
    )

    assert_close(k_pooled, pooling_cache[:current_pages - 1, :] / 32)
