import torch
import flashinfer
from flashinfer.testing.utils import bench_gpu_time
import numpy as np

torch.manual_seed(42)

top_k = 64
page_size = 32
seq_len = 32768
head_dim = 128
num_kv_heads = 8
num_qo_heads = 32
max_length = 131072

num_pages = (max_length + page_size - 1) // page_size
# NHD
paged_kv_cache = torch.empty(num_pages, 2, page_size, num_kv_heads, head_dim).half().to(0).contiguous()

k_append = torch.randn(seq_len, num_kv_heads, head_dim).half().to(0)
v_append = torch.randn(seq_len, num_kv_heads, head_dim).half().to(0)

current_pages = (seq_len + page_size - 1) // page_size
pooling = torch.empty(current_pages, num_kv_heads, head_dim).half().to(0).contiguous()
estimate_attn_scores = torch.empty(num_qo_heads, current_pages).half().to(0)
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
    pooling
)

kv_last_page_len = torch.tensor([last_page_len + 1], dtype=torch.int32, device="cuda:0")
kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")

q = torch.randn(num_qo_heads, head_dim).half().to(0)
k = torch.randn(num_kv_heads, head_dim).half().to(0)
v = torch.randn(num_kv_heads, head_dim).half().to(0)

append_page_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.append_paged_kv_cache_decode(
            k,
            v,
            paged_kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
            pooling
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

estimation_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.decode.estimate(
            q,
            pooling,
            estimate_attn_scores
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

topk_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.topk.radix_top_k_mask_logits(estimate_attn_scores, top_k),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

print(f"Append Page Overhead: {append_page_overhead:.3f} ms")
print(f"Estimation Overhead:  {estimation_overhead:.3f} ms")
print(f"TopK Overhead:  {topk_overhead:.3f} ms")
