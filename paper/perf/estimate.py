import torch
import flashinfer
import numpy as np
from flashinfer.testing import bench_gpu_time

device = "cuda"
torch.manual_seed(42)

head_dim = 128
page_size = 32
num_kv_heads = 8
num_qo_heads = 32
seq_len = 8899
max_length = 32768
dtype = torch.float16

assert num_qo_heads % num_kv_heads == 0
group_size = num_qo_heads // num_kv_heads

num_total_pages = (max_length + page_size - 1) // page_size
num_valid_pages = (seq_len + page_size - 1) // page_size

query = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=device)
pooling = torch.zeros(
    num_total_pages, num_kv_heads, head_dim,
    dtype=dtype, device=device
).contiguous()
pooling_ = torch.randn(num_valid_pages, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
pooling[:num_valid_pages] = pooling_
out = torch.zeros(
    num_qo_heads, num_total_pages,
    dtype=dtype, device=device
)

estimate_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.decode.estimate(
            query=query,
            pooling=pooling,
            num_valid_pages=num_valid_pages,
            out=out,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True,
    )
)

print(f"estimate overhead: {estimate_overhead * 1e3:.3f}us")