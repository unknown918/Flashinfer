import torch
import flashinfer
from flashinfer.testing.utils import bench_gpu_time
import numpy as np

torch.manual_seed(42)


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 2e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


seq_len = 1024
head_dim = 128
page_size = 32
num_kv_heads = 8
num_qo_heads = 32
max_length = 131072

num_pages = (max_length + page_size - 1) // page_size

q = torch.randn(num_qo_heads, head_dim).half().to(0)
pooling = torch.randn(num_pages, num_kv_heads, head_dim).half().to(0)
out = torch.empty(num_qo_heads, num_pages).half().to(0)

flashinfer.decode.estimate(
    q,
    pooling,
    seq_len,
    out
)

estimation_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.decode.estimate(
            q,
            pooling,
            seq_len,
            out
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

print(f"Estimation Overhead:  {estimation_overhead:.3f} ms")
pooling = pooling.repeat_interleave(4, dim=1)
ref = torch.einsum("hd,thd->th", q, pooling[:seq_len])
assert_close(out[:, :seq_len].transpose(0, 1), ref)
