import torch
import flashinfer
from flashinfer.testing import bench_gpu_time
import numpy as np

torch.manual_seed(42)

top_k = 16
num_heads = 32
head_dim = 128
seq_len = 1024


logits = torch.randn(num_heads, seq_len).half().to(0)

radix_topk_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.topk.radix_top_k_mask_logits(logits, top_k),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)
topk_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.sampling.top_k_mask_logits(logits, top_k),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

print(f"radix_topk_overhead: {radix_topk_overhead:.3f}ms")
print(f"topk_overhead: {topk_overhead:.3f}ms")
