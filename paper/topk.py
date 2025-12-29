import torch
import flashinfer
from flashinfer.testing import bench_gpu_time
import numpy as np

torch.manual_seed(42)
batch_size = 32
vocab_size = 4096
top_k = 16
logits = torch.randn(batch_size, vocab_size).half().to(0)

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
