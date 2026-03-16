import torch
import flashinfer
from flashinfer.testing import bench_gpu_time
import numpy as np

torch.manual_seed(42)
batch_size = 32
vocab_size = 1024
k = 16
logits = torch.randn(batch_size, vocab_size, device="cuda")

topk_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.top_k(logits, k),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

_, indices = flashinfer.top_k(logits, k)

topk_mask_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.topk_bool_mask_logits(logits, k, max_length=1024),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

mask = flashinfer.topk_bool_mask_logits(logits, k, max_length=1024)

assert torch.gather(mask, dim=1, index=indices).sum() == k * batch_size

mask = mask.reshape(-1, 4, mask.shape[1]).any(1)

print(f"topk overhead: {topk_overhead:.3f}ms")
print(f"topk mask overhead: {topk_mask_overhead:.3f}ms")
