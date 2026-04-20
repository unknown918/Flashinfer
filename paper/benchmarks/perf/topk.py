import torch
import flashinfer
from flashinfer.testing import bench_gpu_time
import numpy as np

torch.manual_seed(42)

group_size = 4
num_kv_head = 8
num_qo_head = 32

k = 16
page_size = 32
seq_len = 2095
max_length = 32768
num_pages = max_length // page_size
last_page_len = (seq_len - 4) % page_size

logits = torch.randn(num_qo_head, num_pages, device="cuda:0")

indptr = torch.zeros(
    1 + num_kv_head,
    dtype=torch.uint32,
    device=logits.device
)
indices = torch.zeros(
    num_kv_head * (4 + k * group_size * page_size),
    dtype=torch.uint32,
    device=logits.device
)
mask_logits = torch.zeros(
    num_kv_head, num_pages,
    dtype=torch.uint32,
    device=logits.device
)

topk_mask_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.topk_bool_mask_logits(
            topk=k - 1,
            seq_len=seq_len,
            page_size=page_size,
            last_page_len=last_page_len,
            logits=logits,
            indptr=indptr,
            indices=indices,
            group_size=group_size,
            mask_logits=mask_logits
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

print(f"topk mask overhead: {topk_mask_overhead:.3f}ms")
