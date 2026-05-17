import torch
import flashinfer
import numpy as np
from flashinfer.testing import bench_gpu_time

device = "cuda"
torch.manual_seed(42)

k = 32
page_size = 32
seq_len = 8199
max_length = 32768

num_kv_head = 8
num_qo_head = 32
group_size = num_qo_head // num_kv_head

num_total_pages = max_length // page_size
num_valid_pages = (seq_len - 4 + page_size - 1) // page_size
last_page_len = (seq_len - 4) - (num_valid_pages - 1) * page_size
meta_data = torch.zeros(3, dtype=torch.int32, device="cuda")
meta_data[0] = num_valid_pages
meta_data[1] = last_page_len

logits = torch.zeros(
    num_qo_head, num_total_pages,
    dtype=torch.float16,
    device=device
)

logits_ = torch.randn(
    num_qo_head, num_valid_pages,
    dtype=torch.float16,
    device=device
)

indptr = torch.zeros(
    1 + num_kv_head,
    dtype=torch.int32,
    device=device
)

indices = torch.zeros(
    num_kv_head * (4 + k * group_size * page_size),
    dtype=torch.int32,
    device=device
)

mask_logits = torch.zeros(
    num_kv_head,
    num_total_pages,
    dtype=torch.int32,
    device=device
)
row_states_buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

topk_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.topk_bool_mask_logits(
            top_k=k - 1,
            page_size=page_size,
            meta_data=meta_data,
            logits=logits,
            indptr=indptr,
            indices=indices,
            group_size=group_size,
            mask_logits=mask_logits,
            row_states_buffer=row_states_buffer
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

print(f"topk_overhead: {topk_overhead * 1e3:.3f}us")
