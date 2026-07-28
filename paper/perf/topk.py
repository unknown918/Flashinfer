import torch
import flashinfer
import numpy as np
from flashinfer.testing import bench_gpu_time

device = "cuda:0"
torch.manual_seed(42)

k = 2
page_size = 16
seq_len = 16000
max_length = 32768

batch_size = 1
num_kv_head = 8 * batch_size
num_qo_head = 32 * batch_size
group_size = num_qo_head // num_kv_head

num_total_pages = max_length // page_size
num_valid_pages = (seq_len - 4 + page_size - 1) // page_size
last_page_len = (seq_len - 4) - (num_valid_pages - 1) * page_size
meta_data = torch.zeros(2, dtype=torch.int32, device=device)
meta_data[0] = num_valid_pages - 1

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

logits[:, :num_valid_pages] = logits_

sparse_indptr = torch.zeros(
    1 + num_kv_head,
    dtype=torch.int32,
    device=device
)

sparse_indices = torch.zeros(
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
            logits=logits,
            mask_logits=mask_logits,
            indptr=indptr,
            indices=indices,
            sparse_indptr=sparse_indptr,
            sparse_indices=sparse_indices,
            top_k=k - 1,
            meta_data=meta_data,
            group_size=group_size,
            row_states_buffer=row_states_buffer,
            is_bench=True
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

_mask_logits = torch.zeros(
    num_qo_head, num_valid_pages,
    dtype=torch.int32,
    device=device
)
_sparse_indptr = torch.arange(
    1 + num_kv_head,
    dtype=torch.int32,
    device=device
)
_sparse_indices = torch.arange(
    num_kv_head * (4 + k * group_size * page_size),
    dtype=torch.int32,
    device=device
)


def clean_tensor():
    _sparse_indptr.zero_()
    _sparse_indices.zero_()
    _mask_logits.zero_()


clean_overhead = np.median(
    bench_gpu_time(
        lambda: clean_tensor(),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

print(f"total_overhead: {topk_overhead * 1e3:.3f}us")
print(f"clean_overhead: {clean_overhead * 1e3:.3f}us")
print(f"topk_overhead: {(topk_overhead - clean_overhead) * 1e3:.3f}us")
