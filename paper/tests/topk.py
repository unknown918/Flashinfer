import torch
import flashinfer
import numpy as np
from flashinfer.testing import bench_gpu_time
from paper.core.utils import make_metadata_prefill, ceil_div, make_logits

device = "cuda:0"
dtype = torch.float16
torch.manual_seed(42)

k = 32
sink_size = 4
page_size = 32
num_kv_head = 8
num_qo_head = 32
group_size = num_qo_head // num_kv_head

seq_lens = torch.tensor(
    [8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000],
    dtype=torch.int32,
    device=device
).contiguous()
batch_size = seq_lens.shape[0]
nnz = int(seq_lens.sum(dim=0))
max_num_pages = 1024
total_pages = max_num_pages * batch_size

num_pages_per_req = torch.zeros(batch_size, dtype=torch.int32, device=device)
kv_indices = torch.zeros(total_pages, dtype=torch.int32, device=device)
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
batch_indices = torch.zeros(total_pages, dtype=torch.int32, device=device)
positions = torch.zeros(total_pages, dtype=torch.int32, device=device)
batch_len_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

make_metadata_prefill(
    seq_lens,
    num_pages_per_req=num_pages_per_req,
    kv_indices=kv_indices,
    kv_indptr=kv_indptr,
    kv_last_page_len=kv_last_page_len,
    batch_indices=batch_indices,
    positions=positions,
    batch_len_indptr=batch_len_indptr,
    sink_size=sink_size,
    page_size=page_size
)

logits = torch.zeros(
    batch_size * num_qo_head, max_num_pages,
    dtype=dtype,
    device=device
)

ref_indices = torch.tensor([], dtype=torch.int32, device=device)
ref_indptr = torch.zeros(batch_size * num_kv_head + 1, dtype=torch.int32, device=device)


def ref_expand(_logits, batch_id, ref_indptr, ref_indices):
    page_topk = torch.topk(_logits[:, :-1], k=k).indices
    page_topk = page_topk.reshape(num_kv_head, group_size, -1)
    _indices = kv_indices[kv_indptr[batch_id]: kv_indptr[batch_id + 1]]

    for kv_head_idx, group_topk in enumerate(page_topk):
        group_topk = group_topk.flatten()
        # union_overhead = np.median(
        #     bench_gpu_time(
        #         lambda: torch.unique(
        #             group,
        #         ),
        #         dry_run_time_ms=100,
        #         repeat_time_ms=1000,
        #     )
        # )
        # print(f"union_overhead: {union_overhead * 1e3:.3f}us")
        group_topk = torch.unique(group_topk)
        group_indices = torch.cat(
            [
                _indices[group_topk],
                _indices[-1:]
            ],
            dim=-1,
        )
        group_indices = group_indices * num_kv_head + kv_head_idx
        ref_indices = torch.cat(
            [ref_indices, group_indices],
            dim=-1
        )
        ref_indptr[batch_id * num_kv_head + kv_head_idx + 1] = group_indices.size(0)

    return ref_indptr, ref_indices


for batch_idx, seq_len in enumerate(seq_lens):
    seq_len = int(seq_len) - sink_size
    logits_ = make_logits(
        num_qo_head, ceil_div(seq_len, page_size),
        device=device,
        dtype=dtype,
    )
    ref_indptr, ref_indices = ref_expand(logits_, batch_idx, ref_indptr, ref_indices)
    logits[batch_idx * num_qo_head:(batch_idx + 1) * num_qo_head, :ceil_div(seq_len, page_size)] = logits_

logits.contiguous()

mask_logits = torch.zeros(
    batch_size * num_kv_head, max_num_pages,
    dtype=torch.int32,
    device=device
)

sparse_indptr = torch.zeros(
    1 + batch_size * num_kv_head,
    dtype=torch.int32,
    device=device
)

sparse_indices = torch.zeros(
    batch_size * num_kv_head * k * group_size,
    dtype=torch.int32,
    device=device
)

topk_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.topk_bool_mask_logits(
            logits=logits,
            mask_logits=mask_logits,
            indptr=kv_indptr,
            indices=kv_indices,
            sparse_indptr=sparse_indptr,
            sparse_indices=sparse_indices,
            top_k=k,
            group_size=group_size,
            num_pages_per_req=num_pages_per_req - 1,
            is_bench=True
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)
sparse_indptr[1:] += 1
assert torch.equal(sparse_indptr, ref_indptr)

torch.cumsum(ref_indptr, dim=-1, out=ref_indptr)

for idx in range(num_kv_head * batch_size):
    assert torch.equal(
        sparse_indices[ref_indptr[idx]:ref_indptr[idx + 1]].sort().values,
        ref_indices[ref_indptr[idx]:ref_indptr[idx + 1]].sort().values
    )

assert ref_indices[ref_indptr[-1]:].sum() == 0

print(f"topk_overhead: {topk_overhead * 1e3:.3f}us")
