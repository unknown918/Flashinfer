import torch
import flashinfer
from paper.utils.utils import ceil_div, make_metadata_prefill

torch.manual_seed(42)

device = "cuda:0"
sink_size = 4
page_size = 16
num_kv_head = 8
num_qo_head = 32
head_dim = 128
dtype = torch.float16
max_num_pages = 1024
max_num_batches = 16

seq_lens_cpu = [45, 800, 25, 22]
nnz = sum(seq_lens_cpu)
batch_size = len(seq_lens_cpu)
total_pages = sum(ceil_div(seq_len, page_size) for seq_len in seq_lens_cpu)

seq_lens = torch.tensor(
    seq_lens_cpu,
    dtype=torch.int32,
    device=device,
)

append_k = torch.randn(nnz, num_kv_head, head_dim, dtype=dtype, device=device).contiguous()
append_v = torch.randn(nnz, num_kv_head, head_dim, dtype=dtype, device=device).contiguous()

pooling = torch.zeros(max_num_pages, num_kv_head, head_dim, dtype=dtype, device=device)
paged_k_cache = torch.zeros(max_num_pages, page_size, num_kv_head, head_dim, dtype=dtype, device=device)
paged_v_cache = torch.zeros(max_num_pages, page_size, num_kv_head, head_dim, dtype=dtype, device=device)

sink_k_cache = torch.zeros(max_num_batches, sink_size, num_kv_head, head_dim, dtype=dtype, device=device)
sink_v_cache = torch.zeros(max_num_batches, sink_size, num_kv_head, head_dim, dtype=dtype, device=device)

num_pages_per_req = torch.empty(batch_size, dtype=torch.int32, device=device)
kv_indices = torch.empty(total_pages, dtype=torch.int32, device=device)
kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
kv_last_page_len = torch.empty(batch_size, dtype=torch.int32, device=device)
batch_indices = torch.empty(total_pages, dtype=torch.int32, device=device)
positions = torch.empty(total_pages, dtype=torch.int32, device=device)
batch_len_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)

metadata = {
    "num_pages_per_req": num_pages_per_req,
    "kv_indices": kv_indices,
    "kv_indptr": kv_indptr,
    "kv_last_page_len": kv_last_page_len,
    "batch_indices": batch_indices,
    "positions": positions,
    "batch_len_indptr": batch_len_indptr,
}

make_metadata_prefill(seq_lens, metadata, sink_size=4, page_size=16)

flashinfer.append_paged_kv_cache_prefill(
    append_key=append_k,
    append_value=append_v,
    paged_kv_cache=(paged_k_cache, paged_v_cache),
    kv_indices=kv_indices,
    kv_indptr=kv_indptr,
    kv_last_page_len=kv_last_page_len,
    pooling=pooling,
    sink_k=sink_k_cache,
    sink_v=sink_v_cache,
    batch_indices=batch_indices,
    positions=positions,
    batch_len_indptr=batch_len_indptr,
    kv_layout="NHD"
)

for i in range(num_pages_per_req.sum()):
    torch.testing.assert_close(paged_k_cache[i].sum(dim=0), pooling[i], atol=1e-3, rtol=1e-3)

query = torch.randn(batch_size, num_qo_head, head_dim, dtype=dtype, device=device).contiguous()
out = torch.zeros(batch_size * num_qo_head, max_num_pages, dtype=dtype, device=device).contiguous()

flashinfer.estimate(
    query=query,
    pooling=pooling,
    kv_indptr=kv_indptr,
    kv_indices=kv_indices,
    out=out,
)

group_size = 4
for batch_idx in range(batch_size):
    start = kv_indptr[batch_idx]
    end = kv_indptr[batch_idx + 1]
    for qo_head_idx in range(num_qo_head):
        ref_out = torch.matmul(
            query[batch_idx][qo_head_idx],
            pooling[start:end, int(qo_head_idx / group_size)].transpose(0, 1)
        )
        torch.testing.assert_close(ref_out, out[batch_idx * num_qo_head + qo_head_idx, :end - start])

print("Hello")
