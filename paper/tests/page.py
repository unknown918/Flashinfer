import torch
import flashinfer
from paper.core.utils import ceil_div, make_metadata_prefill

torch.manual_seed(42)

device = "cuda:0"
sink_size = 4
page_size = 32
num_kv_head = 8
num_qo_head = 32
head_dim = 128
dtype = torch.float16
max_num_batches = 16

seq_lens = torch.tensor(
    [8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000],
    dtype=torch.int32,
    device=device
).contiguous()
nnz = int(seq_lens.sum())
batch_size = int(seq_lens.shape[0])
max_num_pages = batch_size * 1024

append_k = torch.randn(nnz, num_kv_head, head_dim, dtype=dtype, device=device).contiguous()
append_v = torch.randn(nnz, num_kv_head, head_dim, dtype=dtype, device=device).contiguous()

pooling = torch.zeros(max_num_pages, num_kv_head, head_dim, dtype=dtype, device=device)

# HND: (max_num_pages, num_kv_head, page_size, head_dim)
# NHD: (max_num_pages, page_size, num_kv_head, head_dim)
paged_k_cache = torch.zeros(max_num_pages, num_kv_head, page_size, head_dim, dtype=dtype, device=device)
paged_v_cache = torch.zeros(max_num_pages, num_kv_head, page_size, head_dim, dtype=dtype, device=device)

sink_k_cache = torch.zeros(max_num_batches, num_kv_head, sink_size, head_dim, dtype=dtype, device=device)
sink_v_cache = torch.zeros(max_num_batches, num_kv_head, sink_size, head_dim, dtype=dtype, device=device)

num_pages_per_req = torch.empty(batch_size, dtype=torch.int32, device=device)
kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
kv_last_page_len = torch.empty(batch_size, dtype=torch.int32, device=device)
batch_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
positions = torch.empty(max_num_pages, dtype=torch.int32, device=device)
batch_len_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)

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
    page_size=page_size,
)

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
    kv_layout="HND"
)

for i in range(num_pages_per_req.sum()):
    torch.testing.assert_close(paged_k_cache[i].sum(dim=1), pooling[i], atol=1e-3, rtol=1e-3)
