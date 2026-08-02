import torch
import flashinfer

num_layers = 32
num_qo_heads = 64
num_kv_heads = 16
head_dim = 128
max_num_pages = 128
page_size = 16
# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)
batch_size = 7
shared_prefix_len = 8192
nnz_qo = 100
qo_indptr = torch.tensor(
    [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
)
paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
paged_kv_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)
# 1 <= paged_kv_last_page_len <= page_size
paged_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)
kv_cache_at_layer = [
    torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]
shared_k_data_at_layer = [
    torch.randn(
        shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]
shared_v_data_at_layer = [
    torch.randn(
        shared_prefix_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ) for _ in range(num_layers)
]
# create auxiliary data structures for batch prefill attention
prefill_wrapper.begin_forward(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
)
outputs = []
for i in range(num_layers):
    q = torch.randn(nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    kv_cache = kv_cache_at_layer[i]
    k_shared = shared_k_data_at_layer[i]
    v_shared = shared_v_data_at_layer[i]
    # compute batch prefill attention, reuse auxiliary data structures
    o = prefill_wrapper.forward(
        q, k_shared, v_shared, kv_cache, causal=True
    )
    outputs.append(o)
