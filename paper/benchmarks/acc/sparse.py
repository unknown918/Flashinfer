import torch
import flashinfer

torch.manual_seed(42)

topk = 16
page_size = 16
seq_len = 2054
max_length = 32768
num_kv_head = 8
num_qo_head = 32
head_dim = 128
group_size = num_qo_head // num_kv_head
num_total_pages = max_length // page_size
num_valid_pages = (seq_len - 4 + page_size - 1) // page_size
last_page_len = seq_len - 4 - (num_valid_pages - 1) * page_size

logits = torch.randn(num_qo_head, num_total_pages, dtype=torch.float32, device="cuda:0")

num_blocks_col = num_valid_pages + 1  # sink
block_row_sz = torch.tensor([[1]], dtype=torch.int32, device="cuda").repeat(num_kv_head, 1)
block_col_sz = torch.zeros(num_blocks_col, dtype=torch.int32)
block_col_sz[0] = 4
block_col_sz[-1] = last_page_len

for i in range(1, num_blocks_col):
    block_col_sz[i] = min(page_size, seq_len - 4 - (i - 1) * page_size)

block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_head, 1)
block_mask_map = torch.zeros(num_kv_head, 1, num_blocks_col, dtype=torch.int32)

indices = torch.topk(logits[:, :num_valid_pages - 1], k=topk - 1).indices.to("cpu")
for i in range(num_kv_head):
    for j in range(group_size):
        block_mask_map[i, 0, indices[i * j + j]] = 1
    block_mask_map[i, 0, -1] = 1  # always select last page

q = torch.randn(num_qo_head, 1, head_dim, dtype=torch.half, device="cuda")
k = torch.randn(num_kv_head, seq_len, head_dim, dtype=torch.half, device="cuda")
v = torch.randn(num_kv_head, seq_len, head_dim, dtype=torch.half, device="cuda")

float_workspace_buffer = torch.empty(
    128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
)
sparse_wrapper_fa2 = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
    float_workspace_buffer, backend="fa2"
)

sparse_wrapper_fa2.plan(
    block_mask_map=block_mask_map,
    block_row_sz=block_row_sz,
    block_col_sz=block_col_sz,
    num_qo_heads=num_qo_head,
    num_kv_heads=num_kv_head,
    head_dim=head_dim,
    q_data_type=torch.half,
)

attn_ref = sparse_wrapper_fa2.run(q, k, v)

indptr = torch.zeros(
    1 + num_kv_head,
    dtype=torch.uint32,
    device=logits.device
)
indices = torch.zeros(
    num_kv_head * (4 + topk * group_size * page_size),
    dtype=torch.uint32,
    device=logits.device
)
mask_logits = torch.zeros(
    num_kv_head, num_total_pages,
    dtype=torch.uint32,
    device=logits.device
)

_indptr, _indices = flashinfer.topk_bool_mask_logits(
    topk=topk - 1,
    page_size=page_size,
    last_page_len=last_page_len,
    num_valid_pages=num_valid_pages - 1,  # always select last page
    logits=logits,
    indptr=indptr,
    indices=indices,
    group_size=group_size,
    mask_logits=mask_logits
)

_indptr = _indptr.cumsum(dim=-1)
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
decode_wrapper = flashinfer.SparseSinkAttentionWrapper(workspace_buffer)

decode_wrapper.plan(
    kv_indptr=_indptr.to(torch.int32),
    kv_indices=_indices.to(torch.int32),
    num_qo_heads=num_qo_head,
    num_kv_heads=num_kv_head,
    head_dim=head_dim,
    causal=True
)

q = q.transpose(0, 1).contiguous()[0]
k = k.transpose(0, 1).unsqueeze(1).contiguous()
v = v.transpose(0, 1).unsqueeze(1).contiguous()

attn_out = decode_wrapper.run(
    q=q,
    k=k,
    v=v,
    return_lse=False
)

torch.testing.assert_close(attn_out, attn_ref[:, 0])
