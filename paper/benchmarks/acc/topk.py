import torch
import flashinfer

torch.manual_seed(42)

num_kv_head = 8
num_qo_head = 32
group_size = num_qo_head // num_kv_head

k = 16
page_size = 32
seq_len = 2054
max_length = 32768
num_total_pages = max_length // page_size
num_valid_pages = (seq_len - 4 + page_size - 1) // page_size
last_page_len = (seq_len - 4) - (num_valid_pages - 1) * page_size

logits = torch.randn(num_qo_head, num_total_pages, dtype=torch.float32, device="cuda:0")


def ref_expand():
    offsets = torch.arange(0, page_size, device="cuda:0")
    first_token_indices = torch.arange(0, 4, device="cuda:0")
    last_token_indices = torch.arange(seq_len - last_page_len, seq_len, device="cuda:0")
    page_indices = torch.topk(logits[:, :num_valid_pages - 1], k=k - 1).indices
    page_indices = page_indices.view(num_kv_head, group_size, -1)
    token_indices = None
    token_indptr = torch.zeros(num_kv_head + 1, dtype=torch.int32, device="cuda:0")
    for i, group in enumerate(page_indices):
        union = torch.unique(group.flatten())
        token_indices_per_group = (union.unsqueeze(-1) * page_size + offsets).reshape(-1) + 4
        token_indices_per_group = torch.cat(
            [
                first_token_indices,
                token_indices_per_group,
                last_token_indices
            ], dim=-1
        )
        if i == 0:
            token_indices = token_indices_per_group * num_kv_head + i
        else:
            token_indices = torch.cat(
                [
                    token_indices,
                    # kv layout: (num_pages, page_size, num_kv_heads)
                    token_indices_per_group * num_kv_head + i
                ], dim=-1
            )
        token_indptr[i + 1] = token_indices_per_group.shape[0]

    return token_indptr, token_indices


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
    num_kv_head, num_total_pages,
    dtype=torch.uint32,
    device=logits.device
)

_indptr, _indices = flashinfer.topk_bool_mask_logits(
    topk=k - 1,
    page_size=page_size,
    last_page_len=last_page_len,
    num_valid_pages=num_valid_pages - 1,  # always select last page
    logits=logits,
    indptr=indptr,
    indices=indices,
    group_size=group_size,
    mask_logits=mask_logits
)

ref_indptr, ref_indices = ref_expand()

assert torch.equal(ref_indptr.to(torch.int32), _indptr.to(torch.int32))
_indptr = _indptr.cumsum(dim=-1)
for i in range(num_kv_head):
    start = _indptr[i]
    end = _indptr[i + 1]
    assert torch.equal(
        ref_indices[start:end].sort().values,
        _indices[start:end].to(torch.int32).sort().values
    )
