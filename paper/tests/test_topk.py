import pytest
import torch
import flashinfer
from paper.core.utils import make_metadata_prefill, ceil_div, make_logits

device = "cuda:0"
torch.manual_seed(42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [8, 15, 16, 32, 64])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("max_length", [32768])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_kv_heads, num_qo_heads, sink_size", [(8, 32, 4)])
@pytest.mark.parametrize("seq_len", [1028, 2000, 4000, 8000, 16000, 32000])
def test_topk_bool_mask_logits(
        dtype,
        topk,
        page_size,
        max_length,
        batch_size,
        num_kv_heads,
        num_qo_heads,
        sink_size,
        seq_len,
):
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device).contiguous()
    seq_lens.fill_(seq_len)

    if topk * page_size >= int(seq_len) - sink_size:
        pytest.skip(f"invalid case: topk * page_size = {topk * page_size} >= {seq_len - sink_size}")
    if seq_len > max_length:
        pytest.skip(f"invalid case: seq_len {seq_len} >= max_length {max_length}")

    group_size = num_qo_heads // num_kv_heads
    max_num_pages = int(max_length / page_size)
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
        batch_size * num_qo_heads, max_num_pages,
        dtype=dtype,
        device=device
    )

    ref_indices = torch.tensor([], dtype=torch.int32, device=device)
    ref_indptr = torch.zeros(batch_size * num_kv_heads + 1, dtype=torch.int32, device=device)

    def ref_expand(_logits, batch_id, ref_indptr, ref_indices):
        page_topk = torch.topk(_logits[:, :-1], k=topk).indices
        page_topk = page_topk.reshape(num_kv_heads, group_size, -1)
        _indices = kv_indices[kv_indptr[batch_id]: kv_indptr[batch_id + 1]]

        for kv_head_idx, group_topk in enumerate(page_topk):
            group_topk = group_topk.flatten()
            group_topk = torch.unique(group_topk)
            group_indices = torch.cat(
                [
                    _indices[group_topk],
                    _indices[-1:]
                ],
                dim=-1,
            )
            group_indices = group_indices * num_kv_heads + kv_head_idx
            ref_indices = torch.cat(
                [ref_indices, group_indices],
                dim=-1
            )
            ref_indptr[batch_id * num_kv_heads + kv_head_idx + 1] = group_indices.size(0)

        return ref_indptr, ref_indices

    for batch_idx, seq_len in enumerate(seq_lens):
        seq_len = int(seq_len) - sink_size
        logits_ = make_logits(
            num_qo_heads, ceil_div(seq_len, page_size),
            device=device,
            dtype=dtype,
        )
        ref_indptr, ref_indices = ref_expand(logits_, batch_idx, ref_indptr, ref_indices)
        logits[batch_idx * num_qo_heads:(batch_idx + 1) * num_qo_heads, :ceil_div(seq_len, page_size)] = logits_

    logits.contiguous()

    mask_logits = torch.zeros(
        batch_size * num_kv_heads, max_num_pages,
        dtype=torch.int32,
        device=device
    )

    sparse_indptr = torch.zeros(
        1 + batch_size * num_kv_heads,
        dtype=torch.int32,
        device=device
    )

    sparse_indices = torch.zeros(
        batch_size * num_kv_heads * topk * group_size,
        dtype=torch.int32,
        device=device
    )

    num_pages_per_req.add_(-1)

    flashinfer.topk_bool_mask_logits(
        logits=logits,
        mask_logits=mask_logits,
        indptr=kv_indptr,
        indices=kv_indices,
        sparse_indptr=sparse_indptr,
        sparse_indices=sparse_indices,
        top_k=topk,
        group_size=group_size,
        num_pages_per_req=num_pages_per_req,
    )

    sparse_indptr[1:].add_(1)
    assert torch.equal(sparse_indptr, ref_indptr)

    torch.cumsum(ref_indptr, dim=-1, out=ref_indptr)

    for idx in range(num_kv_heads * batch_size):
        assert torch.equal(
            sparse_indices[ref_indptr[idx]:ref_indptr[idx + 1]].sort().values,
            ref_indices[ref_indptr[idx]:ref_indptr[idx + 1]].sort().values
        )

    assert ref_indices[ref_indptr[-1]:].sum() == 0
    torch.cuda.empty_cache()
