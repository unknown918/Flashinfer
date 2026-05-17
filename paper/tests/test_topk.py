import pytest
import torch
import flashinfer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1021, 1022, 1023, 1024, 3351, 8192, 16384, 32760])
@pytest.mark.parametrize("page_size", [16, 32])  # fixme: page_size=64 has bug
def test_topk_bool_mask_logits(seq_len, page_size):
    torch.manual_seed(42)

    device = torch.device("cuda:0")

    num_kv_head = 8
    num_qo_head = 32
    group_size = num_qo_head // num_kv_head
    k = 16
    max_length = 32768

    num_total_pages = max_length // page_size
    num_valid_pages = (seq_len - 4 + page_size - 1) // page_size
    last_page_len = (seq_len - 4) - (num_valid_pages - 1) * page_size
    meta_data = torch.zeros(3, dtype=torch.int32, device="cuda")
    meta_data[0] = num_valid_pages
    meta_data[1] = last_page_len
    row_states_buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    assert last_page_len > 0
    assert num_valid_pages > 0

    logits = torch.randn(
        num_qo_head, num_total_pages,
        dtype=torch.bfloat16,
        device=device
    )

    def ref_expand():
        offsets = torch.arange(0, page_size, device=device)
        first_token_indices = torch.arange(0, 4, device=device)
        last_token_indices = torch.arange(
            seq_len - last_page_len, seq_len, device=device
        )

        page_indices = torch.topk(
            logits[:, :num_valid_pages - 1], k=k - 1
        ).indices

        page_indices = page_indices.view(num_kv_head, group_size, -1)

        token_indices = None
        token_indptr = torch.zeros(
            num_kv_head + 1,
            dtype=torch.int32,
            device=device
        )

        for idx, group in enumerate(page_indices):
            union = torch.unique(group.flatten())

            token_indices_per_group = (union.unsqueeze(-1) * page_size + offsets).reshape(-1) + 4

            token_indices_per_group = torch.cat(
                [
                    first_token_indices,
                    token_indices_per_group,
                    last_token_indices
                ],
                dim=-1
            )

            kv_layout_indices = token_indices_per_group * num_kv_head + idx

            if idx == 0:
                token_indices = kv_layout_indices
            else:
                token_indices = torch.cat(
                    [token_indices, kv_layout_indices], dim=-1
                )

            token_indptr[idx + 1] = token_indices_per_group.shape[0]

        return token_indptr, token_indices

    indptr = torch.zeros(
        1 + num_kv_head,
        dtype=torch.uint32,
        device=device
    )

    indices = torch.zeros(
        num_kv_head * (4 + k * group_size * page_size),
        dtype=torch.uint32,
        device=device
    )

    mask_logits = torch.zeros(
        num_kv_head,
        num_total_pages,
        dtype=torch.uint32,
        device=device
    )

    _indptr, _indices = flashinfer.topk_bool_mask_logits(
        top_k=k - 1,
        page_size=page_size,
        meta_data=meta_data,
        logits=logits,
        indptr=indptr,
        indices=indices,
        group_size=group_size,
        mask_logits=mask_logits,
        row_states_buffer=row_states_buffer
    )

    _indptr = _indptr.to(torch.int32)

    ref_indptr, ref_indices = ref_expand()

    assert torch.equal(ref_indptr, _indptr)

    _indptr = _indptr.cumsum(dim=-1)

    for i in range(num_kv_head):
        start = _indptr[i]
        end = _indptr[i + 1]

        ref_slice = ref_indices[start:end].sort().values
        out_slice = _indices[start:end].to(torch.int32).sort().values

        assert torch.equal(ref_slice, out_slice)
