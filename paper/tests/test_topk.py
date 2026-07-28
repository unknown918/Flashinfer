import pytest
import torch
import flashinfer


def make_logits(
        num_qo_head: int,
        num_valid_pages: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int = 42,
):
    assert dtype in (torch.float16, torch.bfloat16), "only fp16/bf16 supported"

    base = 0x3000 if dtype == torch.float16 else 0x3F00

    bits = (torch.arange(num_valid_pages, device=device, dtype=torch.int32) + base).to(torch.int16)
    logits = bits.view(dtype).unsqueeze(0).expand(num_qo_head, -1).clone()

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    perm = torch.randperm(num_valid_pages, device=device, generator=g)
    logits = logits.index_select(1, perm)

    return logits


boundaries = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    131062,
]

seq_lens = sorted({
    boundary + delta
    for boundary in boundaries
    for delta in range(-8, 9)
    if boundary + delta > 0
})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [8, 15, 16, 32, 64])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("max_length", [32768, 131072])
@pytest.mark.parametrize("seq_len", seq_lens)
def test_topk_bool_mask_logits(max_length, seq_len, page_size, topk, dtype):
    device = torch.device("cuda:0")

    batch_size = 4
    num_kv_head = 8  * batch_size
    num_qo_head = 32  * batch_size
    group_size = num_qo_head // num_kv_head

    if topk * page_size >= seq_len:
        pytest.skip(f"invalid case: topk * page_size = {topk * page_size} >= {seq_len}")
    if seq_len > max_length:
        pytest.skip(f"invalid case: seq_len {seq_len} >= max_length {max_length}")

    num_total_pages = max_length // page_size
    num_valid_pages = (seq_len - 4 + page_size - 1) // page_size - 1
    last_page_len = (seq_len - 4) - num_valid_pages * page_size
    meta_data_device = torch.zeros(2, dtype=torch.uint32, device=device)
    meta_data_device[0] = num_valid_pages
    meta_data_device[1] = last_page_len
    row_states_buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    assert last_page_len > 0
    assert num_valid_pages > 0

    logits = torch.zeros(
        num_qo_head, num_total_pages,
        dtype=dtype, device=device
    )

    logits_ = make_logits(num_qo_head, num_valid_pages, dtype, device, seed=42)
    logits[:, :num_valid_pages] = logits_

    def ref_expand():
        offsets = torch.arange(0, page_size, device=device)
        first_token_indices = torch.arange(0, 4, device=device)
        last_token_indices = torch.arange(seq_len - last_page_len, seq_len, device=device)

        page_indices = torch.topk(logits[:, :num_valid_pages], k=topk - 1).indices
        page_indices = page_indices.reshape(num_kv_head, group_size, -1)

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
        num_kv_head * (4 + topk * group_size * page_size),
        dtype=torch.uint32,
        device=device
    )

    mask_logits = torch.zeros(
        num_kv_head,
        num_total_pages,
        dtype=torch.uint32,
        device=device
    )

    flashinfer.topk_bool_mask_logits(
        top_k=topk - 1,
        page_size=page_size,
        meta_data=meta_data_device,
        logits=logits,
        indptr=indptr,
        indices=indices,
        group_size=group_size,
        mask_logits=mask_logits,
        row_states_buffer=row_states_buffer
    )

    indptr = indptr.to(torch.int32)

    ref_indptr, ref_indices = ref_expand()

    assert torch.equal(ref_indptr, indptr)

    indptr = indptr.cumsum(dim=-1)

    for i in range(num_kv_head):
        start = indptr[i]
        end = indptr[i + 1]

        ref_slice = ref_indices[start:end].sort().values
        out_slice = indices[start:end].to(torch.int32).sort().values

        assert torch.equal(ref_slice, out_slice)

    torch.cuda.empty_cache()
