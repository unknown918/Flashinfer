import pytest
import torch
import flashinfer

torch.manual_seed(3407)


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [8, 15, 16, 32, 60])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("seq_len", [1021, 1022, 1023, 1024, 3351, 8192, 16384, 32760])
def test_sparse_sink_attention(dtype, topk, page_size, seq_len):
    max_length = 32768
    num_kv_head = 8
    num_qo_head = 32
    head_dim = 128
    num_sink = 4

    if topk * page_size >= seq_len:
        pytest.skip(f"invalid case: topk * page_size = {topk * page_size} >= {seq_len}")

    device = torch.device("cuda:1")
    torch.cuda.empty_cache()

    group_size = num_qo_head // num_kv_head
    num_total_pages = max_length // page_size
    row_states_buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    assert num_qo_head % num_kv_head == 0, "invalid head grouping"

    num_valid_pages = (seq_len - num_sink + page_size - 1) // page_size - 1
    last_page_len = seq_len - num_sink - num_valid_pages * page_size
    meta_data = torch.zeros(2, dtype=torch.int32, device=device)
    meta_data[0] = num_valid_pages
    meta_data[1] = last_page_len

    logits = torch.randn(
        num_qo_head, num_total_pages,
        dtype=torch.float32, device=device
    )

    num_blocks_col = num_valid_pages + 2  # sink + window

    block_row_sz = torch.full(
        (num_kv_head, 1), 1,
        dtype=torch.int32, device=device
    )

    block_col_sz = torch.zeros(
        num_blocks_col, dtype=torch.int32, device=device
    )
    block_col_sz[0] = num_sink
    block_col_sz[-1] = last_page_len

    for i in range(1, num_blocks_col):
        block_col_sz[i] = min(page_size, seq_len - num_sink - (i - 1) * page_size)

    block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_head, 1)

    block_mask_map = torch.zeros(
        num_kv_head, 1, num_blocks_col,
        dtype=torch.int32, device=device
    )

    indices = torch.topk(
        logits[:, :num_valid_pages],
        k=topk - 1
    ).indices.cpu()

    indices += 1  # include sink tokens

    for i in range(num_kv_head):
        for j in range(group_size):
            block_mask_map[i, 0, indices[i * group_size + j]] = 1

        block_mask_map[i, 0, 0] = 1
        block_mask_map[i, 0, -1] = 1

    q = torch.randn(num_qo_head, 1, head_dim, dtype=dtype, device=device)
    k = torch.randn(num_kv_head, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(num_kv_head, seq_len, head_dim, dtype=dtype, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref_wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        workspace, backend="fa2"
    )

    ref_wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_head,
        num_kv_heads=num_kv_head,
        head_dim=head_dim,
        q_data_type=dtype,
    )

    attn_ref = ref_wrapper.run(q, k, v)

    indptr = torch.zeros(
        1 + num_kv_head,
        dtype=torch.uint32,
        device=device
    )

    indices = torch.zeros(
        num_kv_head * (num_sink + topk * group_size * page_size),
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
        meta_data=meta_data,
        logits=logits,
        indptr=indptr,
        indices=indices,
        group_size=group_size,
        mask_logits=mask_logits,
        row_states_buffer=row_states_buffer
    )

    indptr = indptr.cumsum(dim=-1)

    decode_workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    decode_wrapper = flashinfer.SparseSinkAttentionWrapper(
        num_kv_heads=num_kv_head,
        device=device,
        float_workspace_buffer=decode_workspace,
    )

    decode_wrapper.plan(
        kv_indptr=indptr.to(torch.int32),
        kv_indices=indices.to(torch.int32),
        q_data_type=dtype,
        kv_data_type=dtype,
        num_qo_heads=num_qo_head,
        num_kv_heads=num_kv_head,
        head_dim=head_dim,
        causal=True
    )

    q_decode = q.transpose(0, 1)[0].contiguous()
    k_decode = k.transpose(0, 1).unsqueeze(1).contiguous()
    v_decode = v.transpose(0, 1).unsqueeze(1).contiguous()

    out = torch.zeros(
        num_qo_head,
        head_dim,
        dtype=dtype,
        device=device
    )

    attn_out = decode_wrapper.run(
        q=q_decode,
        k=k_decode,
        v=v_decode,
        kv_indptr=indptr.to(torch.int32),
        kv_indices=indices.to(torch.int32),
        out=out,
        return_lse=False
    )

    assert attn_out.shape == (num_qo_head, head_dim)

    assert torch.equal(
        ref_wrapper._paged_kv_indptr_buf,
        indptr.to(torch.int32)
    )

    assert_close(attn_out, attn_ref[:, 0])
