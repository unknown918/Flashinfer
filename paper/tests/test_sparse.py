import pytest
import torch
import flashinfer


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-4, 1e-4),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16]) # fixme: torch.bfloat16 has precision bug
@pytest.mark.parametrize("topk", [8, 15, 16])
@pytest.mark.parametrize("seq_len", [1021, 1022, 1023, 1024, 1192, 3351, 6666, 7777, 8888, 5537])
def test_sparse_sink_attention(dtype, topk, seq_len):
    torch.manual_seed(42)

    page_size = 16
    max_length = 32768
    num_kv_head = 8
    num_qo_head = 32
    head_dim = 128
    sink_size = 4

    group_size = num_qo_head // num_kv_head
    num_total_pages = max_length // page_size

    assert num_qo_head % num_kv_head == 0, "invalid head grouping"

    num_valid_pages = (seq_len - sink_size + page_size - 1) // page_size
    last_page_len = seq_len - sink_size - (num_valid_pages - 1) * page_size

    logits = torch.randn(
        num_qo_head, num_total_pages,
        dtype=torch.float32, device="cuda"
    )

    num_blocks_col = num_valid_pages + 1

    block_row_sz = torch.full(
        (num_kv_head, 1), 1,
        dtype=torch.int32, device="cuda"
    )

    block_col_sz = torch.zeros(
        num_blocks_col, dtype=torch.int32, device="cuda"
    )
    block_col_sz[0] = sink_size
    block_col_sz[-1] = last_page_len

    for i in range(1, num_blocks_col):
        block_col_sz[i] = min(page_size, seq_len - sink_size - (i - 1) * page_size)

    block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_head, 1)

    block_mask_map = torch.zeros(
        num_kv_head, 1, num_blocks_col,
        dtype=torch.int32, device="cuda"
    )

    indices = torch.topk(
        logits[:, :num_valid_pages - 1],
        k=topk - 1
    ).indices.cpu()

    indices += 1 # include sink tokens

    for i in range(num_kv_head):
        for j in range(group_size):
            block_mask_map[i, 0, indices[i * group_size + j]] = 1

        block_mask_map[i, 0, 0] = 1
        block_mask_map[i, 0, -1] = 1

    q = torch.randn(num_qo_head, 1, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(num_kv_head, seq_len, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(num_kv_head, seq_len, head_dim, dtype=dtype, device="cuda")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
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
        device="cuda"
    )

    indices_buf = torch.zeros(
        num_kv_head * (sink_size + topk * group_size * page_size),
        dtype=torch.uint32,
        device="cuda"
    )

    mask_logits = torch.zeros(
        num_kv_head, num_total_pages,
        dtype=torch.uint32,
        device="cuda"
    )

    _indptr, _indices = flashinfer.topk_bool_mask_logits(
        topk=topk - 1,
        page_size=page_size,
        last_page_len=last_page_len,
        num_valid_pages=num_valid_pages - 1,
        logits=logits,
        indptr=indptr,
        indices=indices_buf,
        group_size=group_size,
        mask_logits=mask_logits
    )

    _indptr = _indptr.cumsum(dim=-1)

    decode_workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    decode_wrapper = flashinfer.SparseSinkAttentionWrapper(
        decode_workspace
    )

    decode_wrapper.plan(
        kv_indptr=_indptr.to(torch.int32),
        kv_indices=_indices.to(torch.int32),
        num_qo_heads=num_qo_head,
        num_kv_heads=num_kv_head,
        head_dim=head_dim,
        causal=True
    )

    q_decode = q.transpose(0, 1)[0].contiguous()
    k_decode = k.transpose(0, 1).unsqueeze(1).contiguous()
    v_decode = v.transpose(0, 1).unsqueeze(1).contiguous()

    out = torch.zeros(
        num_qo_head, head_dim,
        dtype=dtype, device="cuda"
    )

    attn_out = decode_wrapper.run(
        q=q_decode,
        k=k_decode,
        v=v_decode,
        out=out,
        return_lse=False
    )

    assert attn_out.shape == (num_qo_head, head_dim)

    torch.testing.assert_close(
        ref_wrapper._paged_kv_indptr_buf,
        _indptr.to(torch.int32)
    )

    assert_close(attn_out, attn_ref[:, 0])
