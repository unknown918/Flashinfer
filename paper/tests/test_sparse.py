import torch
import pytest
import flashinfer
from paper.core.decode import SparseDecodeWrapper, BatchDecodeWrapper
from paper.core.utils import assert_close, make_metadata_prefill, make_logits, ceil_div

device = "cuda:0"
torch.manual_seed(42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("topk", [8, 15, 16, 32, 64])
@pytest.mark.parametrize("max_length", [32768])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_kv_heads, num_qo_heads, sink_size", [(8, 32, 4)])
@pytest.mark.parametrize("seq_len", [1028, 2000, 4000, 8000, 16000, 32000])
def test_sparse_decode(
        dtype,
        page_size,
        topk,
        max_length,
        batch_size,
        head_dim,
        num_kv_heads,
        num_qo_heads,
        sink_size,
        seq_len
):
    if topk * page_size >= seq_len - sink_size:
        pytest.skip(f"invalid case: topk * page_size = {topk * page_size} >= {seq_len - sink_size}")
    if seq_len > max_length:
        pytest.skip(f"invalid case: seq_len {seq_len} >= max_length {max_length}")

    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device).contiguous()
    seq_lens.fill_(seq_len)
    nnz = int(seq_lens.sum())
    group_size = num_qo_heads // num_kv_heads
    max_num_pages = max_length // page_size
    total_pages = batch_size * max_num_pages

    append_k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    append_v = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    pooling = torch.zeros(total_pages, num_kv_heads, head_dim, dtype=dtype, device=device)

    # HND: (total_pages, num_kv_head, page_size, head_dim)
    # NHD: (total_pages, page_size, num_kv_head, head_dim)
    paged_k_cache = torch.zeros(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
    paged_v_cache = torch.zeros(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)

    sink_k_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
    sink_v_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)

    num_pages_per_req = torch.zeros(batch_size, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(total_pages, dtype=torch.int32, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    batch_indices = torch.zeros(total_pages, dtype=torch.int32, device=device)
    positions = torch.zeros(total_pages, dtype=torch.int32, device=device)
    batch_len_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    sink_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

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
        sink_indices=sink_indices,
        batch_indices=batch_indices,
        positions=positions,
        batch_len_indptr=batch_len_indptr,
        kv_layout="HND"
    )

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
    sparse_decoder = SparseDecodeWrapper(float_workspace_buffer=workspace_buffer)

    sparse_proxy_indptr = torch.full(
        (batch_size * num_kv_heads + 1,), topk,
        dtype=torch.int32,
        device=device
    )
    sparse_proxy_indptr[0] = 0
    torch.cumsum(sparse_proxy_indptr, dim=0, out=sparse_proxy_indptr)

    sparse_decoder.plan(
        kv_indptr=sparse_proxy_indptr,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=dtype,
        kv_data_type=dtype
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    sparse_output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()

    num_pages = int(num_pages_per_req[0])
    logits = torch.zeros(batch_size * num_qo_heads, max_num_pages, dtype=dtype, device=device).contiguous()
    logits_ = make_logits(batch_size * num_qo_heads, num_pages, dtype=dtype, device=device).contiguous()
    logits[:, :num_pages].copy_(logits_)

    # last page is manually selected
    sparse_indices = torch.zeros(
        batch_size * num_kv_heads * topk * group_size,
        dtype=torch.int32,
        device=device
    )
    sparse_indptr = torch.zeros(1 + batch_size * num_kv_heads, dtype=torch.int32, device=device)
    mask_logits = torch.zeros(batch_size * num_kv_heads, max_num_pages, dtype=torch.int32, device=device)

    num_pages_per_req.add_(-1)

    # expand top-k token's indices
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
    torch.cumsum(sparse_indptr, dim=0, dtype=torch.int32, out=sparse_indptr)
    kv_last_page_len = kv_last_page_len.repeat_interleave(num_kv_heads).contiguous()

    sparse_decoder.run(
        q=q,
        paged_k_cache=paged_k_cache,
        paged_v_cache=paged_v_cache,
        sink_k_cache=sink_k_cache,
        sink_v_cache=sink_v_cache,
        kv_indptr=sparse_indptr,
        kv_indices=sparse_indices,
        kv_last_page_len=kv_last_page_len,
        out=sparse_output,
        kv_layout="HND",
    )

    block_row_sz = torch.ones(num_kv_heads, 1, dtype=torch.int32, device=device)
    block_col_sz = torch.zeros(num_kv_heads, 1 + num_pages, dtype=torch.int32, device=device)
    block_col_sz[:, 0] = sink_size
    block_col_sz[:, 1:-1] = page_size
    block_col_sz[:, -1] = seq_len - sink_size - page_size * (num_pages - 1)
    for i in range(batch_size):
        workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.VariableBlockSparseAttentionWrapper(workspace_buffer)
        block_mask_map = torch.zeros(num_kv_heads, 1, 1 + num_pages, dtype=torch.bool, device=device)
        block_mask_map[:, :, 0] = True
        block_mask_map[:, :, -1] = True
        page_topk = torch.topk(logits_[i * num_qo_heads:(i + 1) * num_qo_heads, :-1], k=topk).indices
        page_topk = page_topk.reshape(num_kv_heads, group_size, -1)

        for kv_head_idx, group_topk in enumerate(page_topk):
            group_topk = group_topk.flatten()
            group_topk = torch.unique(group_topk)
            block_mask_map[kv_head_idx, :, 1 + group_topk] = True

        wrapper.plan(
            block_mask_map,
            block_row_sz,
            block_col_sz,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        q_ = q[i].unsqueeze(dim=1)
        k = append_k[i * seq_len:(i + 1) * seq_len].transpose(0, 1)
        v = append_v[i * seq_len:(i + 1) * seq_len].transpose(0, 1)
        o = wrapper.run(q_, k, v)
        assert_close(o[:, 0, :], sparse_output[i])

    torch.cuda.empty_cache()
