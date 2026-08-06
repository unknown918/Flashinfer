import torch
import pytest
import flashinfer
from paper.core.utils import make_metadata_prefill, assert_close, ceil_div

device = "cuda:0"
torch.manual_seed(42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("max_length", [32768])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_kv_heads, num_qo_heads, sink_size", [(8, 32, 4)])
@pytest.mark.parametrize("seq_len", [1000, 2000, 4000, 8000, 16000, 32000])
def test_estimate(
        dtype,
        page_size,
        max_length,
        batch_size,
        head_dim,
        num_kv_heads,
        num_qo_heads,
        sink_size,
        seq_len
):
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device).contiguous()
    seq_lens.fill_(seq_len)
    nnz = int(seq_lens.sum(dim=0))
    max_num_pages_per_batch = max_length // page_size
    max_num_pages = batch_size * max_num_pages_per_batch

    append_k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    append_v = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()

    pooling = torch.zeros(batch_size * max_num_pages_per_batch, num_kv_heads, head_dim, dtype=dtype, device=device)
    paged_k_cache = torch.zeros(max_num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
    paged_v_cache = torch.zeros(max_num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)

    sink_k_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
    sink_v_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)

    num_pages_per_req = torch.empty(batch_size, dtype=torch.int32, device=device)
    kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    kv_last_page_len = torch.empty(batch_size, dtype=torch.int32, device=device)
    batch_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
    positions = torch.empty(max_num_pages, dtype=torch.int32, device=device)
    batch_len_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)

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
        page_size=page_size
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

    for i in range(num_pages_per_req.sum()):
        assert_close(paged_k_cache[kv_indices[i]].sum(dim=1), pooling[kv_indices[i]])

    query = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    out = torch.zeros(batch_size * num_qo_heads, max_num_pages_per_batch, dtype=dtype, device=device).contiguous()

    flashinfer.estimate(
        query=query,
        pooling=pooling,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        out=out,
    )

    group_size = num_qo_heads // num_kv_heads
    ref_out = torch.einsum(
        "bkgd,bpkd->bkgp",
        query.reshape(batch_size, num_kv_heads, group_size, head_dim),
        pooling.reshape(batch_size, max_num_pages_per_batch, num_kv_heads, head_dim),
    ).reshape(batch_size * num_qo_heads, max_num_pages_per_batch)
    assert_close(ref_out, out)
    torch.cuda.empty_cache()
