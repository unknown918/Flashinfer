import torch
import pytest
import flashinfer

from ..core.utils import assert_close, make_metadata_prefill
from ..core.prefill import BatchPrefillWrapper

device = "cuda:0"
torch.manual_seed(42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("max_length", [32768])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_kv_heads, num_qo_heads, sink_size", [(8, 32, 4)])
@pytest.mark.parametrize("seq_len", [1028, 2000, 4000, 8000, 16000, 32000])
def test_prefill(
        dtype,
        page_size,
        max_length,
        batch_size,
        head_dim,
        num_kv_heads,
        num_qo_heads,
        sink_size,
        seq_len,
):
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    nnz = int(seq_lens.sum())
    total_pages = batch_size * (max_length // page_size)

    append_k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    append_v = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    pooling = torch.zeros(total_pages, num_kv_heads, head_dim, dtype=dtype, device=device)
    paged_k_cache = torch.zeros(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
    paged_v_cache = torch.zeros_like(paged_k_cache)
    sink_k_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
    sink_v_cache = torch.zeros_like(sink_k_cache)

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
        page_size=page_size,
    )

    sink_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
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
        kv_layout="HND",
    )

    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()

    prefill_wrapper = BatchPrefillWrapper(
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        sink_size=sink_size,
        page_size=page_size,
        kv_layout="HND",
        dtype=dtype,
        device=device,
    )

    prefill_wrapper.plan(
        qo_indices=seq_lens,
        kv_indices=kv_indices[:int(kv_indptr[-1])],
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        sink_indices=sink_indices,
    )
    out = prefill_wrapper.run(q, sink_k_cache, sink_v_cache, paged_k_cache, paged_v_cache)

    for i in range(batch_size):
        start = int(batch_len_indptr[i])
        end = int(batch_len_indptr[i + 1])
        ref_out = flashinfer.single_prefill_with_kv_cache(
            q=q[start:end],
            k=append_k[start:end],
            v=append_v[start:end],
            causal=True,
            kv_layout="NHD",
            backend="fa2",
        )
        assert_close(out[start:end], ref_out)

    torch.cuda.empty_cache()
