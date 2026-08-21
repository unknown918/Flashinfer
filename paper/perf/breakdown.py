import torch
import pytest
import flashinfer
import numpy as np
from flashinfer.testing import bench_gpu_time
from paper.core.decode import SparseDecodeWrapper, BatchDecodeWrapper
from paper.core.utils import make_metadata_prefill, make_logits, ceil_div

device = "cuda:0"
torch.manual_seed(42)


def bench_sparse_decode_overhead(
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
    batch_decoder = BatchDecodeWrapper(float_workspace_buffer=workspace_buffer)
    num_pages = ceil_div(seq_len - sink_size, page_size)

    proxy_indptr = torch.full(
        (batch_size + 1,), num_pages,
        dtype=torch.int32,
        device=device
    )
    proxy_indptr[0] = 0
    torch.cumsum(proxy_indptr, dim=0, out=proxy_indptr)

    batch_decoder.plan(
        kv_indptr=proxy_indptr,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    out = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()

    dense_attention_overhead = np.median(
        bench_gpu_time(
            lambda: batch_decoder.run(
                q=q,
                paged_k_cache=paged_k_cache,
                paged_v_cache=paged_v_cache,
                sink_k_cache=sink_k_cache,
                sink_v_cache=sink_v_cache,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
                out=out,
                kv_layout="HND",
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True,
        )
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

    num_pages = ceil_div(seq_len - sink_size, page_size)
    sparse_output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    logits = torch.zeros(batch_size * num_qo_heads, max_num_pages, dtype=dtype, device=device).contiguous()

    estimate_overhead = np.median(
        bench_gpu_time(
            lambda: flashinfer.estimate(
                query=q,
                pooling=pooling,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                out=logits,
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True,
        )
    )

    # simulated data has problem with topk, need manually construct logits
    logits[:, :num_pages] = make_logits(batch_size * num_qo_heads, num_pages, dtype=dtype, device=device)
    logits.contiguous()

    # last page is manually selected
    sparse_indices = torch.zeros(
        batch_size * num_kv_heads * topk * group_size,
        dtype=torch.int32,
        device=device
    )
    sparse_indptr = torch.zeros(1 + batch_size * num_kv_heads, dtype=torch.int32, device=device)
    mask_logits = torch.zeros(batch_size * num_kv_heads, max_num_pages, dtype=torch.int32, device=device)

    num_pages_per_req.sub_(1)

    # expand top-k token's indices
    topk_overhead = np.median(
        bench_gpu_time(
            lambda: flashinfer.topk_bool_mask_logits(
                logits=logits,
                mask_logits=mask_logits,
                indptr=kv_indptr,
                indices=kv_indices,
                sparse_indptr=sparse_indptr,
                sparse_indices=sparse_indices,
                top_k=topk,
                group_size=group_size,
                num_pages_per_req=num_pages_per_req,
                is_bench=True
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True,
        )
    )

    sparse_indptr[1:].add_(1)
    torch.cumsum(sparse_indptr, dim=0, dtype=torch.int32, out=sparse_indptr)
    kv_last_page_len = kv_last_page_len.repeat_interleave(num_kv_heads).contiguous()

    sparse_attention_overhead = np.median(
        bench_gpu_time(
            lambda: sparse_decoder.run(
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
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True,
        )
    )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Batch Size {batch_size}, Budget {topk * page_size}, SeqLen: {seq_len}")
    print(f"Full Attn: {dense_attention_overhead * 1e3:.3f}us")
    print("-" * 20)
    print(f"Estimate: {estimate_overhead * 1e3:.3f}us")
    print(f"Topk: {topk_overhead * 1e3 - 4:.3f}us")
    print(f"Sparse Attn: {sparse_attention_overhead * 1e3:.3f}")
    print("-" * 20)
    speed_up = dense_attention_overhead / (estimate_overhead + topk_overhead + sparse_attention_overhead)
    print(f"Speed Up: {speed_up:.1f}")
    print("=" * 20)


for dtype in [torch.bfloat16]:
    for page_size in [32]:
        for max_length in [32768]:
            for head_dim in [128]:
                for (num_kv_heads, num_qo_heads, sink_size) in [(8, 32, 4)]:
                    for seq_len in [4000, 8000, 16000, 32000]:
                        for batch_size in [1, 2, 4, 8]:
                            for topk in [32, 64]:
                                with torch.cuda.device(device):
                                    bench_sparse_decode_overhead(
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
                                    )
