import torch
import flashinfer
import numpy as np
from pathlib import Path
from flashinfer.testing import bench_gpu_time
from ..core.decode import SparseDecodeWrapper
from ..core.utils import make_metadata_prefill, assert_close

device = "cuda:0"
torch.manual_seed(42)


def bench_union_decode(
        dtype,
        page_size,
        topk,
        max_length,
        batch_size,
        head_dim,
        num_kv_heads,
        num_qo_heads,
        sink_size,
        seq_len,
):
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

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    sparse_output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()

    sparse_indptr = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indptr_30.pt",
            map_location=device,
            weights_only=True
    )
    sparse_indices = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indices_30.pt",
            map_location=device,
            weights_only=True
    )
    # sparse_indptr = torch.full(
    #     (batch_size * num_kv_heads + 1,), topk,
    #     dtype=torch.int32,
    #     device=device
    # )
    # sparse_indptr[0] = 0
    # torch.cumsum(sparse_indptr, dim=0, out=sparse_indptr)
    
    # sparse_indices = torch.zeros(
    #     topk,
    #     dtype=torch.int32,
    #     device=device
    # )
    # sparse_indices = sparse_indices.unsqueeze(0).repeat(batch_size * num_kv_heads, 1)
    
    # for i in range (batch_size):
    #     for j in range (num_kv_heads):
    #         sparse_indices[i * num_kv_heads + j][0] = i * max_num_pages + j
    #         for k in range (1, topk):
    #             sparse_indices[i * num_kv_heads + j][k] = sparse_indices[i * batch_size + j][0] + k * num_kv_heads
            
    kv_last_page_len = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/kv_last_page_len_2.pt",
            map_location=device,
            weights_only=True
    )

    sparse_decoder.plan(
            kv_indptr=sparse_indptr,
            page_size=page_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=dtype,
            kv_data_type=dtype
    )

    union_overhead = np.median(
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
            use_cuda_graph=True
        )
    )
        
    torch.cuda.empty_cache()

    print("=" * 10, f"Batch Size {batch_size}, Seq Len {seq_len}, Budget {topk * page_size}", "=" * 10)
    print(f"Naive Overhead: {union_overhead * 1e3:.3f} us")


for dtype in [torch.bfloat16]:
    for page_size in [32]:
        for max_length in [32768]:
            for head_dim in [128]:
                for (num_kv_heads, num_qo_heads, sink_size) in [(8, 32, 4)]:
                    for batch_size in [1, 2, 4, 8]:
                        for topk in [8, 16]:
                            for seq_len in [8045]:
                                with torch.cuda.device(device):
                                    bench_union_decode(
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
