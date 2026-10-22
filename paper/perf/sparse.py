import torch
import flashinfer
import numpy as np
from pathlib import Path
from flashinfer.testing import bench_gpu_time
from ..core.decode import SparseDecodeWrapper
from ..core.utils import make_metadata_prefill, assert_close

device = "cuda:0"
torch.manual_seed(42)


def bench_sparse_decode(
        dtype,
        num_layers,
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
    workspace_buffer_ = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
    sparse_decoder = SparseDecodeWrapper(float_workspace_buffer=workspace_buffer)
    sparse_decoder_ = SparseDecodeWrapper(float_workspace_buffer=workspace_buffer_)

    sparse_proxy_indptr = torch.full(
        (batch_size * num_kv_heads + 1,), topk * page_size,
        dtype=torch.int32,
        device=device
    )
    sparse_proxy_indptr[0] = 0
    torch.cumsum(sparse_proxy_indptr, dim=0, out=sparse_proxy_indptr)

    # sparse_proxy_indptr = torch.load(
    #     f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indptr_2.pt",
    #     map_location=device,
    #     weights_only=True
    # )

    sparse_decoder_.plan(
        kv_indptr=sparse_proxy_indptr,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=dtype,
        kv_data_type=dtype
    )

    optimal_times, proxy_times = [], []

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    sparse_output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    sparse_output_ = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()

    for layer_id in range(2, num_layers):
        sparse_indptr = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indptr_{layer_id}.pt",
            map_location=device,
            weights_only=True
        )
        sparse_indices = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indices_{layer_id}.pt",
            map_location=device,
            weights_only=True
        )
        kv_last_page_len = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/kv_last_page_len_{layer_id}.pt",
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

        decode_overhead = np.median(
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

        decode_overhead_ = np.median(
            bench_gpu_time(
                lambda: sparse_decoder_.run(
                    q=q,
                    paged_k_cache=paged_k_cache,
                    paged_v_cache=paged_v_cache,
                    sink_k_cache=sink_k_cache,
                    sink_v_cache=sink_v_cache,
                    kv_indptr=sparse_indptr,
                    kv_indices=sparse_indices,
                    kv_last_page_len=kv_last_page_len,
                    out=sparse_output_,
                    kv_layout="HND",
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
                use_cuda_graph=True
            )
        )
        
        # assert_close(sparse_output_, sparse_output)
        
        sparse_output_.zero_()
        sparse_output_.zero_()

        torch.cuda.empty_cache()

        optimal_times.append(decode_overhead * 1e3)
        proxy_times.append(decode_overhead_ * 1e3)

        print("=" * 10, f"Seq Len {seq_len}, Batch Size {batch_size}, Budget {topk * page_size}, Layer {layer_id}", "=" * 10)
        print(f"Sparse Decode Overhead w/ schedule: {decode_overhead * 1e3:.3f} us")
        print(f"Sparse Decode Overhead w/o schedule: {decode_overhead_ * 1e3:.3f} us")
        output_dir = Path(f"paper/data/schedule/{str(batch_size)}/{str(topk * page_size)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            torch.tensor(optimal_times, dtype=torch.float32),
            output_dir / "optimal.pt",
        )
        torch.save(
            torch.tensor(proxy_times, dtype=torch.float32),
            output_dir / "proxy.pt",
        )


for dtype in [torch.bfloat16]:
    for page_size in [32]:
        for max_length in [32768]:
            for head_dim in [128]:
                for (num_kv_heads, num_qo_heads, sink_size) in [(8, 32, 4)]:
                    for batch_size in [8]:
                        for topk in [32, 64]:
                            for num_layers in [32]:
                                for seq_len in [32044]:
                                    with torch.cuda.device(device):
                                        bench_sparse_decode(
                                            dtype,
                                            num_layers,
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
