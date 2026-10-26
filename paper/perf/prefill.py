import torch
import numpy as np
import flashinfer
from flashinfer.testing import bench_gpu_time
from paper.core.utils import make_metadata_prefill
from paper.core.prefill import BatchPrefillWrapper

device = "cuda:0"
torch.manual_seed(42)


def bench_batch_prefill(
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: str = "cuda:0",
        sink_size: int = 4,
        page_size: int = 32,
        num_kv_heads: int = 8,
        num_qo_heads: int = 32,
        head_dim: int = 128,
        max_length: int = 32768,
):
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    nnz = batch_size * seq_len
    total_pages = batch_size * (max_length // page_size)

    q = torch.randn(nnz, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
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
        kv_layout="HND",
    )

    sink_wrapper = BatchPrefillWrapper(
        head_dim=head_dim,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        sink_size=sink_size,
        page_size=page_size,
        kv_layout="HND",
        dtype=dtype,
        device=device,
    )

    sink_wrapper.plan(
        qo_indices=seq_lens,
        kv_indices=kv_indices[:int(kv_indptr[-1])],
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        sink_indices=sink_indices,
    )

    ragged_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    batch_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ragged_workspace, kv_layout="NHD")
    batch_wrapper.plan(
        qo_indptr=batch_len_indptr,
        kv_indptr=batch_len_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    ragged_out = torch.empty_like(q)

    composed_ms = np.median(
        bench_gpu_time(
            lambda: sink_wrapper.run(
                q,
                sink_k_cache,
                sink_v_cache,
                paged_k_cache,
                paged_v_cache
            ),
            dry_run_time_ms=200,
            repeat_time_ms=2000,
            use_cuda_graph=True,
        )
    )

    ragged_ms = np.median(
        bench_gpu_time(
            lambda: batch_wrapper.run(
                q,
                append_k,
                append_v,
                out=ragged_out
            ),
            dry_run_time_ms=200,
            repeat_time_ms=2000,
            use_cuda_graph=True,
        )
    )

    print(f"Batch Size: {batch_size}, SeqLen: {seq_len}")
    print(f"Batch Prefill: {ragged_ms:.3f} ms")
    print(f"Composed Prefill: {composed_ms:.3f} ms")
    print(f"Speed Up: {ragged_ms / composed_ms:.2f}x")
    print("=" * 20)


if __name__ == "__main__":
    for dtype in [torch.bfloat16]:
        for page_size in [32]:
            for max_length in [32768]:
                for head_dim in [128]:
                    for num_kv_heads, num_qo_heads, sink_size in [(8, 32, 4)]:
                        for seq_len in [32000]:
                            for batch_size in [8]:
                                bench_batch_prefill(
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    dtype=dtype,
                                    device=device,
                                    sink_size=sink_size,
                                    page_size=page_size,
                                    num_kv_heads=num_kv_heads,
                                    num_qo_heads=num_qo_heads,
                                    head_dim=head_dim,
                                    max_length=max_length,
                                )
                                torch.cuda.empty_cache()
