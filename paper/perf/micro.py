import torch
import flashinfer
import numpy as np
from flashinfer.testing import bench_gpu_time
from paper.core.decode import SparseDecodeWrapper
from paper.core.utils import make_metadata_prefill
from flash_attn_3.flash_attn_interface import flash_attn_func

device = "cuda:0"
torch.manual_seed(42)


def micro_bench_sparse_decode_overhead_naive(
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
    if topk * page_size >= seq_len - sink_size:
        return
    if seq_len > max_length:
        return

    query_states = torch.randn(batch_size, 1, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
    key_states = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    value_states = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()

    full_attn_overhead = np.median(
        bench_gpu_time(
            lambda: flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True
        )
    )

    print("=" * 10, "Full", "=" * 10)
    print(f"Attention: {full_attn_overhead * 1e3:.3f}us")

    n_pages = (seq_len - sink_size) // page_size
    tail_start = sink_size + n_pages * page_size
    page_ids = torch.rand(batch_size, n_pages, num_kv_heads, device=device).topk(topk, dim=1).indices
    token_ids = sink_size + (
            page_ids[:, :, None, :] * page_size +
            torch.arange(page_size, device=device)[None, None, :, None]
    ).flatten(1, 2)
    idx = token_ids[..., None].expand(-1, -1, -1, head_dim)

    k_sink = key_states[:, :sink_size, :, :]
    k_blocks = key_states.gather(1, idx)
    k_tail = key_states[:, tail_start:, :, :]

    v_sink = value_states[:, :sink_size, :, :]
    v_blocks = value_states.gather(1, idx)
    v_tail = value_states[:, tail_start:, :, :]

    concat_overhead = np.median(
        bench_gpu_time(
            lambda: torch.cat(
                [
                    k_sink,
                    k_blocks,
                    k_tail
                ], dim=1
            ).contiguous(),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True
        )
    )

    k = torch.cat(
        [
            k_sink,
            k_blocks,
            k_tail
        ], dim=1
    ).contiguous()

    v = torch.cat(
        [
            v_sink,
            v_blocks,
            v_tail
        ],
        dim=1
    ).contiguous()

    k = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=2)
    v = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=2)
    attn_overhead = np.median(
        bench_gpu_time(
            lambda: flash_attn_func(
                query_states,
                k,
                v,
                causal=True
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True
        )
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("=" * 10, "Naive Sparse", "=" * 10)
    print(f"Concat: {concat_overhead * 1e3:.3f}us")
    print(f"Attention: {attn_overhead * 1e3:.3f}us")


def micro_bench_sparse_decode_overhead(
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
        return
    if seq_len > max_length:
        return

    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device).contiguous()
    seq_lens.fill_(seq_len)
    nnz = int(seq_lens.sum())
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

    sparse_attention_overhead = 0
    for i in range(2, 32):
        sparse_indptr = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indptr_{i}.pt",
            map_location=device,
            weights_only=True
        )
        sparse_indices = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/sparse_indices_{i}.pt",
            map_location=device,
            weights_only=True
        )
        kv_last_page_len = torch.load(
            f"paper/data/{seq_len}/{batch_size}/{topk * page_size}/kv_last_page_len_{i}.pt",
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

        sparse_attention_overhead += np.median(
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

    sparse_attention_overhead /= 30
    print("=" * 10, "Ours Sparse", "=" * 10)
    print(f"Estimate: {estimate_overhead * 1e3:.3f}us")
    print(f"Sparse Attn: {sparse_attention_overhead * 1e3:.3f}")


for dtype in [torch.bfloat16]:
    for page_size in [32]:
        for max_length in [32768]:
            for head_dim in [128]:
                for (num_kv_heads, num_qo_heads, sink_size) in [(8, 32, 4)]:
                    for seq_len in [32044]:
                        for batch_size in [8]:
                            for topk in [64]:
                                with torch.cuda.device(device):
                                    print(f"Batch Size {batch_size}, SeqLen: {seq_len}, Budget {topk * page_size}")
                                    micro_bench_sparse_decode_overhead_naive(
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

                                    micro_bench_sparse_decode_overhead(
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
