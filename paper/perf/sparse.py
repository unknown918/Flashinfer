import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

torch.manual_seed(42)


def bench_sink_sparse_attention(
        top_k,
        seq_len,
        page_size,
        head_dim,
        num_qo_heads,
        num_kv_heads,
):
    sink = 4
    max_length = 32768
    dtype = torch.half

    group_size = num_qo_heads // num_kv_heads
    num_total_pages = max_length // page_size

    assert num_qo_heads % num_kv_heads == 0, "invalid head grouping"

    num_valid_pages = (seq_len - sink + page_size - 1) // page_size
    last_page_len = seq_len - sink - (num_valid_pages - 1) * page_size

    logits = torch.zeros(
        num_qo_heads, num_total_pages,
        dtype=torch.float32, device="cuda"
    )

    logits_ = torch.randn(
        num_qo_heads, num_valid_pages,
        dtype=torch.float32, device="cuda"
    )

    logits[:, :num_valid_pages] = logits_

    indptr_buf = torch.zeros(
        1 + num_kv_heads,
        dtype=torch.uint32,
        device="cuda"
    )

    indices_buf = torch.zeros(
        num_kv_heads * (sink + top_k * group_size * page_size),
        dtype=torch.uint32,
        device="cuda"
    )

    mask_logits = torch.zeros(
        num_kv_heads, num_total_pages,
        dtype=torch.uint32,
        device="cuda"
    )

    q = torch.randn(num_qo_heads, head_dim, dtype=dtype, device="cuda").contiguous()
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda").contiguous()
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda").contiguous()

    flashinfer.topk_bool_mask_logits(
        top_k=top_k - 1,
        page_size=page_size,
        last_page_len=last_page_len,
        num_valid_pages=num_valid_pages - 1,
        logits=logits,
        indptr=indptr_buf,
        indices=indices_buf,
        group_size=group_size,
        mask_logits=mask_logits
    )

    indptr_buf = indptr_buf.cumsum(dim=-1)

    decode_workspace = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )

    decode_wrapper = flashinfer.SparseSinkAttentionWrapper(
        num_kv_heads=num_kv_heads,
        device=torch.device("cuda"),
        float_workspace_buffer=decode_workspace,
    )

    decode_wrapper.plan(
        kv_indptr=indptr_buf.to(torch.int32),
        kv_indices=indices_buf.to(torch.int32),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True
    )

    out = torch.zeros(
        num_qo_heads, head_dim,
        dtype=dtype, device="cuda"
    )

    sparse_ms = np.median(
        bench_gpu_time(
            lambda: decode_wrapper.run(
                q=q,
                k=k.unsqueeze(1),
                v=v.unsqueeze(1),
                out=out,
                return_lse=False
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True
        )
    )

    dense_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.decode.single_decode_with_kv_cache(
                q=q,
                k=k,
                v=v,
                kv_layout='NHD',
                return_lse=False
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True
        )
    )

    sparsity = indptr_buf[-1] / (num_kv_heads * seq_len)

    print(
        f"seq_len={seq_len}, top_k={top_k}, page_size={page_size}, "
        f"sparsity={sparsity:.3f}, "
        f"sparse fa2 time: {sparse_ms * 1e3:.3f} us, "
        f"dense fa2 time: {dense_ms * 1e3:.3f} us"
    )


if __name__ == "__main__":
    for num_qo_heads in [32]:
        for num_kv_heads in [8]:
            for head_dim in [128]:
                for seq_len in [32760]:
                    for page_size in [32]:
                        for top_k in [64]:
                            bench_sink_sparse_attention(
                                top_k,
                                seq_len,
                                page_size,
                                head_dim,
                                num_qo_heads,
                                num_kv_heads,
                            )
