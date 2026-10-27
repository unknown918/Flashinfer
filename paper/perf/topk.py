import torch
import numpy as np
import flashinfer
from flashinfer.testing import bench_gpu_time
from ..core.utils import make_metadata_prefill, ceil_div, make_logits

device = "cuda:0"
torch.manual_seed(42)


def bench_topk(
        dtype,
        topk,
        page_size,
        max_length,
        batch_size,
        num_kv_heads,
        num_qo_heads,
        sink_size,
        seq_len,
):
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device).contiguous()
    seq_lens.fill_(seq_len)

    group_size = num_qo_heads // num_kv_heads
    max_num_pages = int(max_length / page_size)
    total_pages = max_num_pages * batch_size

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
        page_size=page_size
    )

    logits = torch.zeros(
        batch_size * num_qo_heads, max_num_pages,
        dtype=dtype,
        device=device
    )

    ref_indices = torch.tensor([], dtype=torch.int32, device=device)
    ref_indptr = torch.zeros(batch_size * num_kv_heads + 1, dtype=torch.int32, device=device)

    def ref_expand(_logits, batch_id, ref_indptr, ref_indices):
        page_topk = torch.topk(_logits[:, :-1], k=topk).indices
        _page_topk = page_topk.repeat_interleave(batch_size, dim=0)
        torch_union_overhead = np.median(
            bench_gpu_time(
                lambda: torch.unique(_page_topk),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
                # use_cuda_graph=True
            )
        )

        page_topk = page_topk.reshape(num_kv_heads, group_size, -1)
        _indices = kv_indices[kv_indptr[batch_id]: kv_indptr[batch_id + 1]]

        for kv_head_idx, group_topk in enumerate(page_topk):
            group_topk = group_topk.flatten()
            group_topk = torch.unique(group_topk)
            group_indices = torch.cat(
                [
                    _indices[group_topk],
                    _indices[-1:]
                ],
                dim=-1,
            )
            group_indices = group_indices * num_kv_heads + kv_head_idx
            ref_indices = torch.cat(
                [ref_indices, group_indices],
                dim=-1
            )
            ref_indptr[batch_id * num_kv_heads + kv_head_idx + 1] = group_indices.size(0)

        return ref_indptr, ref_indices, torch_union_overhead

    torch_union_overhead = 0.0
    for batch_idx, seq_len in enumerate(seq_lens):
        seq_len = int(seq_len) - sink_size
        logits_ = make_logits(
            num_qo_heads, ceil_div(seq_len, page_size),
            device=device,
            dtype=dtype,
        )
        ref_indptr, ref_indices, torch_union_overhead = ref_expand(logits_, batch_idx, ref_indptr, ref_indices)
        logits[batch_idx * num_qo_heads:(batch_idx + 1) * num_qo_heads, :ceil_div(seq_len, page_size)] = logits_

    logits.contiguous()

    mask_logits = torch.zeros(
        batch_size * num_kv_heads, max_num_pages,
        dtype=torch.int32,
        device=device
    )

    sparse_indptr = torch.zeros(
        1 + batch_size * num_kv_heads,
        dtype=torch.int32,
        device=device
    )

    sparse_indices = torch.zeros(
        batch_size * num_kv_heads * topk * group_size,
        dtype=torch.int32,
        device=device
    )

    num_pages_per_req.add_(-1)
    num_pages = ceil_div(seq_len, page_size)

    _logits = logits[:, :num_pages].contiguous()
    logits = logits.contiguous()

    topk_overhead = np.median(
        bench_gpu_time(
            lambda: flashinfer.top_k(
                input=_logits,
                k=topk,
                sorted=False,
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
            use_cuda_graph=True
        )
    )

    torch.cuda.empty_cache()

    fused_topk_overhead = np.median(
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
            use_cuda_graph=True
        )
    )

    print("=" * 10)
    print(f"Batch Size: {batch_size}, TopK: {topk}, Seq Len: {seq_len + 4}")
    print(f"Torch Union Overhead: {torch_union_overhead * 1e3:.2f}us")
    print(f"TopK Overhead: {topk_overhead * 1e3:.2f}us")
    print(f"Fused TopK Overhead: {fused_topk_overhead * 1e3:.2f}us")

    torch.cuda.empty_cache()


for dtype in [torch.bfloat16]:
    for page_size in [32]:
        for max_length in [32768]:
            for (num_kv_heads, num_qo_heads, sink_size) in [(8, 32, 4)]:
                for batch_size in [1, 2, 4, 8]:
                    for topk in [8, 16, 32, 64]:
                        for seq_len in [4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]:
                            with torch.cuda.device(device):
                                bench_topk(
                                    dtype,
                                    topk,
                                    page_size,
                                    max_length,
                                    batch_size,
                                    num_kv_heads,
                                    num_qo_heads,
                                    sink_size,
                                    seq_len
                                )
