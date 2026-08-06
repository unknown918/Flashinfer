import torch
import math
import flashinfer
from flashinfer.testing import bench_gpu_time
from paper.core.utils import make_metadata_prefill
from paper.core.decode import BatchDecodeWrapper
import numpy as np

torch.manual_seed(42)


def ref_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float | None = None,
) -> torch.Tensor:
    num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, k_head_dim = k.shape

    assert k_head_dim == head_dim
    assert v.shape[:2] == (kv_len, num_kv_heads)
    assert num_qo_heads % num_kv_heads == 0

    group_size = num_qo_heads // num_kv_heads
    head_dim_vo = v.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    q_grouped = q.reshape(
        num_kv_heads,
        group_size,
        head_dim,
    )

    logits = torch.einsum(
        "hgd,khd->hgk",
        q_grouped.float(),
        k.float(),
    )
    logits *= sm_scale

    attn = torch.softmax(logits, dim=-1)
    output = torch.einsum(
        "hgk,khd->hgd",
        attn,
        v.float(),
    )

    return output.reshape(
        num_qo_heads,
        head_dim_vo,
    ).to(q.dtype).contiguous()


kv_layout = "HND"
device = "cuda:0"
sink_size = 4
page_size = 32
num_kv_heads = 8
num_qo_heads = 32
head_dim = 128
dtype = torch.float16

seq_lens = torch.tensor(
    [8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000],
    dtype=torch.int32,
    device=device
).contiguous()
nnz = int(seq_lens.sum())
batch_size = int(seq_lens.shape[0])
max_num_pages = batch_size * 1024

pooling = torch.zeros(max_num_pages, num_kv_heads, head_dim, dtype=dtype, device=device)
paged_k_cache = torch.zeros(max_num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
paged_v_cache = torch.zeros(max_num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)

sink_k_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
sink_v_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)

q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
append_k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
append_v = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()

num_pages_per_req = torch.empty(batch_size, dtype=torch.int32, device=device)
kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
kv_last_page_len = torch.empty(batch_size, dtype=torch.int32, device=device)
batch_indices = torch.empty(max_num_pages, dtype=torch.int32, device=device)
positions = torch.empty(max_num_pages, dtype=torch.int32, device=device)
batch_len_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)

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
    batch_indices=batch_indices,
    positions=positions,
    batch_len_indptr=batch_len_indptr,
    kv_layout=kv_layout
)

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
decode_wrapper = BatchDecodeWrapper(float_workspace_buffer=workspace_buffer)

decode_wrapper.plan(
    kv_indptr=kv_indptr,
    page_size=page_size,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    q_data_type=dtype,
    kv_data_type=dtype
)

decode_wrapper.run(
    q=q,
    paged_k_cache=paged_k_cache,
    paged_v_cache=paged_v_cache,
    sink_k_cache=sink_k_cache,
    sink_v_cache=sink_v_cache,
    kv_indptr=kv_indptr,
    kv_indices=kv_indices,
    kv_last_page_len=kv_last_page_len,
    out=output,
    kv_layout=kv_layout,
)

for i in range(batch_size):
    ref_out = ref_attn(
        q[i],
        append_k[batch_len_indptr[i]:batch_len_indptr[i + 1]],
        append_v[batch_len_indptr[i]:batch_len_indptr[i + 1]],
    )
    torch.testing.assert_close(output[i], ref_out, atol=1e-3, rtol=1e-3)

batch_decode_overhead = np.median(
    bench_gpu_time(
        lambda: decode_wrapper.run(
            q=q,
            paged_k_cache=paged_k_cache,
            paged_v_cache=paged_v_cache,
            sink_k_cache=sink_k_cache,
            sink_v_cache=sink_v_cache,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_len=kv_last_page_len,
            out=output,
            kv_layout=kv_layout,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

_q = torch.randn(batch_size * num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
_k = torch.randn(int(seq_lens[0]), batch_size * num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
_v = torch.randn(int(seq_lens[0]), batch_size * num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
single_decode_overhead = np.median(
    bench_gpu_time(
        lambda: flashinfer.single_decode_with_kv_cache(
            _q,
            _k,
            _v,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        use_cuda_graph=True
    )
)

print(f"Single Decode Overhead: {single_decode_overhead}")
print(f"Batch Decode Overhead: {batch_decode_overhead}")
