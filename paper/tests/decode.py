import torch
import math
import flashinfer
from flashinfer.decode import get_batch_decode_module
from paper.core.utils import make_metadata_prefill
from flashinfer.utils import (
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
)

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
pos_encoding_mode = "NONE"
logits_soft_cap = None
sm_scale = None
rope_scale = None
rope_theta = None

device = "cuda:3"
sink_size = 4
page_size = 32
num_kv_heads = 8
num_qo_heads = 32
head_dim = 128
dtype = torch.float16

float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
int_workspace_buffer = torch.empty((8 * 1024 * 1024,), dtype=torch.uint8, device=device)
pin_memory_int_workspace_buffer = torch.empty(
    int_workspace_buffer.shape,
    dtype=torch.uint8,
    pin_memory=True,
    device="cpu",
)
workspace_size = float_workspace_buffer.numel() * float_workspace_buffer.element_size()

seq_lens = torch.tensor(
    [8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000],
    dtype=torch.int32,
    device=device
).contiguous()
batch_size = seq_lens.shape[0]
nnz = int(seq_lens.sum(dim=0))
max_num_pages = 1024 * batch_size

pooling = torch.zeros(max_num_pages, num_kv_heads, head_dim, dtype=dtype, device=device)
paged_k_cache = torch.zeros(max_num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
paged_v_cache = torch.zeros(max_num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)

sink_k_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)
sink_v_cache = torch.zeros(batch_size, num_kv_heads, sink_size, head_dim, dtype=dtype, device=device)

q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
out = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
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
    kv_layout=kv_layout,
)

kv_indptr_host = kv_indptr.to("cpu", non_blocking=True)  # for split-kv binary search

get_module_args = (
    dtype,
    dtype,
    dtype,
    kv_indptr_host.dtype,
    head_dim,  # head_dim_qk
    head_dim,  # head_dim_vo
    PosEncodingMode[pos_encoding_mode].value,
    False,  # use_sliding_window
    False,  # use_logits_soft_cap
)

_cached_module = get_batch_decode_module(*get_module_args)

args = [
    float_workspace_buffer,  # float_workspace_buffer
    int_workspace_buffer,  # int_workspace_buffer
    pin_memory_int_workspace_buffer,  # page_locked_int_workspace_buffer
    kv_indptr_host,  # kv_indptr
    batch_size,  # batch_size
    num_qo_heads,  # num_qo_heads
    num_kv_heads,  # num_kv_heads,
    page_size,  # page_size
    True,  # enable_cuda_graph
    -1,  # window_left
    0.0,  # logits_soft_cap
    head_dim,  # head_dim_qk
    head_dim,  # head_dim_vo
    torch.empty(0, dtype=dtype),
    torch.empty(0, dtype=dtype),
]

_plan_info = _cached_module.plan(*args, )

_check_pos_encoding_mode(pos_encoding_mode)
if logits_soft_cap is None:
    logits_soft_cap = 0.0
if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(q.size(-1))
if rope_scale is None:
    rope_scale = 1.0
if rope_theta is None:
    rope_theta = 1e4

_cached_module.run(
    float_workspace_buffer,
    int_workspace_buffer,
    _plan_info,
    q,
    paged_k_cache,
    paged_v_cache,
    kv_indptr,
    kv_indices,
    kv_last_page_len,
    sink_k_cache,
    sink_v_cache,
    sink_indices,
    out,
    None,  # lse
    TensorLayout[kv_layout].value,
    -1,  # window_left
    False,  # enable pdl
    None,  # alibi_slopes_buf
    logits_soft_cap,
    sm_scale,
    rope_scale,
    rope_theta,
)

for i in range(batch_size):
    ref_out = ref_attn(
        q[i],
        append_k[batch_len_indptr[i]:batch_len_indptr[i + 1]],
        append_v[batch_len_indptr[i]:batch_len_indptr[i + 1]]
    )
    torch.testing.assert_close(out[i], ref_out, atol=1e-3, rtol=1e-3)

print("Hello")
