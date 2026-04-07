import torch
import flashinfer
from flashinfer.testing import bench_gpu_time
import numpy as np

torch.manual_seed(42)

page_size = 32
head_dim = 128
num_kv_heads = 8
num_qo_heads = 32

seq_len = 4 + 32768  # for sink test
top_k = 16

# same layout as in estimate.py
q = torch.randn(num_qo_heads, head_dim).half().to(0)
k = torch.randn(num_kv_heads, seq_len, head_dim).half().to(0)
v = torch.randn(num_kv_heads, seq_len, head_dim).half().to(0)

logits = torch.randn(num_qo_heads, 1024, device="cuda")

_, grouped_indices = torch.topk(logits, k=top_k, dim=-1)
topk_mask = torch.zeros_like(logits, dtype=torch.bool, device="cuda")
source_values = torch.ones_like(grouped_indices, dtype=torch.bool, device="cuda")
block_mask_map = topk_mask.scatter_(dim=-1, index=grouped_indices, src=source_values)
block_mask_map = block_mask_map.reshape(-1, 4, block_mask_map.shape[1]).any(1)
sink_col = torch.ones((num_kv_heads, 1), dtype=torch.bool, device="cuda")
block_mask_map = torch.cat([sink_col, block_mask_map], dim=1).unsqueeze(1)

# handle irregular blocks
# with record_function("Create Mask"):
remain_len = seq_len - 4
num_blocks = remain_len // page_size
last_block = remain_len % page_size
row = [4] + [page_size] * num_blocks
if last_block > 0:
    row.append(last_block)
block_col_sz = torch.tensor(row, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(num_kv_heads, 1)
block_row_sz = torch.tensor([[1]], dtype=torch.int32, device="cuda").repeat(num_kv_heads, 1)

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

wrapper = flashinfer.SparseSinkAttentionWrapper(workspace_buffer)

_indptr, _indices = flashinfer.topk_bool_mask_logits(logits, top_k, max_length=1024, group_size=4, block_size=32)

plan_overhead = np.median(
    bench_gpu_time(
        lambda: wrapper.plan(
            _indptr,
            _indices,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

copy_overhead = np.median(
    bench_gpu_time(
        lambda: _indptr.to("cpu", non_blocking=True),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
)

wrapper.plan(
    _indptr,
    _indices,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal=True
)

print(f"Decode Plan Overhead: {plan_overhead:.3f} ms")
print(f"Indptr Copy Overhead: {copy_overhead:.3f} ms")
