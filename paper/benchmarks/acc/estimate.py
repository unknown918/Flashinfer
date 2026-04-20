import torch
import flashinfer

torch.manual_seed(42)

seq_len = 1024
head_dim = 128
page_size = 32
num_kv_heads = 8
num_qo_heads = 32
max_length = 131072

num_total_pages = (max_length + page_size - 1) // page_size

q = torch.randn(num_qo_heads, head_dim).half().to(0)
pooling = torch.randn(num_total_pages, num_kv_heads, head_dim).half().to(0)
out = torch.zeros(num_qo_heads, num_total_pages).half().to(0)

out = flashinfer.decode.estimate(
    seq_len,
    q,
    pooling,
    out
)

pooling = pooling.repeat_interleave(4, dim=1)
ref = torch.einsum("hd,nhd->nh", q, pooling[:seq_len])
torch.testing.assert_close(out[:, :seq_len].transpose(0, 1), ref)
