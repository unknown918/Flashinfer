import pytest
import torch
import flashinfer

torch.manual_seed(42)


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (3e-2, 3e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [(8, 32)])
@pytest.mark.parametrize("seq_len", [1021, 1022, 1023, 1024, 1192, 3351, 6666, 7777, 8888, 5537])
def test_decode_estimate(dtype, seq_len, num_kv_heads, num_qo_heads, page_size):
    head_dim = 128
    max_length = 32768
    device = torch.device("cuda:0")

    assert num_qo_heads % num_kv_heads == 0
    group_size = num_qo_heads // num_kv_heads

    num_total_pages = (max_length + page_size - 1) // page_size
    num_valid_pages = (seq_len + page_size - 1) // page_size

    query = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=device)
    pooling = torch.zeros(
        num_total_pages, num_kv_heads, head_dim,
        dtype=dtype, device=device
    ).contiguous()
    pooling_ = torch.randn(num_valid_pages, num_kv_heads, head_dim, dtype=dtype, device=device).contiguous()
    pooling[:num_valid_pages] = pooling_
    out = torch.zeros(
        num_qo_heads, num_total_pages,
        dtype=dtype, device=device
    )

    flashinfer.decode.estimate(
        query=query,
        pooling=pooling,
        num_valid_pages=num_valid_pages,
        out=out,
    )

    pooling_ = pooling_.repeat_interleave(group_size, dim=1)

    ref = torch.einsum("hd,nhd->nh", query, pooling_).transpose(0, 1)

    assert out.shape == (num_qo_heads, num_total_pages)
    assert_close(out[:, :num_valid_pages], ref[:])
