import torch
import triton
import triton.language as tl


@triton.jit
def _build_metadata_prefill_kernel(
        seq_lens,
        num_pages_per_req,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        batch_indices,
        positions,
        batch_len_indptr,
        num_req: tl.constexpr,
        num_pages: tl.constexpr,
        page_size: tl.constexpr,
        sink_size: tl.constexpr,
        req_block: tl.constexpr,
        page_block: tl.constexpr,
):
    req = tl.arange(0, req_block)
    req_mask = req < num_req
    seq = tl.load(seq_lens + req, mask=req_mask, other=0).to(tl.int32)
    pages = (seq + page_size - 1) // page_size

    page_end = tl.cumsum(pages, axis=0)
    token_end = tl.cumsum(tl.where(req_mask, seq + sink_size, 0), axis=0)

    tl.store(num_pages_per_req + req, pages, mask=req_mask)
    tl.store(kv_last_page_len + req, seq - (pages - 1) * page_size, mask=req_mask)
    tl.store(kv_indptr + req + 1, page_end, mask=req_mask)
    tl.store(batch_len_indptr + req + 1, token_end, mask=req_mask)
    tl.store(kv_indptr + req, 0, mask=req == 0)
    tl.store(batch_len_indptr + req, 0, mask=req == 0)

    # Every page compares against all request ends in parallel.
    page = tl.arange(0, page_block)
    owner = tl.sum(
        (page[:, None] >= page_end[None, :]) & req_mask[None, :], axis=1
    )
    begin = tl.sum(
        tl.where(req[None, :] == owner[:, None] - 1, page_end[None, :], 0),
        axis=1,
    )
    page_mask = page < num_pages
    tl.store(kv_indices + page, page, mask=page_mask)
    tl.store(batch_indices + page, owner, mask=page_mask)
    tl.store(positions + page, page - begin, mask=page_mask)


def make_metadata_prefill(
        seq_lens: torch.Tensor,
        metadata: dict[str, torch.Tensor],
        sink_size: int,
        page_size: int,
) -> None:
    """Compute and write metadata in-place"""
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int32
    assert seq_lens.ndim == 1 and seq_lens.is_contiguous()
    assert page_size > 0 and sink_size >= 0

    batch_size = seq_lens.numel()
    total_pages = metadata["kv_indices"].numel()
    req_block = triton.next_power_of_2(batch_size)
    page_block = triton.next_power_of_2(total_pages)
    assert req_block * page_block <= 65536

    _build_metadata_prefill_kernel[(1,)](
        seq_lens,
        metadata["num_pages_per_req"],
        metadata["kv_indices"],
        metadata["kv_indptr"],
        metadata["kv_last_page_len"],
        metadata["batch_indices"],
        metadata["positions"],
        metadata["batch_len_indptr"],
        num_req=batch_size,
        num_pages=total_pages,
        page_size=page_size,
        sink_size=sink_size,
        req_block=req_block,
        page_block=page_block,
        num_warps=min(8, max(1, page_block // 32)),
    )


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_logits(
        num_qo_head: int,
        num_valid_pages: int,
        device: str,
        dtype: torch.dtype,
        seed: int = 42,
):
    assert dtype in (torch.float16, torch.bfloat16), "only fp16/bf16 supported"
    assert num_qo_head > 0
    assert num_valid_pages > 0

    total_values = num_qo_head * num_valid_pages

    if dtype == torch.float16:
        base = 0x3000
        max_finite_bits = 0x7BFF
    else:
        base = 0x3F00
        max_finite_bits = 0x7F7F

    capacity = max_finite_bits - base + 1
    assert total_values <= capacity, (
        f"Need {total_values} values，but {dtype} can only generate {capacity} values"
    )

    bits = torch.arange(
        total_values,
        device=device,
        dtype=torch.int32,
    ) + base

    bits = bits.to(torch.int16)
    logits = bits.view(dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    permutation = torch.randperm(
        total_values,
        device=device,
        generator=generator,
    )
    logits = logits.index_select(0, permutation)
    logits = logits.reshape(num_qo_head, num_valid_pages)

    return logits
