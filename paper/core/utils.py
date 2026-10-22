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
        max_pages_per_batch: tl.constexpr,
):
    pid = tl.program_id(0)

    req = tl.arange(0, req_block)
    req_mask = req < num_req

    seq = tl.load(seq_lens + req, mask=req_mask, other=0).to(tl.int32)
    paged_seq = seq - sink_size
    pages = (paged_seq + page_size - 1) // page_size
    page_end = tl.cumsum(pages, axis=0)
    token_end = tl.cumsum(tl.where(req_mask, seq, 0), axis=0)

    # Only the first program writes request-level metadata.
    first_program = pid == 0
    tl.store(num_pages_per_req + req, pages, mask=req_mask & first_program)
    tl.store(kv_last_page_len + req, paged_seq - (pages - 1) * page_size, mask=req_mask & first_program)
    tl.store(kv_indptr + req + 1, page_end, mask=req_mask & first_program)
    tl.store(batch_len_indptr + req + 1, token_end, mask=req_mask & first_program)
    tl.store(kv_indptr, 0, mask=first_program)
    tl.store(batch_len_indptr, 0, mask=first_program)

    # Each program handles only page_block pages.
    page = pid * page_block + tl.arange(0, page_block)
    page_mask = page < num_pages

    owner = tl.sum((page[:, None] >= page_end[None, :]) & req_mask[None, :], axis=1)

    begin = tl.sum(
        tl.where(req[None, :] == owner[:, None] - 1, page_end[None, :], 0),
        axis=1,
    )

    local_page = page - begin
    page_index = owner * max_pages_per_batch + local_page

    tl.store(kv_indices + page, page_index, mask=page_mask)
    tl.store(batch_indices + page, owner, mask=page_mask)
    tl.store(positions + page, local_page, mask=page_mask)


def make_metadata_prefill(
        seq_lens: torch.Tensor,
        num_pages_per_req: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        batch_len_indptr: torch.Tensor,
        sink_size: int,
        page_size: int,
) -> None:
    """Compute and write metadata in-place."""
    assert seq_lens.is_cuda and seq_lens.dtype == torch.int32
    assert seq_lens.ndim == 1 and seq_lens.is_contiguous()
    assert kv_indices.is_contiguous()
    assert page_size > 0 and sink_size >= 0

    batch_size = seq_lens.numel()
    assert batch_size > 0
    assert kv_indices.numel() % batch_size == 0

    max_pages_per_batch = kv_indices.numel() // batch_size

    pages_per_req = [
        ceil_div(int(seq_len) - sink_size, page_size)
        for seq_len in seq_lens
    ]
    assert min(pages_per_req) >= 0
    assert max(pages_per_req) <= max_pages_per_batch

    total_pages = sum(pages_per_req)
    req_block = triton.next_power_of_2(batch_size)

    page_block = 256
    num_page_blocks = ceil_div(total_pages, page_block)

    _build_metadata_prefill_kernel[(num_page_blocks,)](
        seq_lens,
        num_pages_per_req,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        batch_indices,
        positions,
        batch_len_indptr,
        num_req=batch_size,
        num_pages=total_pages,
        page_size=page_size,
        sink_size=sink_size,
        req_block=req_block,
        page_block=page_block,
        max_pages_per_batch=max_pages_per_batch,
        num_warps=8,
    )


def make_logits(
        num_qo_head: int,
        num_valid_pages: int,
        device: str,
        dtype: torch.dtype,
        seed: int = 42,
):
    assert dtype in (torch.float16, torch.bfloat16), "only fp16/bf16 supported"
    base = 0x3000 if dtype == torch.float16 else 0x3F00

    bits = (torch.arange(num_valid_pages, device=device, dtype=torch.int32) + base).to(torch.int16)
    logits = bits.view(dtype).unsqueeze(0).expand(num_qo_head, -1).clone()

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    perm = torch.randperm(num_valid_pages, device=device, generator=g)
    logits = logits.index_select(1, perm)

    return logits


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def assert_close(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype == b.dtype
    if a.dtype == torch.float16:
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
    elif a.dtype == torch.bfloat16:
        torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)
