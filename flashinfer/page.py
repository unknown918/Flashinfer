"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from typing import Optional, Tuple, Union

import torch

from .jit.page import gen_page_module
from .utils import (
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
    register_custom_op,
    register_fake_op,
)


@functools.cache
def get_page_module():
    return gen_page_module().build_and_load()


def block_sparse_indices_to_vector_sparse_offsets(
        block_sparse_indices: torch.Tensor,
        block_sparse_indptr: torch.Tensor,
        vector_sparse_offsets: torch.Tensor,
        vector_sparse_indptr: torch.Tensor,
        kv_lens: torch.Tensor,
        stride_block: int,
        stride_n: int,
        block_size: int,
) -> torch.Tensor:
    if block_size == 1:
        if stride_block == 1:
            return block_sparse_indices
        else:
            return block_sparse_indices * stride_block

    assert block_sparse_indices.dtype == torch.int32
    assert block_sparse_indptr.dtype == torch.int32
    assert vector_sparse_offsets.dtype == torch.int32
    assert vector_sparse_indptr.dtype == torch.int32
    assert kv_lens.dtype == torch.int32
    batch_size = block_sparse_indptr.size(0) - 1
    get_page_module().block_sparse_indices_to_vector_sparse_offsets(
        block_sparse_indices,
        block_sparse_indptr,
        vector_sparse_offsets,
        vector_sparse_indptr,
        kv_lens,
        stride_block,
        stride_n,
        batch_size,
        block_size,
    )
    return vector_sparse_offsets


@register_custom_op(
    "flashinfer::append_paged_mla_kv_cache",
    mutates_args=("ckv_cache", "kpe_cache"),
)
def _append_paged_mla_kv_cache_kernel(
        append_ckv: torch.Tensor,
        append_kpe: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        ckv_cache: Optional[torch.Tensor],
        kpe_cache: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().append_paged_mla_kv_cache(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


@register_custom_op(
    "flashinfer::append_paged_kv_cache",
    mutates_args=("paged_k_cache", "paged_v_cache"),
)
def _append_paged_kv_cache_kernel(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        layout: int,
) -> None:
    batch_indices = batch_indices.int()
    positions = positions.int()
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        layout,
    )


@register_custom_op(
    "flashinfer::append_paged_kv_cache_prefill",
    mutates_args=("paged_k_cache", "paged_v_cache"),
)
def _append_paged_kv_cache_prefill_kernel(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        pooling: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
        sink_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        batch_len_indptr: torch.Tensor,
        layout: int,
) -> None:
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().append_paged_kv_cache_prefill(
        append_key,
        append_value,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        pooling,
        sink_k,
        sink_v,
        sink_indices,
        batch_indices,
        positions,
        batch_len_indptr,
        layout,
    )


@register_custom_op(
    "flashinfer::append_paged_kv_cache_decode",
    mutates_args=("paged_k_cache", "paged_v_cache"),
)
def _append_paged_kv_cache_decode_kernel(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        layout: int,
        pooling: torch.Tensor
) -> None:
    kv_indices = kv_indices.int()
    kv_indptr = kv_indptr.int()
    kv_last_page_len = kv_last_page_len.int()
    get_page_module().append_paged_kv_cache_decode(
        append_key,
        append_value,
        paged_k_cache,
        paged_v_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        pooling,
        layout,
    )


@register_fake_op("flashinfer::append_paged_kv_cache")
def _fake_append_paged_kv_cache_kernel(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        layout: int,
) -> None:
    pass


def get_batch_indices_positions(
        append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Convert append indptr and sequence lengths to batch indices and positions.

    Parameters
    ----------
    append_indptr : torch.Tensor
        The indptr of the ragged tensor, shape: ``[batch_size + 1]``.
    seq_lens: torch.Tensor
        The sequence lengths of each request in the KV-Cache, shape: ``[batch_size]``.
    nnz : int
        The number of entries in the ragged tensor.

    Returns
    -------
    batch_indices: torch.Tensor
        The batch indices of each entry in the ragged tensor, shape: ``[nnz]``.
    positions: torch.Tensor
        The positions of each entry in the ragged tensor, shape: ``[nnz]``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> nnz_kv = 10
    >>> append_indptr = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32, device="cuda:0")
    >>> seq_lens = torch.tensor([5, 5, 5, 5])
    >>> batch_indices, positions = flashinfer.get_batch_indices_positions(append_indptr, seq_lens, nnz_kv)
    >>> batch_indices
    tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], device='cuda:0', dtype=torch.int32)
    >>> positions  # the rightmost column index of each row
    tensor([4, 3, 4, 2, 3, 4, 1, 2, 3, 4], device='cuda:0', dtype=torch.int32)

    Note
    ----
    This function is similar to `CSR2COO <https://docs.nvidia.com/cuda/cusparse/#csr2coo>`_
    conversion in cuSPARSE library, with the difference that we are converting from a ragged
    tensor (which doesn't require a column indices array) to a COO format.

    See Also
    --------
    append_paged_kv_cache
    """
    batch_size = append_indptr.size(0) - 1
    batch_indices = torch.empty((nnz,), device=append_indptr.device, dtype=torch.int32)
    positions = torch.empty((nnz,), device=append_indptr.device, dtype=torch.int32)
    from .triton.page import get_batch_indices_positions_kernel

    get_batch_indices_positions_kernel[(batch_size,)](
        append_indptr, seq_lens, batch_indices, positions, num_stages=2
    )
    return batch_indices, positions


def get_seq_lens(
        kv_indptr: torch.Tensor, kv_last_page_len: torch.Tensor, page_size: int
) -> torch.Tensor:
    r"""Convert KV indptr and last page length to sequence lengths.

    Parameters
    ----------
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    page_size : int
        The size of a page in the paged kv-cache.

    Returns
    -------
    seq_lens: torch.Tensor
        The sequence lengths of each request in the paged kv-cache, shape: ``[batch_size]``.
    """
    return (
            torch.clamp(kv_indptr[1:] - kv_indptr[:-1] - 1, min=0) * page_size
            + kv_last_page_len
    )


def append_paged_mla_kv_cache(
        append_ckv: torch.Tensor,
        append_kpe: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        ckv_cache: Optional[torch.Tensor],
        kpe_cache: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
) -> None:
    r"""Append a batch of key-value pairs to a paged key-value cache,
    Note: current only support ckv=512 and kpe=64

    Parameters
    ----------
    append_ckv : torch.Tensor
        The compressed kv tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], ckv_dim]``.
    append_kpe : torch.Tensor
        The value tensor to append in ragged tensor format, shape:
        ``[append_indptr[-1], kpe_dim]``.
    batch_indices : torch.Tensor
        The batch indices of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    positions : torch.Tensor
        The positions of the each entry in the appended key-value pairs, shape: ``[append_indptr[-1]]``.
    ckv_cache : cache for compressed kv, torch.Tensor, shape: [page_num, page_size, ckv_dim]
    kpe_cache : cache for key position embedding, torch.Tensor, shape: [page_num, page_size, kpe_dim]
    kv_indices : torch.Tensor
        The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]``.
    kv_indptr : torch.Tensor
        The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
    kv_last_page_len : torch.Tensor
        The number of entries in the last page of each request in the paged kv cache,
        shape: ``[batch_size]``.
    """
    _append_paged_mla_kv_cache_kernel(
        append_ckv,
        append_kpe,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )


def append_paged_kv_cache(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        kv_layout: str = "NHD",
) -> None:
    _check_kv_layout(kv_layout)
    _append_paged_kv_cache_kernel(
        append_key,
        append_value,
        batch_indices,
        positions,
        *_unpack_paged_kv_cache(paged_kv_cache, kv_layout),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        TensorLayout[kv_layout].value,
    )


def append_paged_kv_cache_decode(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        pooling: torch.Tensor,
        kv_layout: str = "NHD",
) -> None:
    _check_kv_layout(kv_layout)
    _append_paged_kv_cache_decode_kernel(
        append_key,
        append_value,
        *_unpack_paged_kv_cache(paged_kv_cache, kv_layout),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        TensorLayout[kv_layout].value,
        pooling
    )


def append_paged_kv_cache_prefill(
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        pooling: torch.Tensor,
        sink_k: torch.Tensor,
        sink_v: torch.Tensor,
        sink_indices: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        batch_len_indptr: torch.Tensor,
        kv_layout: str = "NHD",
) -> None:
    _check_kv_layout(kv_layout)
    _append_paged_kv_cache_prefill_kernel(
        append_key,
        append_value,
        *_unpack_paged_kv_cache(paged_kv_cache, kv_layout),
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        pooling,
        sink_k,
        sink_v,
        sink_indices,
        batch_indices,
        positions,
        batch_len_indptr,
        TensorLayout[kv_layout].value
    )
