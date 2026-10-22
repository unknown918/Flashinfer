import torch
import flashinfer
from .utils import make_metadata_prefill
from typing import Union


class PagedKVCache:
    def __init__(
            self,
            batch_size: int,
            num_layers: int,
            head_dim: int,
            num_qo_heads: int,
            num_kv_heads: int,
            max_length: int,
            sink_size: int = 4,
            page_size: int = 32,
            device: Union[str, torch.device] = "cuda:0",
            dtype: torch.dtype = torch.float16
    ):
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.sink_size = sink_size
        self.page_size = page_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        # metadata for sparse attention's kv cache
        self.sink_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.max_num_pages = batch_size * ((self.max_length + page_size - 1) // page_size)
        self.kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.kv_indices = torch.zeros(self.max_num_pages, dtype=torch.int32, device=device)
        self.kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.num_pages_per_req = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.batch_indices = torch.zeros(self.max_num_pages, dtype=torch.int32, device=device)
        self.positions = torch.zeros(self.max_num_pages, dtype=torch.int32, device=device)
        self.batch_len_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

        # KV cache, init on cpu to avoid OOM issues
        self.paged_k_cache = torch.zeros(
            num_layers, self.max_num_pages, num_kv_heads, page_size, head_dim,
            dtype=dtype,
            device="cpu",
            pin_memory=True
        ).contiguous()
        self.paged_k_buffer = torch.zeros(
            self.max_num_pages, num_kv_heads, page_size, head_dim,
            dtype=dtype,
            device=self.device,
        ).contiguous()
        self.paged_v_cache = torch.zeros(
            num_layers, self.max_num_pages, num_kv_heads, page_size, head_dim,
            dtype=dtype,
            device=self.device,
        ).contiguous()
        self.sink_k_cache = torch.zeros(
            num_layers, batch_size, num_kv_heads, sink_size, head_dim,
            dtype=dtype,
            device=self.device,
        ).contiguous()
        self.sink_v_cache = torch.zeros(
            num_layers, batch_size, num_kv_heads, sink_size, head_dim,
            dtype=dtype,
            device=self.device,
        ).contiguous()
        self.pooling = torch.zeros(
            num_layers, self.max_num_pages, self.num_kv_heads, self.head_dim,
            dtype=dtype,
            device=self.device,
        ).contiguous()

    def offload(self, layer_id: int) -> None:
        self.paged_k_cache[layer_id].copy_(
            self.paged_k_buffer,
            non_blocking=True,
        )

    def onload(self, layer_id: int) -> None:
        self.paged_k_buffer.copy_(
            self.paged_k_cache[layer_id],
            non_blocking=True,
        )

    def update_metadata_prefill(self) -> None:
        make_metadata_prefill(
            seq_lens=self.seq_lens,
            num_pages_per_req=self.num_pages_per_req,
            kv_indices=self.kv_indices,
            kv_indptr=self.kv_indptr,
            kv_last_page_len=self.kv_last_page_len,
            batch_indices=self.batch_indices,
            positions=self.positions,
            batch_len_indptr=self.batch_len_indptr,
            sink_size=self.sink_size,
            page_size=self.page_size
        )

    def update_metadata_decode(self) -> None:
        self.num_pages_per_req.zero_()
        self.kv_indices.zero_()
        self.kv_indptr.zero_()
        self.kv_last_page_len.zero_()
        self.batch_indices.zero_()
        self.positions.zero_()
        self.batch_len_indptr.zero_()
        make_metadata_prefill(
            seq_lens=self.seq_lens,
            num_pages_per_req=self.num_pages_per_req,
            kv_indices=self.kv_indices,
            kv_indptr=self.kv_indptr,
            kv_last_page_len=self.kv_last_page_len,
            batch_indices=self.batch_indices,
            positions=self.positions,
            batch_len_indptr=self.batch_len_indptr,
            sink_size=self.sink_size,
            page_size=self.page_size
        )

    def append_page_prefill(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.onload(layer_id)
        torch.cuda.synchronize()  # need immediate use, ensure k-cache is in-place
        flashinfer.append_paged_kv_cache_prefill(
            append_key=k.flatten(0, 1),
            append_value=v.flatten(0, 1),
            paged_kv_cache=(self.paged_k_buffer, self.paged_v_cache[layer_id]),
            kv_indices=self.kv_indices,
            kv_indptr=self.kv_indptr,
            kv_last_page_len=self.kv_last_page_len,
            pooling=self.pooling[layer_id],
            sink_k=self.sink_k_cache[layer_id],
            sink_v=self.sink_v_cache[layer_id],
            sink_indices=self.sink_indices,
            batch_indices=self.batch_indices,
            positions=self.positions,
            batch_len_indptr=self.batch_len_indptr,
            kv_layout="HND"
        )
        self.offload(layer_id)

    def append_page_decode(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        flashinfer.append_paged_kv_cache_decode(
            append_key=k.flatten(0, 1),
            append_value=v.flatten(0, 1),
            paged_kv_cache=(self.paged_k_cache[layer_id], self.paged_v_cache[layer_id]),
            kv_indices=self.kv_indices,
            kv_indptr=self.kv_indptr,
            kv_last_page_len=self.kv_last_page_len,
            pooling=self.pooling[layer_id],
            kv_layout="HND"
        )

    def flush_cache(self):
        self.seq_lens.zero_()
        self.num_pages_per_req.zero_()
        self.kv_indices.zero_()
        self.kv_indptr.zero_()
        self.kv_last_page_len.zero_()
        self.batch_indices.zero_()
        self.positions.zero_()
        self.batch_len_indptr.zero_()

        self.sink_k_cache.zero_()
        self.sink_v_cache.zero_()
        self.paged_k_cache = self.paged_k_cache.zero_().to("cpu")
        self.paged_v_cache.zero_()
        self.pooling.zero_()
