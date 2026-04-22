import torch
import flashinfer


class PagedKVCache:
    def __init__(
            self,
            num_layers: int,
            head_dim: int,
            num_qo_heads: int,
            num_kv_heads: int,
            max_length: int,
            sink: int = 4,
            page_size: int = 32,  # must greater than 16
            device: torch.device = "cuda:0",
            dtype: torch.dtype = torch.float16
    ):
        self.sink = sink
        self.device = device
        self.dtype = dtype
        self.page_size = page_size
        self.max_length = max_length
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.num_pages = (self.max_length + page_size - 1) // page_size

        # paged cache metadata, only support batch size of 1
        self.seq_len = 0
        self.kv_page_indptr = torch.tensor([0, 0], dtype=torch.int32, device=device)
        self.kv_page_indices = torch.arange(0, dtype=torch.int32, device=device)
        self.current_pages = 0

        # paged kv cache
        self.paged_k_cache = torch.zeros(
            num_layers, self.num_pages, page_size, num_kv_heads, head_dim,
            dtype=dtype,
            device=device
        ).contiguous()

        self.paged_v_cache = torch.zeros(
            num_layers, self.num_pages, page_size, num_kv_heads, head_dim,
            dtype=dtype,
            device=device
        ).contiguous()

        # the reduced kv cache buffer
        self.pooling = torch.zeros(
            num_layers, self.num_pages, self.num_kv_heads, self.head_dim,
            dtype=dtype,
            device=device
        ).contiguous()

    def append_page_prefill(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.current_pages = (self.seq_len + self.page_size - 1) // self.page_size
        self.kv_page_indptr[1] = self.current_pages
        self.kv_page_indices = torch.arange(self.current_pages, dtype=torch.int32, device=self.device)
        last_page_len = self.seq_len - ((self.current_pages - 1) * self.page_size)
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=self.device)

        flashinfer.append_paged_kv_cache_prefill(
            self.sink,
            k,
            v,
            (self.paged_k_cache[layer_id], self.paged_v_cache[layer_id]),
            self.kv_page_indices,
            self.kv_page_indptr,
            kv_last_page_len,
            self.pooling[layer_id],
            kv_layout="NHD"
        )

    def append_page_decode(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.current_pages = (self.seq_len + self.page_size - 1) // self.page_size
        self.kv_page_indptr[1] = self.current_pages
        last_page_len = self.seq_len - ((self.current_pages - 1) * self.page_size)
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=self.device)

        if last_page_len == 1:
            self.kv_page_indices = torch.cat(
                [
                    self.kv_page_indices,
                    torch.tensor([self.current_pages - 1], device=self.device)
                ], dim=0
            )

        flashinfer.append_paged_kv_cache_decode(
            k,
            v,
            (self.paged_k_cache[layer_id], self.paged_v_cache[layer_id]),
            self.kv_page_indices,
            self.kv_page_indptr,
            kv_last_page_len,
            self.pooling[layer_id],
            kv_layout="NHD"
        )
