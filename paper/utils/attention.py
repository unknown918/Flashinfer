import torch
import flashinfer
from paper.utils.cache import PagedKVCache


class AttentionRunner:
    def __init__(
            self,
            num_layers: int,
            head_dim: int,
            num_kv_heads: int,
            num_qo_heads: int,
            page_size: int,
            max_length: int,
            sink: int = 4,
            topk: int = 16,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda:0"
    ):
        device = torch.device(device)

        self.sink = sink
        self.topk = topk
        self.page_size = page_size
        self.max_length = max_length
        self.group_size = num_qo_heads // num_kv_heads

        # sparse attention buffers (one for all layers)
        self.indptr = torch.zeros(
            1 + num_kv_heads,
            dtype=torch.int32,
            device=device
        )
        self.indices = torch.zeros(
            num_kv_heads * (self.sink + self.topk * self.group_size * page_size),
            dtype=torch.int32,
            device=device
        )
        self.mask_logits = torch.zeros(
            num_qo_heads // self.group_size, max_length,
            dtype=torch.int32,
            device=device
        )

        num_pages = int((max_length + page_size - 1) / page_size)

        # decode wrapper and buffers
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.decode_wrapper = flashinfer.SparseSinkAttentionWrapper(workspace_buffer)
        self.logits = torch.zeros(num_qo_heads, num_pages, dtype=dtype, device=device)
        self.out = torch.zeros(num_qo_heads, head_dim, dtype=dtype, device=device)

        # paged kv cache manager
        self.cache_manager = PagedKVCache(
            num_layers=num_layers,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            max_length=max_length,
            sink=sink,
            page_size=page_size,
            device=device,
            dtype=dtype
        )

    def prefill(self, layer_id: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k = flashinfer.apply_llama31_rope(
            q,
            k,
            torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device),
            torch.tensor([0], dtype=torch.int32, device=q.device),
        )

        self.cache_manager.append_page_prefill(layer_id, k, v)

        # no sparsity for prefill
        return flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k,
            v=v,
            causal=True,
            return_lse=False
        )

    def decode(self, layer_id: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k = flashinfer.apply_llama31_rope(
            q.unsqueeze(0),
            k.unsqueeze(0),
            torch.tensor([0, 1], dtype=torch.int32, device=q.device),
            torch.tensor([self.cache_manager.seq_len], dtype=torch.int32, device=q.device),
        )

        # TODO: could use a cuda graph to replay this process
        self.cache_manager.append_page_decode(layer_id, k[0], v)

        # fixme: sometimes got zero logits?
        self.logits = flashinfer.decode.estimate(
            seq_len=self.cache_manager.current_pages,
            query=q[0],
            pooling=self.cache_manager.pooling[layer_id] / 32,
            out=self.logits
        )

        num_valid_pages = (self.cache_manager.seq_len - 4 + self.page_size - 1) // self.page_size
        last_page_len = (self.cache_manager.seq_len - 4) - (num_valid_pages - 1) * self.page_size

        # expand top-k token's indices
        self.indptr, self.indices = flashinfer.topk_bool_mask_logits(
            topk=self.topk - 1,
            page_size=self.page_size,
            last_page_len=last_page_len,
            num_valid_pages=num_valid_pages - 1,  # always select last page
            logits=self.logits,
            indptr=self.indptr,
            indices=self.indices,
            group_size=self.group_size,
            mask_logits=self.mask_logits
        )
        self.indptr = self.indptr.cumsum(dim=0, dtype=torch.int32)
        # split-K and other stuff
        self.decode_wrapper.plan(
            kv_indptr=self.indptr,
            kv_indices=self.indices,
            num_qo_heads=self.cache_manager.num_qo_heads,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            causal=True
        )

        attn_output = self.decode_wrapper.run(
            q=q[0],
            k=self.cache_manager.paged_k_cache[layer_id],
            v=self.cache_manager.paged_v_cache[layer_id],
            out=self.out,
            return_lse=False
        )

        self.indptr.zero_()
        self.indices.zero_()
        self.logits.zero_()

        return attn_output
