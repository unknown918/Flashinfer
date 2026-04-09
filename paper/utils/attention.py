import torch
import flashinfer
from cache import PagedKVCache


class AttentionRunner:
    def __init__(
            self,
            num_layers: int,
            head_dim: int,
            num_kv_heads: int,
            num_qo_heads: int,
            page_size: int,
            max_length: int,
            topk: int = 16,
            dtype: torch.dtype = torch.float16,
            device: torch.device = "cuda:0"
    ):
        self.sink = 4  # hard coded in kernel
        self.topk = topk
        self.page_size = page_size
        self.max_length = max_length
        self.group_size = num_qo_heads // num_kv_heads

        # sparse attention buffers (one for all layers)
        self.indptr = torch.zeros(
            1 + num_kv_heads,
            dtype=torch.uint32,
            device=device
        )
        self.indices = torch.zeros(
            num_kv_heads * (self.sink + self.topk * self.group_size * page_size),
            dtype=torch.uint32,
            device=device
        )
        self.mask_logits = torch.zeros(
            num_qo_heads // self.group_size, max_length,
            dtype=torch.uint32,
            device=device
        )

        num_pages = int((max_length + page_size - 1) / page_size)

        # decode wrapper and buffers
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.decode_wrapper = flashinfer.SparseSinkAttentionWrapper(workspace_buffer)
        self.logits = torch.zeros(num_qo_heads, num_pages, dtype=dtype, device=device)
        self.out = torch.zeros(num_kv_heads, head_dim, dtype=dtype, device=device)

        # paged kv cache manager
        self.cache_manager = PagedKVCache(
            num_layers=num_layers,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            max_length=max_length,
            sink=self.sink,
            page_size=page_size,
            device=device,
            dtype=dtype
        )

    def prefill(self, layer_id: int, q: torch.Tenspr, k: torch.Tensor, v: torch.Tenssor) -> torch.Tensor:
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
        # TODO: could use a cuda graph to replay this process
        self.cache_manager.append_page_decode(layer_id, k, v)

        # estimate token importance
        flashinfer.decode.estimate(
            q,
            out=self.logits,
            seq_len=self.cache_manager.seq_len,
            pooling=self.cache_manager.pooling
        )

        # expand top-k token's indices
        flashinfer.topk_bool_mask_logits(
            input=self.logits,
            indptr=self.indptr,
            indices=self.indices,
            mask_logits=self.mask_logits,
            k=self.topk,
            max_length=self.cache_manager.num_pages,
            group_size=self.group_size,
            block_size=self.cache_manager.page_size
        )

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
            q=q,
            k=self.cache_manager.paged_k_cache[layer_id],
            v=self.cache_manager.paged_v_cache[layer_id],
            out=self.out,
            return_lse=False
        )

        return attn_output
