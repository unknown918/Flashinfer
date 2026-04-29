import torch
import flashinfer
from paper.utils.cache import PagedKVCache
from typing import Tuple


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
            num_kv_heads, max_length,
            dtype=torch.int32,
            device=device
        )

        num_pages = int((max_length + page_size - 1) / page_size)

        # rope indexers
        self.rope_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self.rope_offset = torch.tensor([0], dtype=torch.int32, device=device)

        # decode wrapper and buffers
        self.logits = torch.zeros(num_qo_heads, num_pages, dtype=dtype, device=device).contiguous()
        self.output = torch.zeros(num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
        self.decode_wrapper = flashinfer.SparseSinkAttentionWrapper(
            num_kv_heads=num_kv_heads,
            device=device,
            float_workspace_buffer=workspace_buffer
        )

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

    def apply_rope(
            self,
            layer_id: int,
            q: torch.Tensor,
            k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (q.dim() == 3 and k.dim() == 3)  # (seq_len, num_heads, head_dim)
        if layer_id == 0:
            self.cache_manager.seq_len += q.shape[0]

        if q.shape[0] > 1:
            self.rope_indptr[1] = self.cache_manager.seq_len
            self.rope_offset[0] = 0
        else:
            self.rope_indptr[1] = 1
            self.rope_offset[0] = self.cache_manager.seq_len - 1

        flashinfer.rope.apply_llama31_rope_inplace(
            q=q,  # batch size is 1
            k=k,
            indptr=self.rope_indptr,
            offsets=self.rope_offset,
            interleave=False,
            rope_scale=8.0,
            rope_theta=500000.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            old_context_len=8192,
        )
        return q, k

    def prefill(self, layer_id: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        assert (q.dim() == 3 and k.dim() == 3 and v.dim() == 3)  # (seq_len, num_heads, head_dim)

        self.cache_manager.append_page_prefill(layer_id, k, v)

        # no sparsity for prefill
        return flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k,
            v=v,
            causal=True,
            return_lse=False
        )

    def decode(
            self, layer_id: int,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        assert (q.dim() == 2 and k.dim() == 2 and v.dim() == 2)  # (num_heads, head_dim)

        # TODO: could use a cuda graph to replay this process
        self.cache_manager.append_page_decode(layer_id, k, v)

        if layer_id < 2:  # no sparse for first two layers
            torch.cuda.nvtx.range_push("Full Attn")
            output = flashinfer.decode.single_decode_with_kv_cache(
                q,
                self.cache_manager.paged_k_cache[layer_id].reshape(
                    -1, self.cache_manager.num_kv_heads, self.cache_manager.head_dim
                )[:self.cache_manager.seq_len],
                self.cache_manager.paged_v_cache[layer_id].reshape(
                    -1, self.cache_manager.num_kv_heads, self.cache_manager.head_dim
                )[:self.cache_manager.seq_len],
            )
            torch.cuda.nvtx.range_pop()
            return output

        torch.cuda.nvtx.range_push("Sparse Attn")

        num_valid_pages = (self.cache_manager.seq_len - self.sink + self.page_size - 1) // self.page_size
        last_page_len = self.cache_manager.seq_len - self.sink - (num_valid_pages - 1) * self.page_size

        torch.cuda.nvtx.range_push("Estimate")
        flashinfer.decode.estimate(
            query=q,
            pooling=self.cache_manager.pooling[layer_id] / self.cache_manager.page_size,
            num_valid_pages=num_valid_pages,  # last page may have value, or not, but does not affect next kernel
            out=self.logits,
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("TopK")
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
        torch.cuda.nvtx.range_pop()


        torch.cuda.nvtx.range_push("Attn Plan")
        # split-K and other stuff
        self.decode_wrapper.plan(
            kv_indptr=self.indptr,
            kv_indices=self.indices,
            num_qo_heads=self.cache_manager.num_qo_heads,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            causal=True
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Attn Run")
        self.decode_wrapper.run(
            q=q,
            k=self.cache_manager.paged_k_cache[layer_id],
            v=self.cache_manager.paged_v_cache[layer_id],
            out=self.output,
            return_lse=False
        )
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()

        self.indptr.zero_()
        self.indices.zero_()
        self.logits.zero_()
        self.mask_logits.zero_()

        return self.output
