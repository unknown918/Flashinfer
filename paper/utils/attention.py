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
        self.device = torch.device(device)

        self.sink = sink
        self.topk = topk
        self.budgets = topk * page_size
        self.page_size = page_size
        self.max_length = max_length
        self.group_size = num_qo_heads // num_kv_heads

        # sparse attention buffers (one for all layers)
        self.indices = torch.zeros(
            num_kv_heads * (self.sink + self.topk * self.group_size * page_size),
            dtype=torch.int32,
            device=self.device
        )
        self.indptr = torch.zeros(1 + num_kv_heads, dtype=torch.int32, device=self.device)
        self.row_states_buffer = torch.empty(1024 * 1024, dtype=torch.uint8, device=self.device)
        self.mask_logits = torch.zeros(num_kv_heads, max_length, dtype=torch.int32, device=self.device)

        num_pages = int((max_length + page_size - 1) / page_size)
        self.layer_id = 0
        self.num_valid_pages = 0
        self.last_page_len = 0

        # rope indexers
        self.rope_indptr = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        self.rope_offset = torch.tensor([0], dtype=torch.int32, device=self.device)

        # decode wrapper and buffers
        self.logits = torch.zeros(num_qo_heads, num_pages, dtype=dtype, device=self.device).contiguous()
        self.output = torch.zeros(num_qo_heads, head_dim, dtype=dtype, device=self.device).contiguous()
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device).contiguous()
        self.is_planned = False
        self.decode_wrapper = flashinfer.SparseSinkAttentionWrapper(
            num_kv_heads=num_kv_heads,
            device=self.device,
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
            device=self.device,
            dtype=dtype
        )

        # static buffer for decode cuda graph
        self.topk -= 1
        self.q = torch.zeros(num_qo_heads, head_dim, dtype=dtype, device=self.device).contiguous()
        self.graphs = [
            # (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph())
            torch.cuda.CUDAGraph()
            for _ in range(num_layers - 2)
        ]
        self.graph_captured = [
            False
            for _ in range(num_layers - 2)
        ]

        self.proxy_plan()

    def proxy_plan(self):
        proxy_indptr = torch.full(
            (self.cache_manager.num_kv_heads + 1,),
            self.topk * self.page_size + self.sink,
            dtype=torch.int32,
            device=self.device
        ).cumsum(dim=0, dtype=torch.int32)
        proxy_indices = torch.full(
            (self.cache_manager.num_kv_heads * (self.topk * self.page_size + self.sink),),
            1,
            dtype=torch.int32,
            device=self.device
        )
        self.decode_wrapper.plan(
            kv_indptr=proxy_indptr,
            kv_indices=proxy_indices,
            num_qo_heads=self.cache_manager.num_qo_heads,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            causal=True
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
            self,
            layer_id: int,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        assert (q.dim() == 2 and k.dim() == 2 and v.dim() == 2)  # (num_heads, head_dim)

        self.cache_manager.append_page_decode(layer_id, k, v)

        if layer_id < 2:  # no sparsity for the first two layers
            output = flashinfer.decode.single_decode_with_kv_cache(
                q,
                self.cache_manager.paged_k_cache[layer_id].reshape(
                    -1, self.cache_manager.num_kv_heads, self.cache_manager.head_dim
                )[:self.cache_manager.seq_len],
                self.cache_manager.paged_v_cache[layer_id].reshape(
                    -1, self.cache_manager.num_kv_heads, self.cache_manager.head_dim
                )[:self.cache_manager.seq_len],
            )
            return output

        self.q.copy_(q)
        self.layer_id = layer_id

        self.num_valid_pages = (self.cache_manager.seq_len - self.sink + self.page_size - 1) // self.page_size
        self.last_page_len = self.cache_manager.seq_len - self.sink - (self.num_valid_pages - 1) * self.page_size

        flashinfer.estimate(
            query=self.q,
            pooling=self.cache_manager.pooling[self.layer_id] / self.cache_manager.page_size,
            num_valid_pages=self.num_valid_pages,
            out=self.logits,
        )

        # expand top-k token's indices
        flashinfer.topk_bool_mask_logits(
            top_k=self.topk,
            page_size=self.page_size,
            last_page_len=self.last_page_len,
            num_valid_pages=self.num_valid_pages - 1,  # always select last page
            logits=self.logits,
            indptr=self.indptr,
            indices=self.indices,
            group_size=self.group_size,
            mask_logits=self.mask_logits,
            row_states_buffer=self.row_states_buffer,
        )

        g = self.graphs[self.layer_id - 2]
        if not self.graph_captured[self.layer_id - 2]:
            self.graph_captured[layer_id - 2] = True
            with torch.cuda.graph(g):
                torch.cumsum(self.indptr, dim=0, dtype=torch.int32, out=self.indptr)
                self.decode_wrapper.run(
                    q=self.q,
                    k=self.cache_manager.paged_k_cache[self.layer_id],
                    v=self.cache_manager.paged_v_cache[self.layer_id],
                    kv_indptr=self.indptr,
                    kv_indices=self.indices,
                    out=self.output,
                    return_lse=False
                )
        else:
            g.replay()

        self.indptr.zero_()
        self.indices.zero_()
        self.logits.zero_()
        self.mask_logits.zero_()

        return self.output
