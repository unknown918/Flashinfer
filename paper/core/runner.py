import torch
import flashinfer
from flash_attn import flash_attn_func
from .rope import RoPE
from .cache import PagedKVCache
from .decode import AttentionBatchDecodeWrapper, AttentionSparseDecodeWrapper
from typing import Union


class AttentionRunner:
    def __init__(
            self,
            batch_size: int,
            num_layers: int,
            head_dim: int,
            num_kv_heads: int,
            num_qo_heads: int,
            sink_size: int,
            page_size: int,
            max_length: int,
            topk: int = 16,
            dtype: torch.dtype = torch.float16,
            device: Union[str, torch.device] = "cuda:0",
    ):
        self.dtype = dtype
        self.batch_size = batch_size
        self.device = device

        self.topk = topk
        self.sink_size = sink_size
        self.page_size = page_size
        self.max_length = max_length
        self.group_size = num_qo_heads // num_kv_heads

        # sparse attention buffers (one for all layers)
        self.sink_indices = torch.zeros(num_kv_heads * batch_size, dtype=torch.int32, device=device)
        self.sparse_indices = torch.zeros(
            batch_size * num_kv_heads * topk * self.group_size,
            dtype=torch.int32,
            device=device
        )
        self.sparse_indptr = torch.zeros(1 + batch_size * num_kv_heads, dtype=torch.int32, device=device)
        self.mask_logits = torch.zeros(batch_size * num_kv_heads, max_length, dtype=torch.int32, device=device)

        num_pages = int((max_length + page_size - 1) / page_size)
        self.layer_id = 0
        self.num_valid_pages = 0
        self.last_page_len = 0

        # decode wrapper and buffers
        self.logits = torch.zeros(batch_size * num_qo_heads, num_pages, dtype=dtype, device=device).contiguous()
        self.output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
        self.is_planned = False
        self.dense_decoder = AttentionBatchDecodeWrapper(float_workspace_buffer=workspace_buffer)
        self.sparse_decoder = AttentionSparseDecodeWrapper(float_workspace_buffer=workspace_buffer)

        # paged kv cache manager
        self.cache_manager = PagedKVCache(
            batch_size=batch_size,
            num_layers=num_layers,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            max_length=max_length,
            sink_size=sink_size,
            page_size=page_size,
            device=device,
            dtype=dtype
        )

        # Llama-3.1 style RoPE
        self.rope = RoPE(
            batch_size=batch_size,
            rope_scale=8.0,
            rope_theta=500000.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            old_context_len=max_length,
            device=device,
        )

        # static buffer for decode cuda graph
        self.q = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=self.device).contiguous()
        self.graphs = [
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
            (self.cache_manager.num_kv_heads, 1), self.topk * self.page_size,
            dtype=torch.int32,
            device=self.device
        )
        proxy_indptr[0] = 0
        proxy_indptr.cumsum(dim=0, dtype=torch.int32)

        self.dense_decoder.plan(
            kv_indptr=proxy_indptr,
            page_size=self.page_size,
            num_qo_heads=self.cache_manager.num_qo_heads,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            q_data_type=self.dtype,
            kv_data_type=self.dtype
        )

        self.sparse_decoder.plan(
            kv_indptr=proxy_indptr,
            page_size=self.page_size,
            num_qo_heads=self.cache_manager.num_qo_heads,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            q_data_type=self.dtype,
            kv_data_type=self.dtype
        )

    def prefill(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        assert (q.dim() == 4 and k.dim() == 4 and v.dim() == 4)  # (batch_size, seq_len, num_heads, head_dim)

        if self.layer_id == 0:
            self.rope.update_index_prefill(q.shape[1])
            self.cache_manager.seq_lens.fill_(q.shape[1])
            self.cache_manager.update_metadata_prefill()

        q, k = self.rope.apply_rope(q, k)
        self.cache_manager.append_page_prefill(self.layer_id, k, v)

        # no sparsity for prefill
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
        )

    def decode(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
    ) -> torch.Tensor:
        assert (q.dim() == 4 and k.dim() == 4 and v.dim() == 4)  # (batch_size, seq_len, num_heads, head_dim)
        assert (q.shape[1] == 1 and k.shape[1] == 1 and v.shape[1] == 1)

        if self.layer_id == 0:
            self.cache_manager.seq_lens += 1
            self.rope.update_index_decode()
            self.cache_manager.update_metadata_decode()

        q, k = self.rope.apply_rope(q, k)
        self.cache_manager.append_page_decode(self.layer_id, k, v)
        self.q.copy_(q[:, 0, :, :])

        if self.layer_id < 2:  # no sparsity for the first two layers
            self.dense_decoder.run(
                q=self.q,
                paged_k_cache=self.cache_manager.paged_k_cache[self.layer_id],
                paged_v_cache=self.cache_manager.paged_v_cache[self.layer_id],
                sink_k_cache=self.cache_manager.sink_k_cache[self.layer_id],
                sink_v_cache=self.cache_manager.sink_v_cache[self.layer_id],
                kv_indptr=self.cache_manager.kv_indptr,
                kv_indices=self.cache_manager.kv_indices,
                kv_last_page_len=self.cache_manager.kv_last_page_len,
                out=self.output,
                kv_layout="HND",
            )
            return self.output

        # g = self.graphs[self.layer_id - 2]
        # if not self.graph_captured[self.layer_id - 2]:
        #     self.graph_captured[layer_id - 2] = True
        #     with torch.cuda.graph(g):
        flashinfer.estimate(
            query=self.q,
            pooling=self.cache_manager.pooling[self.layer_id],  # / self.cache_manager.page_size,
            kv_indptr=self.cache_manager.kv_indptr,
            kv_indices=self.cache_manager.kv_indices,
            out=self.logits,
        )

        # expand top-k token's indices
        flashinfer.topk_bool_mask_logits(
            logits=self.logits,
            mask_logits=self.mask_logits,
            indptr=self.cache_manager.kv_indptr,
            indices=self.cache_manager.kv_indices,
            sparse_indptr=self.sparse_indptr,
            sparse_indices=self.sparse_indices,
            top_k=self.topk,
            group_size=self.group_size,
            num_pages_per_req=self.cache_manager.num_pages_per_req,
            is_bench=True
        )

        torch.cumsum(self.sparse_indptr, dim=0, dtype=torch.int32, out=self.sparse_indptr)

        self.sparse_decoder.run(
            q=self.q,
            paged_k_cache=self.cache_manager.paged_k_cache[self.layer_id],
            paged_v_cache=self.cache_manager.paged_v_cache[self.layer_id],
            sink_k_cache=self.cache_manager.sink_k_cache[self.layer_id],
            sink_v_cache=self.cache_manager.sink_v_cache[self.layer_id],
            kv_indptr=self.sparse_indptr,
            kv_indices=self.sparse_indices,
            kv_last_page_len=self.cache_manager.kv_last_page_len,
            out=self.output,
            kv_layout="HND",
        )

        self.sparse_indptr.zero_()
        self.sparse_indices.zero_()
        self.logits.zero_()
        self.mask_logits.zero_()
        # else:
        #     g.replay()

        return self.output
