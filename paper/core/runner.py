import torch
import flashinfer
from typing import Union
from .utils import ceil_div, make_streaming_metadata
from .rope import RoPE
from .cache import PagedKVCache
from .decode import BatchDecodeWrapper, SparseDecodeWrapper
from flash_attn import flash_attn_func


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
            num_dense_layers: int = 2,
            topk: int = 16,
            dtype: torch.dtype = torch.bfloat16,
            device: Union[str, torch.device] = "cuda:0",
            streaming: bool = False,
    ):
        self.dtype = dtype
        self.batch_size = batch_size
        self.device = device
        self.layer_id = 0
        self.topk = topk
        self.sink_size = sink_size
        self.page_size = page_size
        self.max_length = max_length
        self.num_dense_layers = num_dense_layers
        self.streaming = streaming
        self.group_size = num_qo_heads // num_kv_heads
        num_pages = ceil_div(max_length, page_size)

        # sparse attention buffers (one for all layers)
        self.sparse_indices = torch.zeros(
            batch_size * num_kv_heads * topk * self.group_size,
            dtype=torch.int32,
            device=device
        )
        self.sparse_indptr = torch.zeros(1 + batch_size * num_kv_heads, dtype=torch.int32, device=device)
        self.streaming_indices = torch.zeros(batch_size * topk, dtype=torch.int32, device=device)
        self.streaming_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.mask_logits = torch.zeros(batch_size * num_kv_heads, num_pages, dtype=torch.int32, device=device)
        self.num_pages_per_req = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.kv_last_page_len = torch.zeros(batch_size * num_kv_heads, dtype=torch.int32, device=device)
        self.logits = torch.zeros(batch_size * num_qo_heads, num_pages, dtype=dtype, device=device).contiguous()
        self.output = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device).contiguous()

        # decode wrapper and buffers
        workspace_buffer_dense = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
        workspace_buffer_sparse = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device).contiguous()
        self.dense_decoder = BatchDecodeWrapper(float_workspace_buffer=workspace_buffer_dense)
        self.sparse_decoder = SparseDecodeWrapper(float_workspace_buffer=workspace_buffer_sparse)

        # kv cache manager
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
            device=device,
        )

        # cuda graph buffers
        self.q = torch.zeros(batch_size, num_qo_heads, head_dim, dtype=dtype, device=self.device).contiguous()
        self.graphs = [
            torch.cuda.CUDAGraph()
            for _ in range(num_layers - num_dense_layers)
        ]
        self.graph_captured = [False] * (num_layers - num_dense_layers)
        self.proxy_plan()

    def prepare_decode(self):
        if self.cache_manager.prefill_offload and self.cache_manager.paged_k_cache.device.type == "cpu":
            self.cache_manager.restore()
        if not self.streaming and not all(self.graph_captured):
            for layer in range(self.num_dense_layers, self.cache_manager.pooling.shape[0]):
                self.capture_sparse_graph(layer)

    def capture_sparse_graph(self, layer):
        index = layer - self.num_dense_layers
        if self.graph_captured[index]:
            return
        with torch.cuda.graph(self.graphs[index]):
            flashinfer.estimate(
                query=self.q,
                pooling=self.cache_manager.pooling[layer],
                kv_indptr=self.cache_manager.kv_indptr,
                kv_indices=self.cache_manager.kv_indices,
                out=self.logits,
            )
            flashinfer.topk_bool_mask_logits(
                logits=self.logits,
                mask_logits=self.mask_logits,
                indptr=self.cache_manager.kv_indptr,
                indices=self.cache_manager.kv_indices,
                sparse_indptr=self.sparse_indptr,
                sparse_indices=self.sparse_indices,
                top_k=self.topk,
                group_size=self.group_size,
                num_pages_per_req=self.num_pages_per_req,
            )
            self.sparse_indptr[1:].add_(1)
            torch.cumsum(self.sparse_indptr, dim=0, dtype=torch.int32, out=self.sparse_indptr)
            self.sparse_decoder.run(
                q=self.q,
                paged_k_cache=self.cache_manager.paged_k_cache[layer],
                paged_v_cache=self.cache_manager.paged_v_cache[layer],
                sink_k_cache=self.cache_manager.sink_k_cache[layer],
                sink_v_cache=self.cache_manager.sink_v_cache[layer],
                kv_indptr=self.sparse_indptr,
                kv_indices=self.sparse_indices,
                kv_last_page_len=self.kv_last_page_len,
                out=self.output,
                kv_layout="HND",
            )
            self.sparse_indptr.zero_()
            self.sparse_indices.zero_()
            self.logits.zero_()
            self.mask_logits.zero_()
        self.graph_captured[index] = True

    def proxy_plan(self):
        dense_proxy_indptr = torch.full(
            (self.batch_size + 1,), self.topk,
            dtype=torch.int32,
            device=self.device
        )
        dense_proxy_indptr[0] = 0
        torch.cumsum(dense_proxy_indptr, dim=0, out=dense_proxy_indptr)

        sparse_proxy_indptr = torch.full(
            (self.batch_size * self.cache_manager.num_kv_heads + 1,), self.topk * self.page_size,
            dtype=torch.int32,
            device=self.device
        )
        sparse_proxy_indptr[0] = 0
        torch.cumsum(sparse_proxy_indptr, dim=0, out=sparse_proxy_indptr)

        self.dense_decoder.plan(
            kv_indptr=dense_proxy_indptr,
            page_size=self.page_size,
            num_qo_heads=self.cache_manager.num_qo_heads,
            num_kv_heads=self.cache_manager.num_kv_heads,
            head_dim=self.cache_manager.head_dim,
            q_data_type=self.dtype,
            kv_data_type=self.dtype
        )

        self.sparse_decoder.plan(
            kv_indptr=sparse_proxy_indptr,
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
            is_qwen: bool = False,
    ) -> torch.Tensor:
        assert (q.dim() == 4 and k.dim() == 4 and v.dim() == 4)  # (batch_size, seq_len, num_heads, head_dim)

        if self.layer_id == 0:
            if self.cache_manager.prefill_offload:
                torch.cuda.synchronize(self.device)
                for graph in self.graphs:
                    graph.reset()
                self.graph_captured = [False] * len(self.graphs)
                self.cache_manager.begin_prefill()
            self.rope.update_index_prefill(q.shape[1])
            self.cache_manager.seq_lens.fill_(q.shape[1])
            self.cache_manager.update_metadata_prefill()

        if is_qwen:
            q, k = self.rope.apply_rope_qwen(q, k)
        else:
            q, k = self.rope.apply_rope_llama(q, k)

        self.cache_manager.append_page_prefill(self.layer_id, k, v)

        # no sparsity for prefill
        return flash_attn_func(
            q=q,
            k=k,
            v=v,
            causal=True
        )

    def decode(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            is_qwen: bool = False,
    ) -> torch.Tensor:
        assert (q.dim() == 4 and k.dim() == 4 and v.dim() == 4)  # (batch_size, seq_len, num_heads, head_dim)
        assert (q.shape[1] == 1 and k.shape[1] == 1 and v.shape[1] == 1)

        if self.layer_id == 0:
            self.prepare_decode()
            self.cache_manager.seq_lens += 1
            self.rope.update_index_decode()
            self.cache_manager.update_metadata_decode()
            if self.streaming:
                make_streaming_metadata(
                    self.cache_manager.num_pages_per_req,
                    self.streaming_indptr,
                    self.streaming_indices,
                    self.topk,
                    self.cache_manager.max_num_pages // self.batch_size,
                )
            else:
                self.kv_last_page_len.view(self.batch_size, -1).copy_(
                    self.cache_manager.kv_last_page_len[:, None]
                )
                self.num_pages_per_req.copy_(self.cache_manager.num_pages_per_req).sub_(1)

        if is_qwen:
            q, k = self.rope.apply_rope_qwen(q, k)
        else:
            q, k = self.rope.apply_rope_llama(q, k)

        self.cache_manager.append_page_decode(self.layer_id, k, v)
        self.q.copy_(q[:, 0, :, :])

        dense = self.layer_id < self.num_dense_layers
        if dense or self.streaming:
            self.dense_decoder.run(
                q=self.q,
                paged_k_cache=self.cache_manager.paged_k_cache[self.layer_id],
                paged_v_cache=self.cache_manager.paged_v_cache[self.layer_id],
                sink_k_cache=self.cache_manager.sink_k_cache[self.layer_id],
                sink_v_cache=self.cache_manager.sink_v_cache[self.layer_id],
                kv_indptr=self.cache_manager.kv_indptr if dense else self.streaming_indptr,
                kv_indices=self.cache_manager.kv_indices if dense else self.streaming_indices,
                kv_last_page_len=self.cache_manager.kv_last_page_len,
                out=self.output,
                kv_layout="HND",
            )
        else:
            self.graphs[self.layer_id - self.num_dense_layers].replay()

        return self.output
