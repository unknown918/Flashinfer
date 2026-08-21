import torch
import os
import flashinfer
from flash_attn import flash_attn_func
from .utils import ceil_div
from .rope import RoPE
from .cache import PagedKVCache
from .decode import BatchDecodeWrapper, SparseDecodeWrapper
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
            num_dense_layers: int = 2,
            topk: int = 16,
            dtype: torch.dtype = torch.bfloat16,
            device: Union[str, torch.device] = "cuda:0",
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
        self.group_size = num_qo_heads // num_kv_heads
        num_pages = ceil_div(max_length, page_size)

        # sparse attention buffers (one for all layers)
        self.sparse_indices = torch.zeros(
            batch_size * num_kv_heads * topk * self.group_size,
            dtype=torch.int32,
            device=device
        )
        self.sparse_indptr = torch.zeros(1 + batch_size * num_kv_heads, dtype=torch.int32, device=device)
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
            for _ in range(num_layers - 2)
        ]
        self.graph_captured = [False] * (num_layers - 2)
        self.proxy_plan()

    def proxy_plan(self):
        dense_proxy_indptr = torch.full(
            (self.batch_size + 1,), self.topk,
            dtype=torch.int32,
            device=self.device
        )
        dense_proxy_indptr[0] = 0
        torch.cumsum(dense_proxy_indptr, dim=0, out=dense_proxy_indptr)

        sparse_proxy_indptr = torch.full(
            (self.batch_size * self.cache_manager.num_kv_heads + 1,), self.topk,
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
    ) -> torch.Tensor:
        assert (q.dim() == 4 and k.dim() == 4 and v.dim() == 4)  # (batch_size, seq_len, num_heads, head_dim)

        if self.layer_id == 0:
            self.rope.update_index_prefill(q.shape[1])
            self.cache_manager.seq_lens.fill_(q.shape[1])
            self.cache_manager.update_metadata_prefill()

        q, k = self.rope.apply_rope(q, k)
        self.cache_manager.append_page_prefill(self.layer_id, k, v)

        # no sparsity for prefill
        return flash_attn_func(q=q, k=k, v=v, causal=True)

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
            self.kv_last_page_len.view(
                self.batch_size,
                self.cache_manager.num_kv_heads,
            ).copy_(self.cache_manager.kv_last_page_len[:, None])
            self.num_pages_per_req.copy_(self.cache_manager.num_pages_per_req)
            self.num_pages_per_req.sub_(1)  # last page is manually selected

        q, k = self.rope.apply_rope(q, k)
        self.cache_manager.append_page_decode(self.layer_id, k, v)
        self.q.copy_(q[:, 0, :, :])

        if self.layer_id < self.num_dense_layers:  # no sparsity for the first two layers
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

        # g = self.graphs[self.layer_id - self.num_dense_layers]
        # if not self.graph_captured[self.layer_id - self.num_dense_layers]:
        #     self.graph_captured[self.layer_id - self.num_dense_layers] = True
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
            num_pages_per_req=self.num_pages_per_req,
        )

        self.sparse_indptr[1:].add_(1)
        torch.cumsum(self.sparse_indptr, dim=0, dtype=torch.int32, out=self.sparse_indptr)

        self.sparse_decoder.run(
            q=self.q,
            paged_k_cache=self.cache_manager.paged_k_cache[self.layer_id],
            paged_v_cache=self.cache_manager.paged_v_cache[self.layer_id],
            sink_k_cache=self.cache_manager.sink_k_cache[self.layer_id],
            sink_v_cache=self.cache_manager.sink_v_cache[self.layer_id],
            kv_indptr=self.sparse_indptr,
            kv_indices=self.sparse_indices,
            kv_last_page_len=self.kv_last_page_len,
            out=self.output,
            kv_layout="HND",
        )

        os.makedirs(f"paper/data/{self.batch_size}/", exist_ok=True)
        os.makedirs(f"paper/data/{self.batch_size}/{self.topk * self.page_size}/", exist_ok=True)
        os.makedirs(f"paper/data/{self.batch_size}/{self.topk * self.page_size}/", exist_ok=True)

        torch.save(
            self.sparse_indptr.detach().cpu(),
            f"paper/data/{self.batch_size}/{self.topk * self.page_size}/sparse_indptr_{self.layer_id}.pt"
        )
        torch.save(
            self.sparse_indices.detach().cpu(),
            f"paper/data/{self.batch_size}/{self.topk * self.page_size}/sparse_indices_{self.layer_id}.pt"
        )
        torch.save(
            self.kv_last_page_len.detach().cpu(),
            f"paper/data/{self.batch_size}/{self.topk * self.page_size}/kv_last_page_len_{self.layer_id}.pt"
        )

        self.sparse_indptr.zero_()
        self.sparse_indices.zero_()
        self.logits.zero_()
        self.mask_logits.zero_()
        # else:
        #     g.replay()

        return self.output
