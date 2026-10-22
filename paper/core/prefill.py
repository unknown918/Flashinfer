import torch
import flashinfer


class BatchPrefillWrapper:
    def __init__(
            self,
            head_dim: int,
            num_qo_heads: int,
            num_kv_heads: int,
            sink_size: int = 4,
            page_size: int = 32,
            kv_layout: str = "HND",
            dtype: torch.dtype = torch.bfloat16,
            device: str = "cuda:0",
    ) -> None:
        self.head_dim = head_dim
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.sink_size = sink_size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device

        work_space_bytes = 128 * 1024 * 1024
        sink_causal_workspace = torch.empty(work_space_bytes, dtype=torch.uint8, device=device)
        sink_noncausal_workspace = torch.empty(work_space_bytes, dtype=torch.uint8, device=device)
        page_workspace = torch.empty(work_space_bytes, dtype=torch.uint8, device=device)
        self.sink_causal_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            sink_causal_workspace,
            kv_layout=kv_layout
        )
        self.sink_noncausal_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            sink_noncausal_workspace,
            kv_layout=kv_layout,
        )
        self.page_causal_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            page_workspace,
            kv_layout=kv_layout,
        )

    def plan(
            self,
            qo_indices: torch.Tensor,
            kv_indices: torch.Tensor,
            kv_indptr: torch.Tensor,
            kv_last_page_len: torch.Tensor,
            sink_indices: torch.Tensor,
    ) -> None:
        self.q_lens = qo_indices.tolist()
        batch_size = qo_indices.numel()
        sink_qo_indptr = torch.arange(
            batch_size + 1, dtype=torch.int32, device=self.device
        ) * self.sink_size
        page_qo_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                (qo_indices - self.sink_size).cumsum(0, dtype=torch.int32),
            ]
        )
        sink_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
        sink_last_page_len = torch.full(
            (batch_size,), self.sink_size, dtype=torch.int32, device=self.device
        )

        self.sink_causal_wrapper.plan(
            qo_indptr=sink_qo_indptr,
            paged_kv_indptr=sink_indptr,
            paged_kv_indices=sink_indices,
            paged_kv_last_page_len=sink_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.sink_size,
            causal=True,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

        self.sink_noncausal_wrapper.plan(
            qo_indptr=page_qo_indptr,
            paged_kv_indptr=sink_indptr,
            paged_kv_indices=sink_indices,
            paged_kv_last_page_len=sink_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.sink_size,
            causal=False,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

        self.page_causal_wrapper.plan(
            qo_indptr=page_qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=True,
            pos_encoding_mode="NONE",
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

        total_q = sum(self.q_lens)
        sink_q_len = batch_size * self.sink_size
        page_q_len = total_q - sink_q_len
        shape_sink = (sink_q_len, self.num_qo_heads, self.head_dim)
        shape_page = (page_q_len, self.num_qo_heads, self.head_dim)
        self.q_sink = torch.zeros(shape_sink, dtype=self.dtype, device=self.device)
        self.q_page = torch.zeros(shape_page, dtype=self.dtype, device=self.device)
        self.sink_output = torch.zeros_like(self.q_sink)
        self.sink_v = torch.zeros_like(self.q_page)
        self.page_v = torch.zeros_like(self.q_page)
        self.sink_s = torch.zeros(page_q_len, self.num_qo_heads, dtype=torch.float32, device=self.device)
        self.page_s = torch.zeros_like(self.sink_s)
        self.output = torch.zeros(total_q, self.num_qo_heads, self.head_dim, dtype=self.dtype, device=self.device)

        self.slices = []
        q_offset = page_offset = 0
        for i, q_len in enumerate(self.q_lens):
            page_len = q_len - self.sink_size
            self.slices.append((i, q_offset, page_offset, q_len, page_len))
            q_offset += q_len
            page_offset += page_len

    def split_q_(self, q: torch.Tensor):
        for i, q_offset, page_offset, q_len, page_len in self.slices:
            self.q_sink[i * self.sink_size:(i + 1) * self.sink_size].copy_(
                q[q_offset:q_offset + self.sink_size]
            )
            self.q_page[page_offset:page_offset + page_len].copy_(
                q[q_offset + self.sink_size:q_offset + q_len]
            )

    def run(
            self,
            q: torch.Tensor,
            sink_k_cache: torch.Tensor,
            sink_v_cache: torch.Tensor,
            paged_k_cache: torch.Tensor,
            paged_v_cache: torch.Tensor,
    ) -> torch.Tensor:
        self.split_q_(q)
        self.sink_causal_wrapper.run(
            self.q_sink,
            (sink_k_cache, sink_v_cache),
            out=self.sink_output,
        )

        self.sink_noncausal_wrapper.run(
            self.q_page,
            (sink_k_cache, sink_v_cache),
            out=self.sink_v,
            lse=self.sink_s,
            return_lse=True,
        )

        self.page_causal_wrapper.run(
            self.q_page,
            (paged_k_cache, paged_v_cache),
            out=self.page_v,
            lse=self.page_s,
            return_lse=True,
        )

        flashinfer.merge_state_in_place(
            self.sink_v,
            self.sink_s,
            self.page_v,
            self.page_s
        )

        for i, q_offset, page_offset, q_len, page_len in self.slices:
            self.output[q_offset:q_offset + self.sink_size].copy_(
                self.sink_output[i * self.sink_size:(i + 1) * self.sink_size]
            )
            self.output[q_offset + self.sink_size:q_offset + q_len].copy_(
                self.sink_v[page_offset:page_offset + page_len]
            )

        return self.output
