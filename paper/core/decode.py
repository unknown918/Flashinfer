import torch
import math
from typing import Optional, Union
from flashinfer.decode import get_batch_decode_module
from flashinfer.utils import (
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
)


class AttentionSparseDecodeWrapper:
    def __init__(
            self,
            float_workspace_buffer: torch.Tensor,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        self._int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None

    def reset_workspace_buffer(
            self,
            float_workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._workspace_size = float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    def plan(
            self,
            kv_indptr: torch.Tensor,
            page_size: int,
            num_qo_heads: int,
            num_kv_heads: int,
            head_dim: int,
            pos_encoding_mode: str = "NONE",
            use_fp16_qk_reduction: bool = False,
            logits_soft_cap: Optional[float] = None,
            sm_scale: Optional[float] = None,
            rope_scale: Optional[float] = None,
            rope_theta: Optional[float] = None,
            non_blocking: bool = True,
            q_data_type: Union[str, torch.dtype] = "float16",
            kv_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        self._o_dtype = q_data_type
        batch_size = kv_indptr.shape[0] - 1
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)  # for split-kv binary search
        get_module_args = (
            q_data_type,
            kv_data_type,
            self._o_dtype,
            kv_indptr_host.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            False,  # use_sliding_window
            False,  # use_logits_soft_cap
        )

        self._cached_module = get_batch_decode_module(*get_module_args)

        args = [
            self._float_workspace_buffer,  # float_workspace_buffer
            self._int_workspace_buffer,  # int_workspace_buffer
            self._pin_memory_int_workspace_buffer,  # page_locked_int_workspace_buffer
            kv_indptr_host,  # kv_indptr
            batch_size,  # batch_size
            num_qo_heads // num_kv_heads,  # num_qo_heads
            1,  # num_kv_heads,
            page_size,  # page_size
            True,  # enable_cuda_graph
            -1,  # window_left
            0.0,  # logits_soft_cap
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            torch.empty(0, dtype=q_data_type),
            torch.empty(0, dtype=q_data_type),
        ]

        self.sink_indices = torch.arange(batch_size, dtype=torch.int32, device=self.device)
        self._plan_info = self._cached_module.plan(*args)
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._num_kv_heads = num_kv_heads
        self._gqa_group_size = num_qo_heads // num_kv_heads

    def run(
            self,
            q: torch.Tensor,
            paged_k_cache: torch.Tensor,
            paged_v_cache: torch.Tensor,
            sink_k_cache: torch.Tensor,
            sink_v_cache: torch.Tensor,
            kv_indptr: torch.Tensor,
            kv_indices: torch.Tensor,
            kv_last_page_len: torch.Tensor,
            out: torch.Tensor = None,
            kv_layout: str = "HND",
    ) -> torch.Tensor:
        _check_pos_encoding_mode(self._pos_encoding_mode)
        if self._logits_soft_cap is None:
            self._logits_soft_cap = 0.0
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(q.size(-1))
        if self._rope_scale is None:
            self._rope_scale = 1.0
        if self._rope_theta is None:
            self._rope_theta = 1e4

        import einops
        q = einops.rearrange(
            q,
            "batch_size (num_kv_heads gqa_group_size) head_dim -> (batch_size num_kv_heads) gqa_group_size head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()

        # NHD: [max_num_pages, page_size, num_kv_heads, head_dim]
        # HND: [max_num_pages, num_kv_heads, page_size, head_dim]
        paged_k_cache = einops.rearrange(
            paged_k_cache,
            "max_num_pages num_kv_heads page_size head_dim -> (max_num_pages num_kv_heads) 1 page_size head_dim",
        ).contiguous()
        paged_v_cache = einops.rearrange(
            paged_v_cache,
            "max_num_pages num_kv_heads page_size head_dim -> (max_num_pages num_kv_heads) 1 page_size head_dim",
        ).contiguous()

        sink_k_cache = einops.rearrange(
            sink_k_cache,
            "batch_size num_kv_heads sink_size head_dim -> (batch_size num_kv_heads) 1 sink_size head_dim",
        ).contiguous()
        sink_v_cache = einops.rearrange(
            sink_v_cache,
            "batch_size num_kv_heads sink_size head_dim -> (batch_size num_kv_heads) 1 sink_size head_dim",
        ).contiguous()

        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            paged_k_cache,
            paged_v_cache,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            sink_k_cache,
            sink_v_cache,
            self.sink_indices,
            out,
            None,  # lse
            TensorLayout[kv_layout].value,
            -1,  # window_left
            False,  # enable pdl
            None,  # alibi_slopes_buf
            self._logits_soft_cap,
            self._sm_scale,
            self._rope_scale,
            self._rope_theta,
        )

        return out


class AttentionBatchDecodeWrapper:
    def __init__(
            self,
            float_workspace_buffer: torch.Tensor,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        self._int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = False
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None

    def reset_workspace_buffer(
            self,
            float_workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._workspace_size = float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    def plan(
            self,
            kv_indptr: torch.Tensor,
            page_size: int,
            num_qo_heads: int,
            num_kv_heads: int,
            head_dim: int,
            pos_encoding_mode: str = "NONE",
            use_fp16_qk_reduction: bool = False,
            logits_soft_cap: Optional[float] = None,
            sm_scale: Optional[float] = None,
            rope_scale: Optional[float] = None,
            rope_theta: Optional[float] = None,
            non_blocking: bool = True,
            q_data_type: Union[str, torch.dtype] = "float16",
            kv_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        self._o_dtype = q_data_type
        batch_size = kv_indptr.shape[0] - 1
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)  # for split-kv binary search
        get_module_args = (
            q_data_type,
            kv_data_type,
            self._o_dtype,
            kv_indptr_host.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            False,  # use_sliding_window
            False,  # use_logits_soft_cap
        )

        self._cached_module = get_batch_decode_module(*get_module_args)

        args = [
            self._float_workspace_buffer,  # float_workspace_buffer
            self._int_workspace_buffer,  # int_workspace_buffer
            self._pin_memory_int_workspace_buffer,  # page_locked_int_workspace_buffer
            kv_indptr_host,  # kv_indptr
            batch_size,  # batch_size
            num_qo_heads,  # num_qo_heads
            num_kv_heads,  # num_kv_heads,
            page_size,  # page_size
            True,  # enable_cuda_graph
            -1,  # window_left
            0.0,  # logits_soft_cap
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            torch.empty(0, dtype=q_data_type),
            torch.empty(0, dtype=q_data_type),
        ]

        self.sink_indices = torch.arange(batch_size, dtype=torch.int32, device=self.device)
        self._plan_info = self._cached_module.plan(*args)
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._num_kv_heads = num_kv_heads
        self._gqa_group_size = num_qo_heads // num_kv_heads

    def run(
            self,
            q: torch.Tensor,
            paged_k_cache: torch.Tensor,
            paged_v_cache: torch.Tensor,
            sink_k_cache: torch.Tensor,
            sink_v_cache: torch.Tensor,
            kv_indptr: torch.Tensor,
            kv_indices: torch.Tensor,
            kv_last_page_len: torch.Tensor,
            out: torch.Tensor = None,
            kv_layout: str = "HND",
    ) -> torch.Tensor:
        _check_pos_encoding_mode(self._pos_encoding_mode)
        if self._logits_soft_cap is None:
            self._logits_soft_cap = 0.0
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(q.size(-1))
        if self._rope_scale is None:
            self._rope_scale = 1.0
        if self._rope_theta is None:
            self._rope_theta = 1e4

        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            paged_k_cache,
            paged_v_cache,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            sink_k_cache,
            sink_v_cache,
            self.sink_indices,
            out,
            None,  # lse
            TensorLayout[kv_layout].value,
            -1,  # window_left
            False,  # enable pdl
            None,  # alibi_slopes_buf
            self._logits_soft_cap,
            self._sm_scale,
            self._rope_scale,
            self._rope_theta,
        )

        return out
