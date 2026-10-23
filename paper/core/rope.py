import torch
import flashinfer
from typing import Tuple, Union


class RoPE:
    def __init__(
            self,
            batch_size: int,
            rope_scale: float,
            rope_theta: float,
            low_freq_factor: float,
            high_freq_factor: float,
            device: Union[str, torch.device],
    ):
        self.batch_size = batch_size
        self.rope_scale = rope_scale
        self.rope_theta = rope_theta
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.interleave = False
        self.device = device

        self.rope_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.rope_offset = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.rope_offset_helper = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)

    def update_index_prefill(self, seq_len):
        self.rope_indptr[1:] += seq_len
        torch.cumsum(self.rope_indptr, dim=0, out=self.rope_indptr)

    def update_index_decode(self):
        self.rope_offset += self.rope_indptr[1]
        self.rope_indptr[1:] = self.rope_offset_helper

    def apply_rope_llama(
            self,
            q: torch.Tensor,
            k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (q.dim() == 4 and k.dim() == 4)  # (batch-size, seq_len, num_heads, head_dim)

        flashinfer.rope.apply_llama31_rope_inplace(
            q=q.flatten(0, 1),
            k=k.flatten(0, 1),
            indptr=self.rope_indptr,
            offsets=self.rope_offset,
            interleave=self.interleave,
            rope_scale=self.rope_scale,
            rope_theta=self.rope_theta,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
        )

        return q, k

    def apply_rope_qwen(
            self,
            q: torch.Tensor,
            k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (q.dim() == 4 and k.dim() == 4)  # (batch-size, seq_len, num_heads, head_dim)

        flashinfer.rope.apply_rope_inplace(
            q=q.flatten(0, 1),
            k=k.flatten(0, 1),
            indptr=self.rope_indptr,
            offsets=self.rope_offset,
            rotary_dim=int(q.shape[-1]),
            interleave=False,
            rope_scale=1.0,
            rope_theta=1000000.0,
        )

        return q, k

    def flush(self):
        self.rope_indptr.zero_()
        self.rope_offset.zero_()
