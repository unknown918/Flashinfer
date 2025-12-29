"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
from types import SimpleNamespace
from typing import Optional, Union
import torch

from .jit.topk import gen_topk_module
from .utils import register_custom_op, _get_cache_buf


def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)


@functools.cache
def get_topk_module():
    module = gen_topk_module().build_and_load()

    @register_custom_op("flashinfer::top_k_mask_logits", mutates_args=())
    def radix_top_k_mask_logits(
            logits: torch.Tensor,
            maybe_top_k_arr: Optional[torch.Tensor],
            top_k_val: int,
    ) -> torch.Tensor:
        logits = logits.float()
        mask_logits = torch.zeros_like(logits, dtype=torch.uint8)

        # void radix_topk_mask_logits(TensorView logits, TensorView masked_logits,
        # TensorView row_states_buffer, int64_t top_k)
        row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
            f"radix_topk_row_states_{logits.device}",
            1024 * 1024,  # 1MB
            logits.device
        )

        module.radix_topk_mask_logits(
            logits,
            mask_logits,
            row_states_buffer,
            top_k_val,
        )
        return mask_logits.to(bool)

    # Register the module
    return SimpleNamespace(top_k_mask=radix_top_k_mask_logits)


def radix_top_k_mask_logits(
        logits: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    return get_topk_module().top_k_mask(
        logits, *_to_tensor_scalar_tuple(top_k)
    )
