import torch
from torch import nn
from typing import Callable, Optional

import flashinfer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.llama.configuration_llama import LlamaConfig

import transformers.models.llama.modeling_llama as llama_modeling
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

device = "cuda:2"


class SparseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.seq_len = 0
        self.sink = 4
        self.topk = 16
        self.page_size = 32
        self.max_length = 1024
        self.num_kv_heads = config.num_key_value_heads
        self.num_qo_heads = config.num_attention_heads

        assert config.num_attention_heads % config.num_key_value_heads == 0
        self.num_key_value_groups = int(config.num_attention_heads // config.num_key_value_heads)
        self.num_pages = self.max_length // self.page_size

        paged_k_cache = torch.zeros(
            self.num_kv_heads, self.num_pages, self.page_size, self.head_dim
        ).half().to(device).contiguous()
        paged_v_cache = torch.zeros(
            self.num_kv_heads, self.num_pages, self.page_size, self.head_dim
        ).half().to(device).contiguous()

        self.paged_kv_cache = (paged_k_cache, paged_v_cache)

        self.pooling = torch.empty(
            self.num_pages, self.num_kv_heads, self.head_dim
        ).half().to(device).contiguous()
        self.logits = torch.empty(
            self.num_qo_heads, self.num_pages
        ).half().to(device).contiguous()

        self.indptr = torch.zeros(
            1 + self.num_kv_heads,
            dtype=torch.uint32,
            device=device
        )
        self.indices = torch.zeros(
            self.num_kv_heads * (self.sink + self.topk * self.num_key_value_groups * self.page_size),
            dtype=torch.uint32,
            device=device
        )
        self.mask_logits = torch.zeros(
            self.num_kv_heads, self.num_pages,
            dtype=torch.uint32,
            device=device
        )
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.decode_wrapper = flashinfer.SparseSinkAttentionWrapper(workspace_buffer)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        assert input_shape.size(0) == 1, "Only support single-batch inference"

        seq_len = input_shape.size(2)
        if self.layer_idx == 0:
            self.seq_len += seq_len

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # [batch_size, num_heads, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        current_pages = (seq_len + self.page_size - 1) // self.page_size
        if seq_len % self.page_size == 0:
            last_page_len = self.page_size
        else:
            last_page_len = seq_len % self.page_size

        kv_page_indptr = torch.cat(
            [torch.zeros(1).int().to(0), torch.tensor([current_pages]).int().to(0)], dim=0
        ).int()
        kv_page_indices = torch.arange(current_pages, dtype=torch.int32, device="cuda:0")
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda:0")

        if seq_len > 1:
            flashinfer.append_paged_kv_cache_prefill(
                key_states[0],
                value_states[0],
                self.paged_kv_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
                self.pooling,
                kv_layout="HNND"
            )
            attn_output = flashinfer.single_prefill_with_kv_cache(
                q=query_states[0],
                k=key_states[0],
                v=value_states[0],
                causal=True,
                return_lse=False
            )
        else:
            flashinfer.append_paged_kv_cache_decode(
                key_states[0],
                query_states[0],
                self.paged_kv_cache,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len,
                self.pooling,
                kv_layout="HNND"
            )

            flashinfer.decode.estimate(
                query_states[0],
                seq_len=self.seq_len,
                out=self.logits,
                pooling=self.pooling
            )

            flashinfer.topk_bool_mask_logits(
                k=self.topk,
                input=self.pooling,
                indptr=self.indptr,
                indices=self.indices,
                mask_logits=self.mask_logits,
                max_length=self.num_pages,
                group_size=self.num_key_value_groups,
                block_size=self.page_size
            )

            self.indptr.zero_()
            self.zero_()
            self.mask_logits.zero_()

            attn_output = self.decode_wrapper.run(
                q=query_states,
                k=self.paged_kv_cache[0],
                v=self.paged_kv_cache[1],
                return_lse=False
            )

        return attn_output, None


llama_modeling.LlamaAttention = SparseAttention

model_id = "../../Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.float16},
    device=device,
)

messages = [
    {
        "role": "user",
        "content": "Who are you?"
    },
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
