import transformers
import torch

# TODO: Fix shape !!!
# TODO: Fix hard-code in topk and sink

# patch llama
# from .llama import LlamaForCausalLM
#
# transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

device = "cuda:2"

model_id = "../Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.bfloat16},
    device=device,
)

# input without transpose: torch.Size([1, 39, 32, 128]) # [num_page, page_size, num_heads, head_dim], NHD
# output shape:  torch.Size([1, 39, 4096])

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
