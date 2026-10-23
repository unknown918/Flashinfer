import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from .models.llama import LlamaForCausalLM
from .models.qwen import Qwen3ForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
transformers.models.qwen3.Qwen3ForCausalLM = Qwen3ForCausalLM

model_path = "/hpc2hdd/home/qyang284/workspace/models/Llama-3.1-8B"
prompt_path = "paper/prompt.txt"

device = "cuda:0"
batch_size = 4
sink_size = 4
page_size = 32
budget = 2048
num_dense_layers = 2
seq_len = 8000
max_length = 32768
max_new_tokens = 100

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

model.eval()

with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()

article_ids = tokenizer(
    prompt,
    add_special_tokens=False,
    return_tensors="pt",
).input_ids[0]
article_ids = article_ids[:seq_len]
article = tokenizer.decode(article_ids, skip_special_tokens=True)
messages = [
    {
        "role": "user",
        "content": f"Please read this article, and summarize it briefly.\n\n{article}"
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
).to(device)

input_ids = input_ids.repeat(batch_size, 1).to(device)

with torch.inference_mode():
    # init system
    model.ss_init(
        device=device,
        budget=budget,
        batch_size=batch_size,
        sink_size=sink_size,
        page_size=page_size,
        max_length=max_length,
        num_dense_layers=num_dense_layers,
        dtype=torch.bfloat16,
    )

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        eos_token_id=None,
        pad_token_id=tokenizer.eos_token_id
    )

    model.ss_flush()

new_tokens = outputs[-1][input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"Response:\n {response}")
