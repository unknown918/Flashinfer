import time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

warm_up = 2
iteration = 5
decode_len = 256
budget = 1024
page_size = 32
context_len = 16384
max_length = 32768

dtype = torch.float16
device = torch.device("cuda:0")
model_id = "../../Llama-3.1-8B-Instruct"

import transformers
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

hidden_size = model.config.hidden_size

prefill_latency = []
decode_latency = []

model.eval()
with torch.inference_mode():
    model.ss_init(
        device="cuda:0",
        budget=budget,
        page_size=page_size,
        max_length=max_length,
        dtype=torch.float16,
    )
    for _ in tqdm(range(warm_up + iteration)):
        model.ss_flush()  # flush cache

        # Prefill Stage
        hidden_states = torch.randn(1, context_len, hidden_size, dtype=dtype, device=device)
        torch.cuda.synchronize()
        ts = time.perf_counter()
        model(inputs_embeds=hidden_states)
        torch.cuda.synchronize()
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        # Start decoding decode_len tokens
        for _ in range(decode_len):
            hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            model(inputs_embeds=hidden_states)
            torch.cuda.synchronize()
            te = time.perf_counter()
            decode_latency.append(te - ts)

avg_prefill_latency = np.mean(prefill_latency[warm_up:])
avg_decode_latency = np.mean(decode_latency[warm_up:])

print("page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency")
print(f"{page_size},{budget},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}")

# page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency
# 32,4096,8192,256,2.3687527952715755,0.05941147545154867
# 32,512,8192,256,2.34726024242118,0.05964126595863084
