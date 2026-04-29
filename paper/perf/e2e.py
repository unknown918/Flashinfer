import time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# import transformers
# from paper.models.llama import LlamaForCausalLM
# transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

warm_up = 2
iteration = 5
decode_len = 256
budget = 4096
page_size = 32
context_len = 8192
max_length = 32768

dtype = torch.float16
device = torch.device("cuda:0")
model_id = "../../Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

model.eval()
# model.ss_init(
#     device="cuda:0",
#     budget=budget,
#     page_size=page_size,
#     max_length=max_length,
#     dtype=torch.float16,
# )
hidden_size = model.config.hidden_size

prefill_latency = []
decode_latency = []

with torch.inference_mode():
    for _ in tqdm(range(warm_up + iteration)):
        # model.ss_flush() # flush cache
        torch.cuda.empty_cache()

        # Prefill Stage
        ts = time.perf_counter()
        hidden_states = torch.randn(1, context_len, hidden_size, dtype=dtype, device=device)
        model(inputs_embeds=hidden_states)
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        # Start decoding decode_len tokens
        for _ in range(decode_len):
            ts = time.perf_counter()
            hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
            model(inputs_embeds=hidden_states)
            te = time.perf_counter()
            decode_latency.append(te - ts)

avg_prefill_latency = np.mean(prefill_latency[warm_up:])
avg_decode_latency = np.mean(decode_latency[warm_up:])

print("page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency")
print(f"{page_size},{budget},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}")

# sparse
# 32,4096,8192,256,2.3654978952370582,0.06178360435300033

# dense
# 32,4096,8192,256,2.4434303576126695,0.05056821436090551
