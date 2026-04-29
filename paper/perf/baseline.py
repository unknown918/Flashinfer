import time
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

warm_up = 2
iteration = 5
decode_len = 256
context_len = 16384
model_id = "../../Llama-3.1-8B-Instruct"

dtype = torch.float16
device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    trust_remote_code=True,
).to(device)
model.eval()

prefill_latency = []
decode_latency = []

with torch.inference_mode():
    for i in tqdm(range(warm_up + iteration)):
        torch.cuda.empty_cache()
        input_ids = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(1, context_len),
            device=device
        )
        torch.cuda.synchronize()
        ts = time.perf_counter()

        outputs = model(
            input_ids=input_ids,
            use_cache=True
        )

        torch.cuda.synchronize()
        te = time.perf_counter()

        prefill_latency.append(te - ts)

        past_key_values = outputs.past_key_values
        next_token = input_ids[:, -1:]

        for _ in range(decode_len):

            torch.cuda.synchronize()
            ts = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )

            torch.cuda.synchronize()
            te = time.perf_counter()

            decode_latency.append(te - ts)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

avg_prefill_latency = np.mean(prefill_latency[warm_up:])
avg_decode_latency = np.mean(decode_latency[warm_up:])

print("context_len,decode_len,avg_prefill_latency(s),avg_decode_latency(s)")
print(f"{context_len},{decode_len},{avg_prefill_latency:.6f},{avg_decode_latency:.6f}")
