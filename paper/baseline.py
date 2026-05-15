import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda")
model_id = "../Llama-3.1-8B-Instruct"

seq_len = 8192
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to(device)

model.eval()
with open("./prompt.txt", 'r', encoding='utf-8') as f:
    prompt = f.read()

messages = [
    {
        "role": "user",
        "content": f"{prompt}{prompt}\nPlease, can you read this paper carefully, and give me a brief summary.\n"
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

assert seq_len <= input_ids.shape[1]

# warm up
for _ in range(3):
    _ = model.generate(input_ids, max_new_tokens=1)
torch.cuda.synchronize()

torch.cuda.synchronize()
t0 = time.time()
_ = model.generate(
    input_ids[:, -seq_len:],
    max_new_tokens=1,
    do_sample=False,
    temperature=0,
    pad_token_id=tokenizer.eos_token_id
)
torch.cuda.synchronize()
ttft = time.time() - t0

torch.cuda.synchronize()
t1 = time.time()
outputs = model.generate(
    input_ids[:, -seq_len:],
    max_new_tokens=100,
    do_sample=False,
    temperature=0,
    pad_token_id=tokenizer.eos_token_id
)
torch.cuda.synchronize()
total_time = time.time() - t1
tpot = (total_time - ttft) / 100

new_tokens = outputs[0][seq_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
print(f"Seq Len: {seq_len}")
print(f"TTFT: {ttft * 1e3:3f} ms")
print(f"TPOT: {tpot * 1e3:3f} ms")
