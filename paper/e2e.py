import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

device = torch.device("cuda")
model_id = "../Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)

model.eval()
with open("./prompt.txt", 'r', encoding='utf-8') as f:
    prompt = f.read()

messages = [
    {
        "role": "user",
        "content": f"{prompt}{prompt}\nRead this paper carefully, and give me a brief summary.\n"
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

seq_len = 8192
budget = 4096
model.ss_init(
    device="cuda",
    budget=budget,
    page_size=32,
    max_length=32768,
    dtype=torch.float16,
)

assert seq_len <= input_ids.shape[1]

# warm up
for _ in range(3):
    _ = model.generate(input_ids, max_new_tokens=2) # capture cuda graph
    model.ss_flush()
torch.cuda.synchronize()

model.ss_flush()

torch.cuda.synchronize()
t0 = time.time()
_ = model.generate(
    input_ids[:, -8192:],
    max_new_tokens=1,
    do_sample=False,
    temperature=0,
    pad_token_id=tokenizer.eos_token_id
)
torch.cuda.synchronize()
ttft = time.time() - t0

model.ss_flush()

torch.cuda.synchronize()
t1 = time.time()
outputs = model.generate(
    input_ids[:, -8192:],
    max_new_tokens=10,
    do_sample=False,
    temperature=0,
    pad_token_id=tokenizer.eos_token_id
)
torch.cuda.synchronize()
total_time = time.time() - t1
tpot = (total_time - ttft) / 10

model.ss_flush()

new_tokens = outputs[0][seq_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
print(f"Seq Len: {seq_len}")
print(f"Budget: {budget}")
print(f"TTFT: {ttft}")
print(f"TPOT: {tpot}")

# sparse
# Seq Len: 8192
# Budget: 4096
# TTFT: 2.154984712600708
# TPOT: 0.0330620527267456

# dense
# Seq Len: 8192
# TTFT: 2.2001969814300537
# TPOT: 0.0335003137588501
