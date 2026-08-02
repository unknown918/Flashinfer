import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

device = "cuda:0"
batch_size = 1
sink_size = 4
page_size = 32
seq_len = 4096
budget = 2048
max_length = 32768
max_new_tokens = 2
prompt_path = "./prompt.txt"
model_id = "/home1/csu_yqj/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

model.eval()

with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()

messages = [
    {
        "role": "user",
        "content": f"Please read this article, and summarize it briefly.\n{prompt}\n"
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)[:, :seq_len].to(device)

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
        dtype=torch.bfloat16,
    )

    # warmup (cuda graph capture)
    for _ in range(3):
        _ = model.generate(input_ids, max_new_tokens=10)
        model.ss_flush()

    torch.cuda.synchronize()

    # TTFT
    model.ss_flush()
    torch.cuda.synchronize()
    t0 = time.time()

    _ = model.generate(
        input_ids,
        max_new_tokens=1,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.eos_token_id
    )

    torch.cuda.synchronize()
    ttft = time.time() - t0

    # full generation
    model.ss_flush()
    torch.cuda.synchronize()
    t1 = time.time()

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        eos_token_id=None,
        pad_token_id=tokenizer.eos_token_id
    )

    torch.cuda.synchronize()
    total_time = time.time() - t1

    tpot = (total_time - ttft) / (max_new_tokens - 1)

    model.ss_flush()

new_tokens = outputs[0][seq_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n===== OUTPUT =====\n")
print(response)

print("\n===== STATS =====\n")
print(f"Seq Len     : {seq_len}")
print(f"Budget      : {budget}")
print(f"TTFT        : {ttft * 1e3:.3f} ms")
print(f"TPOT        : {tpot * 1e3:.3f} ms")
