import time
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from .models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="Llama-3.1-8B-Instruct")
parser.add_argument("--prompt-path", type=str, default="prompt.txt")
parser.add_argument("--budget", type=int, default=2048)
parser.add_argument("--seq-len", type=int, default=32000)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-dense-layers", type=int, default=2)
args = parser.parse_args()

model_path = args.model_path
prompt_path = args.prompt_path
budget = args.budget
seq_len = args.seq_len
batch_size = args.batch_size
num_dense_layers = args.num_dense_layers

device = "cuda:0"
sink_size = 4
page_size = 32
max_length = 32768
max_new_tokens = 2

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)

model.eval()

with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()

messages = [
    {
        "role": "user",
        "content": f"Please read this article, and summarize it briefly.{prompt}\n"
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)

assert seq_len <= input_ids.shape[1]
input_ids = input_ids[:, :seq_len]
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

    # warmup (cuda graph capture)
    for _ in range(3):
        _ = model.generate(input_ids, max_new_tokens=10)
        model.ss_flush()

    torch.cuda.synchronize()

    # TTFT
    model.ss_flush()
    torch.cuda.synchronize()
    t0 = time.time()
    print("Generation Started!")
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

print("\n===== STATS =====")
print(f"TPOT        : {tpot * 1e3:.3f} ms")
