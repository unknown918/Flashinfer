import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="./Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt_path", type=str, default="./paper/prompt.txt")

    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=100)

    return parser.parse_args()


args = parse_args()
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to(device)

model.eval()

with open(args.prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()

messages = [
    {
        "role": "user",
        "content": f"Please read this article, and tell me what is CuTe Layout.\n{prompt}\n"
    },
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)[:, :args.seq_len].to(device)

seq_len = input_ids.shape[1]

assert seq_len <= input_ids.shape[1]

with torch.inference_mode():
    for _ in range(3):
        _ = model.generate(input_ids, max_new_tokens=2)
    torch.cuda.synchronize()

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

    torch.cuda.synchronize()
    t1 = time.time()

    outputs = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.eos_token_id
    )

    torch.cuda.synchronize()
    total_time = time.time() - t1

    tpot = (total_time - ttft) /(args.max_new_tokens - 1)

new_tokens = outputs[0][seq_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n===== OUTPUT =====\n")
print(response)

print("\n===== STATS =====")
print(f"Seq Len     : {seq_len}")
print(f"TTFT        : {ttft * 1e3:.3f} ms")
print(f"TPOT        : {tpot * 1e3:.3f} ms")
