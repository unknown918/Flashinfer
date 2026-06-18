import time
import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="./Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt_path", type=str, default="./paper/prompt.txt")

    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)

    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--page_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=131072)

    return parser.parse_args()


args = parse_args()
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
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

with torch.inference_mode():
    # init system
    model.ss_init(
        device=device,
        budget=args.budget,
        page_size=args.page_size,
        max_length=args.max_length,
        dtype=torch.float16,
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
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0,
        eos_token_id=None,
        pad_token_id=tokenizer.eos_token_id
    )

    torch.cuda.synchronize()
    total_time = time.time() - t1

    tpot = (total_time - ttft) / (args.max_new_tokens - 1)

    model.ss_flush()

new_tokens = outputs[0][seq_len:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n===== OUTPUT =====\n")
print(response)

print("\n===== STATS =====")
print(f"Seq Len     : {args.seq_len}")
print(f"Budget      : {args.budget}")
print(f"TTFT        : {ttft * 1e3:.3f} ms")
print(f"TPOT        : {tpot * 1e3:.3f} ms")
