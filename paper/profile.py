import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from paper.models.llama import LlamaForCausalLM

device = torch.device("cuda")
model_id = "./Llama-3.1-8B-Instruct"
use_sparse = True

with open("./paper/prompt.txt", 'r', encoding='utf-8') as f:
    prompt = f.read()

messages = [
    {
        "role": "user",
        "content": f"{prompt}\n{prompt}\nRead this paper carefully, and give me a brief summary.\n"
    },
]

seq_len = 8192

parser = argparse.ArgumentParser()
parser.add_argument("--is_sparse", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(model_id)

if args.is_sparse:
    transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )[:seq_len].to(device)
    model.eval()
    model.ss_init(
        device="cuda",
        budget=1024,
        page_size=32,
        max_length=32768,
        dtype=torch.float16,
    )
    outputs = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.eos_token_id
    )
    model.ss_flush()
    new_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).to(device)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )[:seq_len].to(device)
    model.eval()
    outputs = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(response)
