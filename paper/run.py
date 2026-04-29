import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

device = torch.device("cuda:0")
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
        "content": f"{prompt}\nPlease, can you read this paper carefully, and give me a brief summary.\n"
    },
]  # 5532

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

print("Sequence Length: ", input_ids.shape[1])

model.ss_init(
    device="cuda:0",
    budget=1024,
    page_size=32,
    max_length=32768,
    dtype=torch.float16,
)

generated_outputs = model.generate(
    input_ids,
    max_new_tokens=10,
    do_sample=False,
    temperature=0,
    pad_token_id=tokenizer.eos_token_id
)

model.ss_flush()

new_tokens = generated_outputs[0][input_ids.shape[-1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
