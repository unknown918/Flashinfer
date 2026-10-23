import time
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from .models.llama import LlamaForCausalLM
from .models.qwen import Qwen3ForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
transformers.models.qwen3.Qwen3ForCausalLM = Qwen3ForCausalLM


class BatchTimingStreamer(BaseStreamer):
    """Measure batch TTFT and decode duration within one generate() call."""

    def __init__(self, device):
        self.device = torch.device(device)
        self._prompt_seen = False
        self.start_time = None
        self.first_token_time = None
        self.end_time = None

    def _sync(self):
        torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self.start_time = time.perf_counter()

    def put(self, value):
        # The first put() contains the prompt for the whole batch.
        if not self._prompt_seen:
            self._prompt_seen = True
            return

        # The next put() corresponds to the first generated token for the batch.
        if self.first_token_time is None:
            self._sync()
            self.first_token_time = time.perf_counter()

    def end(self):
        self._sync()
        self.end_time = time.perf_counter()


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="Llama-3.1-8B-Instruct")
parser.add_argument("--prompt-path", type=str, default="prompt.txt")
parser.add_argument("--budget", type=int, default=2048)
parser.add_argument("--seq-len", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-dense-layers", type=int, default=2)
parser.add_argument("--max-new-tokens", type=int, default=100)
args = parser.parse_args()

model_path = args.model_path
prompt_path = args.prompt_path
budget = args.budget
seq_len = args.seq_len
batch_size = args.batch_size
num_dense_layers = args.num_dense_layers
max_new_tokens = args.max_new_tokens

device = "cuda:0"
sink_size = 4
page_size = 32
max_length = 32768

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)

model.eval()

with open(prompt_path, "r", encoding="utf-8") as f:
    prompt = f.read()

article_ids = tokenizer(
    prompt,
    add_special_tokens=False,
    return_tensors="pt",
).input_ids[0]
article_ids = article_ids[:seq_len]
article = tokenizer.decode(article_ids, skip_special_tokens=True)
messages = [
    {
        "role": "user",
        "content": f"Please read this article, and summarize it briefly.\n{article}",
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False
).to(device)
input_ids = input_ids.repeat(batch_size, 1).to(device)
# assert input_ids.shape[0] >= seq_len

generation_kwargs = dict(
    max_new_tokens=max_new_tokens,
    min_new_tokens=max_new_tokens,
    do_sample=False,
    eos_token_id=None,
    pad_token_id=tokenizer.eos_token_id,
)

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

    # warmup; use the same generation shape as measurement
    for _ in range(3):
        _ = model.generate(input_ids, **generation_kwargs)
        torch.cuda.synchronize()
        model.ss_flush()

    # Measure TTFT and TPOT in the same generation call.
    model.ss_flush()
    torch.cuda.synchronize()

    timer = BatchTimingStreamer(device)
    timer.start()

    outputs_ids = model.generate(
        input_ids,
        streamer=timer,
        **generation_kwargs,
    )

    torch.cuda.synchronize()

    if timer.first_token_time is None or timer.end_time is None:
        raise RuntimeError("The streamer did not receive generated tokens correctly.")

    generated_length = outputs_ids.shape[1] - input_ids.shape[1]
    assert generated_length == max_new_tokens

    ttft = timer.first_token_time - timer.start_time
    decode_time = timer.end_time - timer.first_token_time
    tpot = decode_time / (generated_length - 1)
    batch_throughput = batch_size * (generated_length - 1) / decode_time

    model.ss_flush()

outputs = outputs_ids[0][input_ids.shape[1]:]
response = tokenizer.decode(outputs, skip_special_tokens=True)

print(
    "\n===== STATS =====\n"
    f"Seq Len      : {seq_len}\n"
    f"Batch size   : {batch_size}\n"
    f"Throughput   : {batch_throughput:.3f} token/s\n"
    f"TPOT         : {tpot * 1e3:.3f} ms/token\n"
)
