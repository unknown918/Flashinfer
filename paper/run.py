import time
import statistics
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from .models.llama import LlamaForCausalLM
from .models.qwen import Qwen3ForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
transformers.models.qwen3.Qwen3ForCausalLM = Qwen3ForCausalLM


class BatchTimingCriteria(StoppingCriteria):
    """Record token boundaries without copying every token to the CPU."""

    def __init__(self, device, runner):
        self.device = torch.device(device)
        self.runner = runner
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.decode_start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.token_count = 0

    def start(self):
        torch.cuda.synchronize(self.device)
        self.start_time = time.perf_counter()
        self.start_event.record(torch.cuda.current_stream(self.device))

    def __call__(self, input_ids, scores, **kwargs):
        if self.token_count == 0:
            self.first_token_event.record(torch.cuda.current_stream(self.device))
            # Restore KV and recapture before the TPOT boundary, after prefill
            # activations have been released. Wall time still includes setup.
            self.runner.prepare_decode()
            self.decode_start_event.record(torch.cuda.current_stream(self.device))
        self.token_count += 1
        return False

    def end(self):
        self.end_event.record(torch.cuda.current_stream(self.device))
        self.end_event.synchronize()
        self.wall_time = time.perf_counter() - self.start_time


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="Llama-3.1-8B-Instruct")
parser.add_argument("--prompt-path", type=str, default="prompt.txt")
parser.add_argument("--budget", type=int, default=2048)
parser.add_argument("--seq-len", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-dense-layers", type=int, default=2)
parser.add_argument("--max-new-tokens", type=int, default=100)
parser.add_argument("--streaming", action="store_true")
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
    model.attn_runner.streaming = args.streaming
    if seq_len == 32000 and batch_size == 8:
        # spare room for activation
        model.attn_runner.cache_manager.enable_prefill_offload()

    # warmup; use the same generation shape as measurement
    for _ in range(3):
        _ = model.generate(input_ids, **generation_kwargs)
        torch.cuda.synchronize()
        model.ss_flush()

    timings = []
    for _ in range(3):
        model.ss_flush()
        timer = BatchTimingCriteria(device, model.attn_runner)
        timer.start()
        outputs_ids = model.generate(
            input_ids,
            stopping_criteria=StoppingCriteriaList([timer]),
            **generation_kwargs,
        )
        timer.end()

        generated_length = outputs_ids.shape[1] - input_ids.shape[1]
        assert generated_length == max_new_tokens
        assert timer.token_count == generated_length
        ttft = timer.start_event.elapsed_time(timer.first_token_event) / 1e3
        decode_time = timer.decode_start_event.elapsed_time(timer.end_event) / 1e3
        timings.append((ttft, decode_time, timer.wall_time))

    tpot_samples = [decode / (max_new_tokens - 1) for _, decode, _ in timings]
    tpot = statistics.median(tpot_samples)
    batch_throughput = batch_size / tpot
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
