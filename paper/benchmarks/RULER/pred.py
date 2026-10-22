import os
import json
import random
import numpy as np
import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from paper.models.qwen import Qwen3ForCausalLM
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM
transformers.models.qwen3.Qwen3ForCausalLM = Qwen3ForCausalLM

transformers.logging.set_verbosity_error()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_chat(prompt):
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def truncate_prompt(prompt, max_prompt_length=32000):
    tokenized_prompt = tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]

    if len(tokenized_prompt) > max_prompt_length:
        half = max_prompt_length // 2

        prompt = (
            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
            tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        )

    return prompt


def split_thinking(output_ids):
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id not in output_ids:
        return tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        ).strip(), "", False

    split = len(output_ids) - output_ids[::-1].index(think_end_id)
    thinking = tokenizer.decode(
        output_ids[:split],
        skip_special_tokens=True
    ).strip()
    answer = tokenizer.decode(
        output_ids[split:],
        skip_special_tokens=True
    ).strip()

    return thinking, answer, True


def get_pred(data, task, out_path):
    for i, obj in enumerate(tqdm(data, desc=task)):
        try:
            model.ss_flush()
            torch.cuda.synchronize(device)

            prompt = obj["input"] + obj.get("answer_prefix", "")

            # First truncate the raw prompt to at most 32000 tokens.
            prompt = truncate_prompt(
                prompt,
                max_prompt_length=32000,
            )

            # Then apply chat template and tokenize.
            chat_input = build_chat(prompt)
            inputs = tokenizer(
                chat_input,
                return_tensors="pt",
                truncation=False,
            ).to(device)

            context_length = inputs.input_ids.shape[-1]
            sample_max_new_tokens = min(
                max_new_tokens,
                max_length - context_length
            )

            if sample_max_new_tokens <= 0:
                raise ValueError(
                    f"input exceeds max_length: "
                    f"context_length={context_length}, "
                    f"max_length={max_length}"
                )

            output_ids = model.generate(
                **inputs,
                max_new_tokens=sample_max_new_tokens,
                do_sample=False,
            )[0, context_length:].tolist()

            if enable_thinking:
                thinking, pred, finished = split_thinking(output_ids)
            else:
                thinking = ""
                pred = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True
                ).strip()
                finished = None

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "index": obj.get("index", i),
                            "pred": pred,
                            "thinking": thinking,
                            "finished_thinking": finished,
                            "output_tokens": len(output_ids),
                            "max_new_tokens": sample_max_new_tokens,
                            "answers": obj["outputs"],
                            "length": obj.get("length"),
                        },
                        ensure_ascii=False
                    )
                    + "\n"
                )

        except torch.cuda.OutOfMemoryError:
            print(f"\n[OOM] length={length} task={task} idx={i}")
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] length={length} task={task} idx={i}: {e}")


seed = 42
seed_everything(seed)

TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

# config
device = "cuda:0"
model_name = "Qwen3-8B"
length = 32768

max_length = 32768
max_prompt_length = 32000
max_new_tokens = 768

page_size = 32
budget = 1024
pred_root = "pred"

tokenizer = AutoTokenizer.from_pretrained(f"/root/autodl-tmp/{model_name}")
model = AutoModelForCausalLM.from_pretrained(
    f"/root/autodl-tmp/{model_name}",
    torch_dtype=torch.bfloat16
).to(device)

model.ss_init(
    batch_size=1,
    device=device,
    budget=budget,
    page_size=page_size,
    max_length=32768,
    num_dense_layers=2,
    dtype=torch.bfloat16,
)

num_samples = 100
enable_thinking = False
dataset = load_dataset(f"lighteval/RULER-{length}-Qwen-3")

for task in TASKS:
    out_path = f"{pred_root}/{model_name}/{length}/{budget}/{task}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = dataset[task]
    if num_samples is not None:
        data = data.shuffle(seed=seed).select(range(min(num_samples, len(data))))
    get_pred(data, task, out_path)

model.ss_flush()