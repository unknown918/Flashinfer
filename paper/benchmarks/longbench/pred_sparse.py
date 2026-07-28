import os
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from paper.models.llama import LlamaForCausalLM

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM

transformers.logging.set_verbosity_error()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


# building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # Llama-3.1-prompt template
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a helpful assistant<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )

    return prompt


def get_pred(data, max_length, max_gen, prompt_format, dataset, out_path):
    for i, json_obj in enumerate(tqdm(data)):
        try:
            model.ss_flush()
            torch.cuda.synchronize()

            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt"
            ).input_ids[0]

            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = (
                        tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                        tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                )

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name)

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]

            if dataset == "samsum":
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=0,
                    min_length=context_length + 1,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.encode("\n", add_special_tokens=False)[-1]
                    ],
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=0,
                )[0]

            pred = tokenizer.decode(
                output[context_length:],
                skip_special_tokens=True
            )

            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"]
                }, f, ensure_ascii=False)
                f.write('\n')

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] dataset={dataset} idx={i}")
            continue

        except Exception as e:
            print(f"\n[ERROR] dataset={dataset} idx={i}")
            print(str(e))
            continue


seed_everything(42)

# full dataset
# datasets = [
#     "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
#     "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
#     "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
# ]

datasets = [
    "qasper", "multifieldqa_en",  # Single-Doc QA
    "hotpotqa", "2wikimqa", "musique",  # Multi-Doc QA
    "gov_report", "qmsum", "multi_news",  # Summarization
    "triviaqa",  # Few-shot
    "passage_retrieval_en",  # Synthetic
    "lcc", "repobench-p"  # Code
]

model2maxlen = json.load(open("config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

# config
device = "cuda:0"
model_name = "Llama-3.1-8B-Instruct"
max_length = model2maxlen[model_name]
page_size = 32
budget = 2048

# predict on each dataset
os.makedirs("pred", exist_ok=True)
os.makedirs(f"pred/{model_name}-{budget}", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(f"./{model_name}", )
model = AutoModelForCausalLM.from_pretrained(
    f"./{model_name}",
    torch_dtype=torch.bfloat16
).to(device)

model.ss_init(
    device=device,
    budget=budget,
    page_size=page_size,
    max_length=max_length,
    dtype=torch.bfloat16,
)

for dataset in datasets:
    data = []
    out_path = f"pred/{model_name}-{budget}/{dataset}.jsonl"
    with open(f"data/{dataset}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    # json_obj
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    get_pred(
        data=data,
        max_length=max_length,
        max_gen=max_gen,
        prompt_format=prompt_format,
        dataset=dataset,
        out_path=out_path
    )

model.ss_flush()
