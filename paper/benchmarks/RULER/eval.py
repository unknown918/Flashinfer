import os
import json
import re

from rich.console import Console
from rich.table import Table


def clean(text):
    return re.sub(r"[\x00-\x1f]", "\n", text).strip().lower()


def string_match_part(pred, answers):
    pred = clean(pred)
    return max(float(clean(str(answer)) in pred) for answer in answers)


def string_match_all(pred, answers):
    pred = clean(pred)
    return (
            sum(float(clean(str(answer)) in pred) for answer in answers)
            / len(answers)
    )


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

model_name = "Qwen3-8B"

budgets = [
    "Dense",
    "256",
    "512",
    "1024",
    "2048",
]

lengths = [32768, ]

pred_root = f"pred/{model_name}"

results = {}

for length in lengths:
    results[str(length)] = {}

    for budget in budgets:
        scores = []
        output_tokens = []

        for task in TASKS:
            path = f"{pred_root}/{length}/{budget}/{task}.jsonl"

            if not os.path.exists(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                data = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

            metric = (
                string_match_part
                if task.startswith("qa_")
                else string_match_all
            )

            scores.extend(
                metric(obj["pred"], obj["answers"])
                for obj in data
            )

            output_tokens.extend(
                obj["output_tokens"]
                for obj in data
            )

        if scores:
            results[str(length)][budget] = {
                "score": round(sum(scores) / len(scores) * 100, 2),
                "num_samples": len(scores),
                "avg_output_tokens": round(sum(output_tokens) / len(output_tokens), 2),
            }

os.makedirs(pred_root, exist_ok=True)
summary_path = f"{pred_root}/summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )


console = Console(force_terminal=False)

table = Table(
    title=f"{model_name} Evaluation Results",
    show_header=True,
)

table.add_column("Budget", justify="right")

for length in lengths:
    table.add_column(
        str(length),
        justify="right",
    )

for budget in budgets:
    row = [budget]

    for length in lengths:
        result = results.get(str(length), {}).get(budget)

        if result is None:
            row.append("-")
        else:
            row.append(f'{result["score"]:.2f}')

    table.add_row(*row)

console.print(table)
console.print(f"\nSummary saved to: {summary_path}")