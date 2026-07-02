import os
import re
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

model_name = "Llama-3.1-8B-Instruct"
base_dir = "./pred"
full_budget = 32768  # the budget value that represents Full Attention

# discover all "{model_name}-{budget}" folders under base_dir and load their result.json
results = {}  # budget -> {task: score}
pattern = re.compile(rf"^{re.escape(model_name)}-(\d+)$")

for name in sorted(os.listdir(base_dir)):
    m = pattern.match(name)
    if not m:
        continue
    budget = int(m.group(1))
    path = os.path.join(base_dir, name, "result.json")
    if not os.path.exists(path):
        print(f"[warn] missing result.json, skipped: {path}")
        continue
    with open(path, "r", encoding="utf-8") as f:
        results[budget] = json.load(f)

if not results:
    raise ValueError(f"No '{model_name}-<budget>' folders with result.json found under {base_dir}.")

if full_budget not in results:
    raise ValueError(
        f"Full-attention results (budget={full_budget}) not found under {base_dir}. "
        f"Found budgets: {sorted(results.keys())}"
    )

budgets = sorted(results)

tasks = sorted(set.intersection(*(set(result) for result in results.values())))
if not tasks:
    raise ValueError("No overlapping tasks across all budget results.")

full_vals = [results[full_budget][t] for t in tasks]
budget_vals = {budget: [results[budget][t] for t in tasks] for budget in budgets}

order = np.argsort(full_vals)[::-1]
tasks = [tasks[i] for i in order]
full_vals = [full_vals[i] for i in order]
budget_vals = {
    budget: [vals[i] for i in order]
    for budget, vals in budget_vals.items()
}

task_labels = {
    "2wikimqa": "2WikiMQA",
    "gov_report": "GovReport",
    "hotpotqa": "HotpotQA",
    "lcc": "LCC",
    "multi_news": "Multi-News",
    "multifieldqa_en": "MultiFieldQA-en",
    "musique": "MuSiQue",
    "passage_retrieval_en": "PassageRetrieval-en",
    "qasper": "Qasper",
    "qmsum": "QMSum",
    "repobench-p": "RepoBench-P",
    "triviaqa": "TriviaQA",
}

font_dir = os.path.dirname(os.path.abspath(__file__))
calibri_regular_path = os.path.join(font_dir, "calibri-regular.ttf")
calibri_bold_path = os.path.join(font_dir, "calibri-bold.ttf")

for font_path in [calibri_regular_path, calibri_bold_path]:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)

label_font = fm.FontProperties(fname=calibri_regular_path, size=10.3)
label_font_bold = fm.FontProperties(fname=calibri_bold_path, size=10.3, weight="bold")
small_font = fm.FontProperties(fname=calibri_regular_path, size=8.2)
summary_font_bold = fm.FontProperties(fname=calibri_bold_path, size=9.4, weight="bold")
legend_font_bold = fm.FontProperties(fname=calibri_bold_path, size=9.3, weight="bold")
axis_font_bold = fm.FontProperties(fname=calibri_bold_path, size=11.0, weight="bold")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "grid.color": "#e6e8eb",
    "grid.linewidth": 0.65,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#4a4a4a",
    "ytick.color": "#222222",
    "xtick.labelsize": 9.5,
})

budget_colors = {
    full_budget: "#b8bdc3",
}
sparse_budgets = [b for b in budgets if b != full_budget]
sparse_cmap = plt.get_cmap("tab10")
for idx, budget in enumerate(sparse_budgets):
    color = sparse_cmap(idx % sparse_cmap.N)
    budget_colors[budget] = color

n = len(tasks)
y = np.arange(n)
bar_height = min(0.22, 0.72 / len(budgets))
offsets = (np.arange(len(budgets)) - (len(budgets) - 1) / 2) * bar_height

fig, ax = plt.subplots(figsize=(6.8, max(3.0, 0.40 * n + 1.15)), dpi=200)
ax.grid(axis="x")

for offset, budget in zip(offsets, budgets):
    vals = budget_vals[budget]
    label = (
        f"Full Attention ({budget // 1024}k)"
        if budget == full_budget
        else f"Sparse Attention ({budget})"
    )
    ax.barh(
        y + offset,
        vals,
        height=bar_height * 0.86,
        color=budget_colors[budget],
        label=label,
        zorder=3,
    )

for offset, budget in zip(offsets, budgets):
    color = "#6b6f76" if budget == full_budget else budget_colors[budget]
    for yi, v in zip(y + offset, budget_vals[budget]):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontproperties=small_font, color=color)

ax.set_yticks(y)
ax.set_yticklabels([])  # draw left-aligned labels manually in a fixed label column
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)

for yi, task in zip(y, tasks):
    ax.text(
        -0.34, yi, task_labels.get(task, task),
        transform=ax.get_yaxis_transform(),  # x: axes fraction, y: data coords
        ha="left", va="center",
        fontproperties=label_font_bold,
        color="#202124",
        clip_on=False,
    )

ax.set_xlabel("Score", fontproperties=axis_font_bold)
xmax = max(max(vals) for vals in budget_vals.values())
ax.set_xlim(0, xmax * 1.13)

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(left=False)

ax.legend(
    frameon=False,
    loc="lower right",
    prop=legend_font_bold,
    handlelength=1.4,
    labelspacing=0.4,
)

fig.subplots_adjust(left=0.31, right=0.985, top=0.905, bottom=0.105)
fig.savefig("./longbench.pdf", bbox_inches="tight")
