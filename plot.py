#!/usr/bin/env python3
import os
import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = "/home/hei/RADLADS-paper/results"
OUT_DIR = "/home/hei/RADLADS-paper/plots"

selected_tasks = "winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,mmlu,mathqa,race,gsm8k".split(",")

def load_results(results_root: str):
    """
    Walk each model directory under results_root, load lm_eval_results.json,
    and return a nested dict:
        data[task][model] = {"acc": float or None, "acc_norm": float or None}
    """
    pattern = os.path.join(results_root, "*", "lm_eval_results.json")
    files = glob.glob(pattern)
    data = defaultdict(dict)

    for fp in sorted(files):
        model_dir = os.path.basename(os.path.dirname(fp))
        with open(fp, "r") as f:
            j = json.load(f)

        for task_name, task_res in j.items():
            acc = task_res.get("acc,none", None)
            acc_norm = task_res.get("acc_norm,none", None)
            # map to renamed keys
            data[task_name][model_dir] = {
                "acc": acc,
                "acc_norm": acc_norm,
            }

    return data


def plot_task(task_name, task_data, out_dir):
    """
    task_data: dict[model] -> {"acc": float or None, "acc_norm": float or None}
    Creates a bar chart with models on x-axis and two bars (acc, acc_norm).
    """
    models = sorted(task_data.keys())
    accs = [task_data[m]["acc"] for m in models]
    acc_norms = [task_data[m]["acc_norm"] for m in models]

    # Handle case where one of the metrics is missing entirely
    has_acc = any(v is not None for v in accs)
    has_acc_norm = any(v is not None for v in acc_norms)

    if not has_acc and not has_acc_norm:
        return  # nothing to plot

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))

    bars = []
    labels = []

    if has_acc:
        bars_acc = ax.bar(x - width/2, accs, width, label="acc")
        bars.append(bars_acc)
        labels.append("acc")

    if has_acc_norm:
        bars_accn = ax.bar(x + (0 if not has_acc else width/2),
                           acc_norms, width, label="acc_norm")
        bars.append(bars_accn)
        labels.append("acc_norm")

    ax.set_title(f"{task_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)  # accuracy in [0,1]
    ax.legend()

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_aggregate(data, metric: str, out_dir: str):
    """
    Create a grouped bar plot for one metric ('acc' or 'acc_norm'):
    - x-axis: tasks (one group per task)
    - within each group: one bar per model
    """
    # collect all tasks and models
    tasks = sorted(data.keys())
    tasks = [x for x in tasks if x in selected_tasks]
    all_models = sorted({m for task_data in data.values() for m in task_data.keys()})

    # keep only models that have at least one non-None value for this metric
    models = []
    for model in all_models:
        vals = []
        for task_name in tasks:
            metrics = data[task_name].get(model, {})
            vals.append(metrics.get(metric))
        if any(v is not None for v in vals):
            models.append(model)

    if not models:
        return

    x = np.arange(len(tasks))
    total_width = 0.8
    bar_width = total_width / len(models)

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 0.6), 5))

    for i, model in enumerate(models):
        # center the whole group around each task position
        offsets = x - total_width / 2 + i * bar_width + bar_width / 2

        heights = []
        for task_name in tasks:
            val = data[task_name].get(model, {}).get(metric)
            heights.append(val if val is not None else 0.0)

        bars = ax.bar(offsets, heights, bar_width, label=model)
        # ax.bar_label(bars, fmt='%.1f') # Formats labels to one decimal place


    ax.set_title(f"{metric} per task (grouped by model)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 0), bbox_transform=fig.transFigure)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7) 

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"grouped_{metric}_by_task.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    data = load_results(RESULTS_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)

    # per-task plots
    for task_name, task_data in data.items():
        if task_name in selected_tasks:
            plot_task(task_name, task_data, OUT_DIR)

    # aggregate over all tasks for each metric
    plot_aggregate(data, "acc", OUT_DIR)
    plot_aggregate(data, "acc_norm", OUT_DIR)

    print(f"Saved task and aggregate plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()