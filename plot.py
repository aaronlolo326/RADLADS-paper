#!/usr/bin/env python3
import os
import argparse
import glob
import json
import re
import csv
from functools import cmp_to_key
from collections import defaultdict

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint

RESULTS_ROOT = "/home/hei/RADLADS-paper/results"
OUT_DIR = "/home/hei/RADLADS-paper/plots"

all_tasks = ["arc_challenge", "arc_easy", "gsm8k", "mathqa", "mmlu", "piqa", "race", "winogrande", "hellaswag", "lambada_openai", "openbookqa"]
unsupported_tasks = []
selected_tasks = "hellaswag,lambada_openai,mmlu".split(",")

pretrained_models = [
    "Qwen__Qwen2.5-7B-Instruct",
    "Qwen__Qwen3-8B-Base"
]

def parse_model_name(model_name):
    # "[a]__[b]-[c]__[d]"
    split1 = model_name.split('__')
    if len(split1) < 3:
        raise ValueError(f"Unexpected model name format: {model_name}")
    a = split1[0]
    b_and_c = split1[1]
    # REGEX: Match "[b]-[c]", where [b] can contain hyphens
    m = re.match(r"(.*)-(\d+(?:-\d+k)?)$", b_and_c)
    if m:
        b, c = m.group(1), m.group(2)
    else:
        # fallback: assume all of b_and_c is b, and c is ""
        b, c = b_and_c, ""
    # d is everything after the 2nd '__' (e.g., "__qwen2_final") or "" if not present
    d = split1[2] if len(split1) > 2 else ""
    return a, b, c, d

def sort_key(model_name):
    a, b, c, d = parse_model_name(model_name)

    # [a] : date, compare as integer if possible
    try:
        a_key = int(a)
    except:
        a_key = a

    # [b] : lexicographical
    b_key = b

    # [c] : "1", "2", "4", or "4-16k", "4-32k", etc.
    # Ideally, sort "1" < "2" < "4" < "4-16k" < "4-32k" < ...
    if c in ("1", "2", "4"):
        c_key = (0, int(c))
    else:
        m = re.match(r"(\d+)-(\d+)k", c)
        if m:
            n1, n2 = m.groups()
            c_key = (1, int(n1), int(n2))
        else:
            c_key = (2, c)

    # [d] : [model]_[ckpt], where [ckpt] can be init, 1, 2, ..., final
    if '-' in d:
        model_d, ckpt = d.rsplit('-', 1)
    else:
        model_d, ckpt = '', d
    # init < numbers < final
    if ckpt == "init":
        ckpt_key = (0, 0)
    elif ckpt == "final":
        ckpt_key = (2, 0)
    else:
        # try to parse number, otherwise lex sort
        try:
            ckpt_num = int(ckpt)
            ckpt_key = (1, ckpt_num)
        except:
            ckpt_key = (1, ckpt)
    d_key = (model_d, ckpt_key)

    return (a_key, b_key, c_key, d_key)

def load_results(results_root: str, runs_name: List[str], exc_runs: List[str], final_only: bool, step2_only: bool):
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
        if model_dir in pretrained_models:
            proceed = True
        else:
            a,b,c,d = parse_model_name(model_dir)
            if step2_only:
                # print (a,b,c,d)
                if not c.startswith("4"):
                    continue
            # if c.startswith("4"):
            #     continue
            run_name_step, ckpt = model_dir.rsplit("__", maxsplit=1)
            if runs_name == []:
                proceed = True
                for exc_run in exc_runs:
                    if run_name_step.startswith(exc_run):
                        proceed = False
                        break
            else:
                proceed = False
                for targ_run_name in runs_name:
                    if run_name_step.startswith(targ_run_name) and targ_run_name not in exc_runs:
                        proceed = True
                        break
            if final_only and "final" not in ckpt:
                continue

        if proceed:
            with open(fp, "r") as f:
                j = json.load(f)

            for task_name, task_res in j.items():
                if task_name in ['gsm8k']:
                    key = "exact_match,strict-match"
                else:
                    key = "acc,none"
                acc = task_res.get(key, None)
                acc_norm = task_res.get("acc_norm,none", None)
                # map to renamed keys
                data[task_name][model_dir] = {
                    "acc": float(f"{acc * 100:.2f}") if acc is not None else None,
                    "acc_norm": float(f"{acc_norm * 100:.2f}") if acc_norm is not None else None,
                }
                # print (task_name, acc)

    return data


# def plot_task(task_name, task_data, out_dir):
#     """
#     task_data: dict[model] -> {"acc": float or None, "acc_norm": float or None}
#     Creates a bar chart with models on x-axis and two bars (acc, acc_norm).
#     """
#     models = sorted(task_data.keys())
#     accs = [task_data[m]["acc"] for m in models]
#     acc_norms = [task_data[m]["acc_norm"] for m in models]

#     # Handle case where one of the metrics is missing entirely
#     has_acc = any(v is not None for v in accs)
#     has_acc_norm = any(v is not None for v in acc_norms)

#     if not has_acc and not has_acc_norm:
#         return  # nothing to plot

#     x = np.arange(len(models))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))

#     bars = []
#     labels = []

#     if has_acc:
#         bars_acc = ax.bar(x - width/2, accs, width, label="acc")
#         bars.append(bars_acc)
#         labels.append("acc")

#     if has_acc_norm:
#         bars_accn = ax.bar(x + (0 if not has_acc else width/2),
#                            acc_norms, width, label="acc_norm")
#         bars.append(bars_accn)
#         labels.append("acc_norm")

#     ax.set_title(f"{task_name}")
#     ax.set_xticks(x)
#     ax.set_xticklabels(models, rotation=45, ha="right")
#     ax.set_ylabel("Score")
#     ax.set_ylim(0, 1.0)  # accuracy in [0,1]
#     ax.legend()

#     plt.tight_layout()
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f"{task_name}.png")
#     plt.savefig(out_path, dpi=200)
#     plt.close(fig)


def plot_aggregate(data, metric: str, out_dir: str, date_str: str, runs_name: List[str]):
    """
    Create a grouped horizontal bar plot for one metric ('acc' or 'acc_norm'):
    - y-axis: tasks (one group per task)
    - within each group: one bar per model
    """
    # collect all tasks and models
    tasks = sorted(data.keys())
    tasks = [x for x in tasks if x in selected_tasks]

    def cmp_method(model_name1, model_name2):
        """
        Model names are like: "[a]__[b]-[c]__[d]". Compare two model names according to a custom sort order:
        - primarily sort according to [a], which is a simple date string like 20251208
        - then sort according to [b], which is a name like qwen3-8b_rwkv7qknorm_s3-2048
        - then sort according to [c], which can take a value from ["1", "2", "4"] or is f"4-{i}k" where i is a number
        - finally sort according to [d], which is like [model]_[ckpt], and we want [model]_init < [model]_1 < [model]_2 < ... < [model]_final
        """
        import re

        if model_name1 in pretrained_models:
            k1 = model_name1 
        else:
            k1 = sort_key(model_name1)
        if model_name2 in pretrained_models:
            k2 = model_name2
        else:
            k2 = sort_key(model_name2)
        
        if isinstance(k1, str):
            if isinstance(k2, str):
                if k1 < k2:
                    return -1
                elif k1 > k2:
                    return 1
                else:
                    return 0
            else:
                return -1
        if isinstance(k2, str):
            return 1
            
        if k1 < k2:
            return -1
        elif k1 > k2:
            return 1
        else:
            return 0


    
        
    all_models = sorted({m for task_data in data.values() for m in task_data.keys()}, key=cmp_to_key(cmp_method))
    # pprint (all_models)
    
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

    y = np.arange(len(tasks))
    total_height = 0.8
    bar_height = total_height / len(models)

    fig, ax = plt.subplots(figsize=(13,len(all_tasks) * 1))

    # Use a consistent color for each model based on model name
    color_map = plt.get_cmap('tab20')

    for i, model in enumerate(models):
        # center the whole group around each task position
        offsets = y - total_height / 2 + i * bar_height + bar_height / 2

        widths = []
        for task_name in tasks:
            val = data[task_name].get(model, {}).get(metric)
            widths.append(val if val is not None else 0.0)

        # Apply color_map to choose a color for this model
        color = color_map(i % color_map.N)

        bars = ax.barh(
            offsets,
            widths,
            bar_height, 
            label=model,
            color=color
        )
        # ax.bar_label(bars, fmt='%.2f') # Formats labels to two decimal places

    ax.set_title(f"{metric} per task")
    ax.set_yticks(y)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 90)
    # Ensure legend shows the same color as bars
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(0.6, 0.2), bbox_transform=fig.transFigure)

    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.6, bottom=0.10)
    os.makedirs(out_dir, exist_ok=True)

    multi_run = len(runs_name) != 1
    if not multi_run:
        out_dir = os.path.join(out_dir, runs_name[0])
        os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "_".join([date_str, f"grouped_{metric}.png"]))
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved task and aggregate plots to: {out_path}")


def generate_csv(data, metric: str, out_dir: str):
    """
    Generate CSV file where columns are tasks, rows are model names, and values are the metric.
    Column order: arc_challenge, arc_easy, gsm8k, mathqa, mmlu, piqa, race, winogrande, then the rest.
    Rows are sorted by increasing average score.
    """
    # Define the preferred column order
    preferred_tasks = ["arc_challenge", "arc_easy", "gsm8k", "mathqa", "mmlu", "piqa", "race", "winogrande", "hellaswag", "lambada_openai", "openbookqa"]
    column_order = preferred_tasks
    # Get all tasks from data
    # all_tasks_in_data = set(data.keys())
    
    # Build column order: preferred tasks first (if they exist), then the rest sorted
    # column_order = []
    # for task in preferred_tasks:
    #     if task in all_tasks_in_data:
    #         column_order.append(task)
            # all_tasks_in_data.remove(task)
    
    # Add the remaining tasks in sorted order
    # column_order.extend(sorted(all_tasks_in_data))
    
    # Get all models from data
    all_models = set()
    for task_data in data.values():
        all_models.update(task_data.keys())
    
    # Calculate average score for each model and collect data
    model_data = []
    for model in all_models:
        scores = []
        row_data = {}
        # show_model = True
        for task in column_order:
            val = data.get(task, {}).get(model, {}).get(metric)
            row_data[task] = val
            if val is not None:
                scores.append(val)
            else:
                # if task in unsupported_tasks:
                scores.append(-1)
                # else:
                #     show_model = False
                    # break
        # if not show_model:
        #     continue
        
        # Calculate average (only for non-None values)
        scores_ = [x for x in scores if x != -1]
        avg_score = sum(scores_) / len(scores_) if scores_ else 0.0
        model_data.append((model, avg_score, row_data))
    
    # Sort by increasing average score
    model_data.sort(key=lambda x: x[1])
    
    # Write CSV file
    os.makedirs(out_dir, exist_ok=True)
    csv_filename = f"all_results_{metric}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(["model"] + column_order +  ["avg_score"])
        
        # Write data rows
        for model, avg_score, row_data in model_data:
            row = [model]
            for task in column_order:
                val = row_data[task]
                # Write empty string for None values, or the value itself
                row.append("" if val is None else val)
            row += [avg_score]
            writer.writerow(row)
    
    print(f"Saved CSV file to: {csv_path}")


def main():

    parser = argparse.ArgumentParser(description="plot.py")
    parser.add_argument("--runs_name", help="runs_name", default="",)
    parser.add_argument("--exc_runs", help="exc_runs", default="",)
    parser.add_argument("--date_str", help="date_str", default="",)
    parser.add_argument("--final_only", help="final ckpt only", action="store_true")
    parser.add_argument("--step2_only", help="step 2 ckpt only", action="store_true")
    parser.add_argument("--csv", help="gen csv", action="store_true")
    # parser.add_argument("--age", type=int, help="Your age", default=30)
    args = parser.parse_args()

    runs_name = args.runs_name.replace(",", " ")
    runs_name = runs_name.split()
    exc_runs = args.exc_runs.replace(",", " ")
    exc_runs = exc_runs.split()
    
    data = load_results(RESULTS_ROOT, runs_name=runs_name, exc_runs=exc_runs, final_only=args.final_only, step2_only=args.step2_only)
    os.makedirs(OUT_DIR, exist_ok=True)

    # per-task plots
    # for task_name, task_data in data.items():
    #     if task_name in selected_tasks:
    #         plot_task(task_name, task_data, OUT_DIR)

    # aggregate over all tasks for each metric
    plot_aggregate(data, "acc", OUT_DIR, date_str=args.date_str, runs_name=runs_name)
    plot_aggregate(data, "acc_norm", OUT_DIR, date_str=args.date_str, runs_name=runs_name)
    
    # generate CSV files
    if args.csv:
        generate_csv(data, "acc", OUT_DIR)
        # generate_csv(data, "acc_norm", OUT_DIR)

if __name__ == "__main__":
    main()