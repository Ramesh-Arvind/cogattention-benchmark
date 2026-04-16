"""Point-biserial discrimination (CTT item analysis).

Task-level: for each task T, correlate pass/fail across models with each model's
total score. r_pb ranges in [-1, 1]; >0.3 is considered discriminating in
classical test theory.

Item-level (pilot only): same analysis at per-item granularity across the 3
pilot models. Reported as exploratory given small model pool.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_leaderboard(csv_path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    """Parse leaderboard CSV.

    Returns:
      task_model_correct: {task_name: {model: 0/1}}
      model_total: {model: overall numerical CAS}
    """
    task_model: Dict[str, Dict[str, int]] = defaultdict(dict)
    model_total: Dict[str, float] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model"].strip()
            task = row["Task_Name"].strip()
            numeric = row["Numerical_Result"].strip()
            boolean = row["Boolean_Result"].strip()
            if not task and numeric:
                model_total[model] = float(numeric)
            elif task and boolean:
                task_model[task][model] = 1 if boolean.lower() == "true" else 0
    return dict(task_model), model_total


def point_biserial(binary: np.ndarray, continuous: np.ndarray) -> float:
    """r_pb = Pearson correlation of a binary vector with a continuous vector."""
    if len(binary) < 2 or np.std(binary) == 0 or np.std(continuous) == 0:
        return float("nan")
    return float(np.corrcoef(binary.astype(float), continuous)[0, 1])


def analyze_tasks(
    task_model: Dict[str, Dict[str, int]],
    model_total: Dict[str, float],
) -> List[Dict]:
    """Per-task point-biserial against model total CAS."""
    models = sorted(model_total.keys())
    totals = np.array([model_total[m] for m in models])
    rows = []
    for task in sorted(task_model.keys()):
        coverage = task_model[task]
        covered = [m for m in models if m in coverage]
        if len(covered) < 3:
            continue
        binary = np.array([coverage[m] for m in covered])
        cont = np.array([model_total[m] for m in covered])
        p = float(binary.mean())
        r_pb = point_biserial(binary, cont)
        rows.append({
            "task": task,
            "n_models": len(covered),
            "p_value_difficulty": round(p, 3),
            "r_pb": None if np.isnan(r_pb) else round(r_pb, 3),
            "discriminating": (not np.isnan(r_pb)) and r_pb >= 0.3,
            "pass_rate_by_model": {m: coverage[m] for m in covered},
        })
    return rows


def load_pilot(results_dir: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    """Load pilot item-level correctness and per-model totals."""
    item_model: Dict[str, Dict[str, int]] = defaultdict(dict)
    model_total: Dict[str, float] = {}
    for fname in sorted(os.listdir(results_dir)):
        if not (fname.startswith("pilot_") and fname.endswith("_results.json")):
            continue
        model = fname.replace("pilot_", "").replace("_results.json", "")
        with open(os.path.join(results_dir, fname)) as f:
            rows = json.load(f)
        scores = []
        for r in rows:
            metric_keys = ("primary_accuracy", "dual_task_score", "accuracy", "score")
            val = next((r[k] for k in metric_keys if k in r and isinstance(r[k], (int, float))), None)
            if val is None:
                continue
            item_model[r["task_id"]][model] = 1 if val >= 0.5 else 0
            scores.append(val)
        if scores:
            model_total[model] = float(np.mean(scores))
    return dict(item_model), model_total


def analyze_items(
    item_model: Dict[str, Dict[str, int]],
    model_total: Dict[str, float],
    min_variance: bool = True,
) -> List[Dict]:
    rows = []
    for item, coverage in item_model.items():
        covered = sorted(coverage.keys())
        if len(covered) < 3:
            continue
        binary = np.array([coverage[m] for m in covered])
        cont = np.array([model_total[m] for m in covered])
        if min_variance and (binary.std() == 0):
            continue
        r_pb = point_biserial(binary, cont)
        p = float(binary.mean())
        rows.append({
            "item": item,
            "task_type": item.rsplit("_", 2)[0] if "_" in item else item,
            "n_models": len(covered),
            "p": round(p, 3),
            "r_pb": None if np.isnan(r_pb) else round(r_pb, 3),
        })
    return rows


def aggregate_by_task(item_rows: List[Dict]) -> List[Dict]:
    per_task: Dict[str, List[float]] = defaultdict(list)
    per_task_p: Dict[str, List[float]] = defaultdict(list)
    for r in item_rows:
        if r["r_pb"] is not None:
            per_task[r["task_type"]].append(r["r_pb"])
            per_task_p[r["task_type"]].append(r["p"])
    agg = []
    for t in sorted(per_task):
        vals = per_task[t]
        agg.append({
            "task_type": t,
            "n_items_with_variance": len(vals),
            "mean_r_pb": round(float(np.mean(vals)), 3),
            "median_r_pb": round(float(np.median(vals)), 3),
            "pct_discriminating": round(float(np.mean(np.array(vals) >= 0.3)), 3),
            "mean_p": round(float(np.mean(per_task_p[t])), 3),
        })
    return agg


def render_markdown(task_rows, item_agg, out_path):
    lines = ["# Point-Biserial Discrimination (CTT)\n"]
    lines.append("## Task-level (7-model leaderboard)\n")
    lines.append("`r_pb` is Pearson correlation between task pass/fail across models and model total CAS. `p` is item difficulty (fraction passing). Classical threshold: `r_pb >= 0.3` → discriminating.\n")
    lines.append("| Task | N | p (difficulty) | r_pb | Discriminating |")
    lines.append("|------|---|----------------|------|----------------|")
    for r in task_rows:
        mark = "Yes" if r["discriminating"] else "No"
        rpb = "n/a (no variance)" if r["r_pb"] is None else r["r_pb"]
        lines.append(f"| {r['task']} | {r['n_models']} | {r['p_value_difficulty']} | {rpb} | {mark} |")

    if item_agg:
        lines.append("\n## Item-level (pilot, N=3 models, exploratory)\n")
        lines.append("Item-level `r_pb` is underpowered at N=3 but surfaces items with zero variance (everyone right/wrong).\n")
        lines.append("| Task type | Items w/ variance | Mean r_pb | Median r_pb | % discriminating | Mean p |")
        lines.append("|-----------|-------------------|-----------|-------------|-----------------|--------|")
        for r in item_agg:
            lines.append(f"| {r['task_type']} | {r['n_items_with_variance']} | {r['mean_r_pb']} | {r['median_r_pb']} | {r['pct_discriminating']} | {r['mean_p']} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", default=os.path.join(ROOT, "rameshln_cog-attention_leaderboard (3).csv"))
    parser.add_argument("--pilot-dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "point_biserial.md"))
    parser.add_argument("--json-out", default=os.path.join(ROOT, "results", "point_biserial.json"))
    args = parser.parse_args()

    print(f"Loading leaderboard: {args.leaderboard}")
    task_model, model_total = load_leaderboard(args.leaderboard)
    print(f"  {len(model_total)} models, {len(task_model)} tasks")
    print(f"  Model totals: {model_total}")

    task_rows = analyze_tasks(task_model, model_total)

    item_agg = []
    if os.path.isdir(args.pilot_dir):
        item_model, pilot_total = load_pilot(args.pilot_dir)
        if item_model and pilot_total:
            print(f"Loading pilot: {len(pilot_total)} models, {len(item_model)} items")
            item_rows = analyze_items(item_model, pilot_total)
            item_agg = aggregate_by_task(item_rows)

    render_markdown(task_rows, item_agg, args.out)

    with open(args.json_out, "w") as f:
        json.dump({"task_level": task_rows, "item_level_by_task": item_agg}, f, indent=2)
    print(f"Written: {args.json_out}")


if __name__ == "__main__":
    main()
