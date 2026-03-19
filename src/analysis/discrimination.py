#!/usr/bin/env python3
"""Post-hoc discrimination analysis for pilot evaluation results.

Usage:
    python3 -m src.analysis.discrimination --results-dir results/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

from src.generators.base import DIFFICULTY_LEVELS
from src.scorers.composite import PRIMARY_METRIC

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")


def load_all_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load per-model result files. Returns {model_name: [result_dicts]}."""
    model_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("pilot_") and fname.endswith("_results.json"):
            model_name = fname.replace("pilot_", "").replace("_results.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                model_results[model_name] = json.load(f)
    return model_results


def analyze_difficulty_gradient(model_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Check that accuracy degrades Easy > Medium > Hard > Expert per task."""
    report = {}
    for model, results in model_results.items():
        task_diff: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            task_type = r["task_type"]
            difficulty = r["difficulty"]
            metric = PRIMARY_METRIC.get(task_type, "accuracy")
            if metric in r:
                task_diff[task_type][difficulty].append(r[metric])

        model_report = {}
        for task_type, diffs in sorted(task_diff.items()):
            means = {}
            for d in DIFFICULTY_LEVELS:
                vals = diffs.get(d, [])
                means[d] = round(sum(vals) / len(vals), 4) if vals else None

            # Check monotonic decrease
            valid_means = [means[d] for d in DIFFICULTY_LEVELS if means[d] is not None]
            monotonic = all(valid_means[i] >= valid_means[i + 1] for i in range(len(valid_means) - 1))

            model_report[task_type] = {
                "means_by_difficulty": means,
                "monotonic_decrease": monotonic,
                "gradient": round(valid_means[0] - valid_means[-1], 4) if len(valid_means) >= 2 else 0.0,
            }
        report[model] = model_report
    return report


def analyze_cross_model_spread(model_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Flag tasks where cross-model spread < 0.1 (insufficient discrimination)."""
    # Aggregate per-task mean by model
    task_model_means: Dict[str, Dict[str, float]] = defaultdict(dict)
    for model, results in model_results.items():
        task_scores: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            metric = PRIMARY_METRIC.get(r["task_type"], "accuracy")
            if metric in r:
                task_scores[r["task_type"]].append(r[metric])
        for task_type, scores in task_scores.items():
            task_model_means[task_type][model] = round(sum(scores) / len(scores), 4)

    report = {}
    for task_type, model_means in sorted(task_model_means.items()):
        vals = list(model_means.values())
        spread = round(max(vals) - min(vals), 4) if len(vals) >= 2 else 0.0
        report[task_type] = {
            "model_means": model_means,
            "spread": spread,
            "sufficient": spread >= 0.1,
        }
    return report


def detect_ceiling_floor(model_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Detect tasks where best model > 0.95 (ceiling) or worst model < 0.05 (floor)."""
    task_model_means: Dict[str, Dict[str, float]] = defaultdict(dict)
    for model, results in model_results.items():
        task_scores: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            metric = PRIMARY_METRIC.get(r["task_type"], "accuracy")
            if metric in r:
                task_scores[r["task_type"]].append(r[metric])
        for task_type, scores in task_scores.items():
            task_model_means[task_type][model] = round(sum(scores) / len(scores), 4)

    report = {}
    for task_type, model_means in sorted(task_model_means.items()):
        best = max(model_means.values())
        worst = min(model_means.values())
        report[task_type] = {
            "best_model_mean": best,
            "worst_model_mean": worst,
            "ceiling_hit": best > 0.95,
            "floor_hit": worst < 0.05,
        }
    return report


def generate_report(
    gradient: Dict,
    spread: Dict,
    ceiling_floor: Dict,
    output_dir: str,
):
    """Write discrimination_analysis.md and pilot_metrics.json."""
    # Markdown report
    lines = ["# Discrimination Analysis Report\n"]

    lines.append("## 1. Difficulty Gradient (per model)\n")
    for model, tasks in gradient.items():
        lines.append(f"### Model: `{model}`\n")
        lines.append("| Task | Easy | Medium | Hard | Expert | Monotonic | Gradient |")
        lines.append("|------|------|--------|------|--------|-----------|----------|")
        for task, info in sorted(tasks.items()):
            m = info["means_by_difficulty"]
            mono = "Yes" if info["monotonic_decrease"] else "**NO**"
            grad = info["gradient"]
            lines.append(
                f"| {task} | {m.get('Easy', '-')} | {m.get('Medium', '-')} | "
                f"{m.get('Hard', '-')} | {m.get('Expert', '-')} | {mono} | {grad} |"
            )
        lines.append("")

    lines.append("## 2. Cross-Model Spread\n")
    lines.append("| Task | Spread | Sufficient (>=0.1) | Model Scores |")
    lines.append("|------|--------|---------------------|--------------|")
    for task, info in sorted(spread.items()):
        suf = "Yes" if info["sufficient"] else "**NO**"
        scores_str = ", ".join(f"{m}={v}" for m, v in info["model_means"].items())
        lines.append(f"| {task} | {info['spread']} | {suf} | {scores_str} |")
    lines.append("")

    lines.append("## 3. Ceiling / Floor Detection\n")
    lines.append("| Task | Best Model Mean | Worst Model Mean | Ceiling (>0.95) | Floor (<0.05) |")
    lines.append("|------|----------------|-----------------|-----------------|---------------|")
    for task, info in sorted(ceiling_floor.items()):
        ceil = "**YES**" if info["ceiling_hit"] else "No"
        floor = "**YES**" if info["floor_hit"] else "No"
        lines.append(
            f"| {task} | {info['best_model_mean']} | {info['worst_model_mean']} | {ceil} | {floor} |"
        )
    lines.append("")

    # Actionable summary
    lines.append("## 4. Actionable Flags\n")
    flags = []
    for task, info in spread.items():
        if not info["sufficient"]:
            flags.append(f"- **{task}**: Cross-model spread {info['spread']} < 0.1 — task may not discriminate between models.")
    for task, info in ceiling_floor.items():
        if info["ceiling_hit"]:
            flags.append(f"- **{task}**: Ceiling hit (best={info['best_model_mean']}) — consider harder instances.")
        if info["floor_hit"]:
            flags.append(f"- **{task}**: Floor hit (worst={info['worst_model_mean']}) — consider easier instances or scoring recalibration.")
    for model, tasks in gradient.items():
        for task, info in tasks.items():
            if not info["monotonic_decrease"] and info["gradient"] < 0.05:
                flags.append(f"- **{task}** ({model}): Non-monotonic difficulty with flat gradient ({info['gradient']}).")

    if flags:
        lines.extend(flags)
    else:
        lines.append("No critical flags. All tasks show adequate discrimination.")

    lines.append("")

    md_path = os.path.join(output_dir, "discrimination_analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {md_path}")

    # JSON metrics
    metrics = {
        "difficulty_gradient": gradient,
        "cross_model_spread": spread,
        "ceiling_floor": ceiling_floor,
    }
    json_path = os.path.join(output_dir, "pilot_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Written: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Discrimination analysis for pilot results")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    model_results = load_all_results(args.results_dir)
    if not model_results:
        print(f"No pilot result files found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded results for models: {list(model_results.keys())}")

    gradient = analyze_difficulty_gradient(model_results)
    spread = analyze_cross_model_spread(model_results)
    ceiling_floor = detect_ceiling_floor(model_results)

    generate_report(gradient, spread, ceiling_floor, args.results_dir)


if __name__ == "__main__":
    main()
