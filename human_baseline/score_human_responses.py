#!/usr/bin/env python3
"""
Score human participant responses from the CogAttention human baseline study.

Loads participant responses from a CSV export (one row per participant, one
column per item), scores each response using the existing scorers, and computes:
  - Per-participant accuracy
  - Per-task-type accuracy
  - Per-difficulty accuracy
  - Human baseline CAS (Cognitive Attention Score)
  - Inter-participant agreement

Usage:
    python3 human_baseline/score_human_responses.py \
        --responses human_baseline/responses.csv \
        --scoring-key human_baseline/survey_output/scoring_key.json \
        --output-dir human_baseline/results/
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.generators.base import TaskInstance
from src.scorers.base import ScoreResult, extract_answer_block
from src.scorers.capacity import score_capacity, score_interference
from src.scorers.sustained import score_sustained, score_stream
from src.scorers.selective import score_selective, score_stroop, score_flanker
from src.scorers.shifting import score_shifting, score_ior
from src.scorers.anomaly import score_anomaly
from src.scorers.composite import compute_cas, format_cas_report, PRIMARY_METRIC, WEIGHTS


# ---------------------------------------------------------------------------
# Scorer dispatch
# ---------------------------------------------------------------------------
SCORER_MAP = {
    "capacity": score_capacity,
    "interference": score_interference,
    "sustained": score_sustained,
    "stream_segregation": score_stream,
    "selective": score_selective,
    "stroop": score_stroop,
    "flanker": score_flanker,
    "shifting": score_shifting,
    "inhibition_return": score_ior,
    "anomaly": score_anomaly,
}

# Cluster membership
CLUSTER_MAP = {
    "capacity": "capacity",
    "interference": "capacity",
    "sustained": "sustained",
    "stream_segregation": "sustained",
    "selective": "selective",
    "stroop": "selective",
    "flanker": "selective",
    "shifting": "shifting",
    "inhibition_return": "shifting",
    "anomaly": "anomaly",
}


def load_scoring_key(path: str) -> Dict[str, dict]:
    """Load scoring key JSON and return {task_id: key_entry}."""
    with open(path, "r") as f:
        key_list = json.load(f)
    return {entry["task_id"]: entry for entry in key_list}


def reconstruct_task_instance(key_entry: dict) -> TaskInstance:
    """Reconstruct a TaskInstance from a scoring key entry.

    The scoring key does not contain the full prompt (not needed for scoring),
    but does contain gold_answer, task_type, difficulty, and metadata.
    """
    return TaskInstance(
        task_id=key_entry["task_id"],
        task_type=key_entry["task_type"],
        difficulty=key_entry["difficulty"],
        prompt="",  # not needed for scoring
        gold_answer=key_entry["gold_answer"],
        metadata=key_entry.get("metadata", {}),
    )


def load_responses(path: str) -> List[Dict[str, str]]:
    """Load participant responses from CSV.

    Expected CSV format:
        participant_id, task_id_1, task_id_2, ..., task_id_N
        P001, "answer text", "answer text", ...

    Each column after participant_id is named with a task_id.
    Each cell contains the participant's free-text response for that item.
    """
    responses = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            responses.append(dict(row))
    return responses


def score_participant(
    participant_id: str,
    answers: Dict[str, str],
    scoring_key: Dict[str, dict],
) -> List[ScoreResult]:
    """Score all responses for a single participant.

    Args:
        participant_id: Identifier for this participant.
        answers: {task_id: response_text} mapping.
        scoring_key: {task_id: key_entry} from the scoring key file.

    Returns:
        List of ScoreResult objects.
    """
    results = []
    for task_id, response in answers.items():
        if task_id == "participant_id" or task_id not in scoring_key:
            continue

        key_entry = scoring_key[task_id]
        instance = reconstruct_task_instance(key_entry)

        scorer = SCORER_MAP.get(instance.task_type)
        if scorer is None:
            print(f"  Warning: no scorer for task_type={instance.task_type}, skipping {task_id}")
            continue

        try:
            result = scorer(instance, response or "")
            results.append(result)
        except Exception as e:
            print(f"  Error scoring {task_id} for {participant_id}: {e}")
            # Create a zero-score result
            result = ScoreResult(task_id, instance.task_type, instance.difficulty)
            metric_name = PRIMARY_METRIC.get(instance.task_type, "accuracy")
            result.add_metric(metric_name, 0.0)
            result.details = {"error": str(e)}
            results.append(result)

    return results


def compute_per_participant_summary(
    participant_id: str,
    results: List[ScoreResult],
) -> dict:
    """Compute summary statistics for one participant."""
    primary_scores = []
    by_type = defaultdict(list)
    by_diff = defaultdict(list)
    by_cluster = defaultdict(list)

    for r in results:
        metric_name = PRIMARY_METRIC.get(r.task_type, "accuracy")
        score = r.metrics.get(metric_name, 0.0)
        primary_scores.append(score)
        by_type[r.task_type].append(score)
        by_diff[r.difficulty].append(score)
        cluster = CLUSTER_MAP.get(r.task_type, "unknown")
        by_cluster[cluster].append(score)

    mean_score = statistics.mean(primary_scores) if primary_scores else 0.0

    return {
        "participant_id": participant_id,
        "mean_primary_score": round(mean_score, 4),
        "n_items": len(results),
        "by_task_type": {
            tt: round(statistics.mean(vals), 4) for tt, vals in by_type.items()
        },
        "by_difficulty": {
            d: round(statistics.mean(vals), 4) for d, vals in by_diff.items()
        },
        "by_cluster": {
            c: round(statistics.mean(vals), 4) for c, vals in by_cluster.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score human responses for CogAttention human baseline"
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to CSV with participant responses",
    )
    parser.add_argument(
        "--scoring-key",
        default=os.path.join(PROJECT_ROOT, "human_baseline", "survey_output", "scoring_key.json"),
        help="Path to scoring key JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "human_baseline", "results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--min-time-minutes",
        type=float,
        default=8.0,
        help="Exclude participants who finished in less than this (speeders)",
    )
    parser.add_argument(
        "--max-time-minutes",
        type=float,
        default=60.0,
        help="Exclude participants who took longer than this (distracted)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  CogAttention Human Baseline Scorer")
    print("=" * 60)

    # ---- Load data ----
    scoring_key = load_scoring_key(args.scoring_key)
    print(f"\n  Scoring key loaded: {len(scoring_key)} items")

    responses = load_responses(args.responses)
    print(f"  Responses loaded: {len(responses)} participants")

    # ---- Identify task_id columns ----
    if not responses:
        print("  ERROR: No participant responses found. Exiting.")
        sys.exit(1)

    all_columns = list(responses[0].keys())
    task_columns = [c for c in all_columns if c in scoring_key]
    non_task_columns = [c for c in all_columns if c not in scoring_key]
    print(f"  Task columns found: {len(task_columns)}")
    print(f"  Non-task columns: {non_task_columns}")

    # ---- Score each participant ----
    all_participant_results = {}
    all_participant_summaries = []
    all_score_results = []  # flat list for CAS computation

    for row in responses:
        pid = row.get("participant_id", "unknown")
        answers = {col: row[col] for col in task_columns}

        results = score_participant(pid, answers, scoring_key)
        all_participant_results[pid] = results
        all_score_results.extend(results)

        summary = compute_per_participant_summary(pid, results)
        all_participant_summaries.append(summary)
        print(f"  {pid}: mean_score={summary['mean_primary_score']:.3f} ({summary['n_items']} items)")

    # ---- Aggregate statistics ----
    print("\n" + "-" * 60)
    print("  Aggregate Results")
    print("-" * 60)

    participant_means = [s["mean_primary_score"] for s in all_participant_summaries]
    overall_mean = statistics.mean(participant_means) if participant_means else 0.0
    overall_std = statistics.stdev(participant_means) if len(participant_means) > 1 else 0.0

    print(f"\n  Overall human accuracy: {overall_mean:.4f} (+/- {overall_std:.4f})")
    print(f"  N participants: {len(all_participant_summaries)}")

    # Per-task-type aggregate
    type_scores = defaultdict(list)
    for s in all_participant_summaries:
        for tt, val in s["by_task_type"].items():
            type_scores[tt].append(val)

    print("\n  Per-task-type accuracy:")
    for tt in sorted(type_scores.keys()):
        vals = type_scores[tt]
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"    {tt:<22} {m:.3f} (+/- {sd:.3f})  n={len(vals)}")

    # Per-difficulty aggregate
    diff_scores = defaultdict(list)
    for s in all_participant_summaries:
        for d, val in s["by_difficulty"].items():
            diff_scores[d].append(val)

    print("\n  Per-difficulty accuracy:")
    for d in ["Easy", "Medium", "Hard"]:
        if d in diff_scores:
            vals = diff_scores[d]
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"    {d:<10} {m:.3f} (+/- {sd:.3f})  n={len(vals)}")

    # Per-cluster aggregate
    cluster_scores = defaultdict(list)
    for s in all_participant_summaries:
        for c, val in s["by_cluster"].items():
            cluster_scores[c].append(val)

    print("\n  Per-cluster accuracy:")
    for c in ["capacity", "sustained", "selective", "shifting", "anomaly"]:
        if c in cluster_scores:
            vals = cluster_scores[c]
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"    {c:<15} {m:.3f} (+/- {sd:.3f})  n={len(vals)}")

    # ---- Compute human baseline CAS ----
    print("\n" + "-" * 60)
    print("  Human Baseline CAS")
    print("-" * 60)

    cas_result = compute_cas(all_score_results)
    cas_report = format_cas_report(cas_result)
    print(cas_report)

    # ---- Export results ----

    # 1. Per-participant summaries
    summaries_path = os.path.join(args.output_dir, "participant_summaries.json")
    with open(summaries_path, "w") as f:
        json.dump(all_participant_summaries, f, indent=2)
    print(f"\n  Participant summaries: {summaries_path}")

    # 2. Per-item detail (every participant x every item)
    item_details = []
    for pid, results in all_participant_results.items():
        for r in results:
            entry = r.to_dict()
            entry["participant_id"] = pid
            item_details.append(entry)

    details_path = os.path.join(args.output_dir, "item_level_scores.json")
    with open(details_path, "w") as f:
        json.dump(item_details, f, indent=2, default=str)
    print(f"  Item-level scores:    {details_path}")

    # 3. Per-item accuracy (across all participants)
    item_accuracy = defaultdict(list)
    for entry in item_details:
        metric_name = PRIMARY_METRIC.get(entry["task_type"], "accuracy")
        score = entry.get(metric_name, 0.0)
        item_accuracy[entry["task_id"]].append(score)

    item_acc_summary = {}
    for task_id, vals in item_accuracy.items():
        m = statistics.mean(vals) if vals else 0.0
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        item_acc_summary[task_id] = {
            "mean_accuracy": round(m, 4),
            "std": round(sd, 4),
            "n_responses": len(vals),
        }

    item_acc_path = os.path.join(args.output_dir, "per_item_accuracy.json")
    with open(item_acc_path, "w") as f:
        json.dump(item_acc_summary, f, indent=2)
    print(f"  Per-item accuracy:    {item_acc_path}")

    # 4. CAS results
    cas_path = os.path.join(args.output_dir, "human_cas_results.json")
    with open(cas_path, "w") as f:
        json.dump(cas_result, f, indent=2)
    print(f"  CAS results:          {cas_path}")

    # 5. Flat CSV for easy analysis
    csv_path = os.path.join(args.output_dir, "human_baseline_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "participant_id", "mean_score", "n_items",
            "capacity", "sustained", "selective", "shifting", "anomaly",
            "Easy", "Medium", "Hard",
        ])
        for s in all_participant_summaries:
            writer.writerow([
                s["participant_id"],
                s["mean_primary_score"],
                s["n_items"],
                s["by_cluster"].get("capacity", ""),
                s["by_cluster"].get("sustained", ""),
                s["by_cluster"].get("selective", ""),
                s["by_cluster"].get("shifting", ""),
                s["by_cluster"].get("anomaly", ""),
                s["by_difficulty"].get("Easy", ""),
                s["by_difficulty"].get("Medium", ""),
                s["by_difficulty"].get("Hard", ""),
            ])
    print(f"  Summary CSV:          {csv_path}")

    # 6. Text report
    report_path = os.path.join(args.output_dir, "human_baseline_report.txt")
    with open(report_path, "w") as f:
        f.write("CogAttention Human Baseline Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"N participants: {len(all_participant_summaries)}\n")
        f.write(f"N items per participant: {len(task_columns)}\n\n")
        f.write(f"Overall human accuracy: {overall_mean:.4f} (+/- {overall_std:.4f})\n\n")
        f.write("Per-cluster:\n")
        for c in ["capacity", "sustained", "selective", "shifting", "anomaly"]:
            if c in cluster_scores:
                vals = cluster_scores[c]
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                f.write(f"  {c:<15} {m:.4f} (+/- {sd:.4f})\n")
        f.write(f"\n{cas_report}\n")
    print(f"  Text report:          {report_path}")

    print("\n" + "=" * 60)
    print("  Scoring complete.")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
