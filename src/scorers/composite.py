"""
Composite Cognitive Attention Score (CAS) aggregator.

Combines per-task scores into a single cognitive profile.
14 task types with per-task weights (see WEIGHTS dict below).
Grouped by ability:
  - Capacity (capacity + interference + blink): 0.17
  - Sustained (sustained + stream + dilution + niah + multihop): 0.38
  - Selective (selective + stroop + flanker): 0.18
  - Shifting (shifting + inhibition_return): 0.14
  - Stimulus-driven (anomaly): 0.13
"""

import math
from typing import List, Dict
from .base import ScoreResult
import statistics


WEIGHTS = {
    "capacity": 0.07,
    "interference": 0.07,
    "blink": 0.03,
    "sustained": 0.07,
    "stream_segregation": 0.07,
    "context_dilution": 0.08,
    "semantic_niah": 0.08,
    "multihop": 0.08,
    "selective": 0.07,
    "stroop": 0.07,
    "flanker": 0.04,
    "shifting": 0.07,
    "inhibition_return": 0.07,
    "anomaly": 0.13,
}

# Primary metric per task type
PRIMARY_METRIC = {
    "capacity": "accuracy",
    "interference": "accuracy",
    "blink": "accuracy",
    "sustained": "recall",
    "stream_segregation": "stream_accuracy",
    "context_dilution": "accuracy",
    "semantic_niah": "accuracy",
    "multihop": "accuracy",
    "selective": "sas_score",
    "stroop": "accuracy",
    "flanker": "accuracy",
    "shifting": "overall_accuracy",
    "inhibition_return": "overall_accuracy",
    "anomaly": "dual_task_score",
    "visual_stroop": "accuracy",
    "visual_inattentional": "ib_composite",
}


def _geometric_weighted_mean(scores: Dict[str, float], weights: Dict[str, float], epsilon: float = 1e-6) -> float:
    """Compute weighted geometric mean of scores.

    Geometric mean penalizes zero scores — a single zero tanks the composite.
    This makes the metric non-compensatory: a model cannot hide a weakness
    by excelling elsewhere.

    Args:
        scores: {task_type: score} where score in [0, 1].
        weights: {task_type: weight}.
        epsilon: Small value added to scores to avoid log(0).

    Returns:
        Weighted geometric mean in [0, 1].
    """
    log_sum = 0.0
    weight_sum = 0.0
    for task_type, score in scores.items():
        w = weights.get(task_type, 0.0)
        if w <= 0 or task_type not in weights:
            continue
        log_sum += w * math.log(max(score, epsilon))
        weight_sum += w

    if weight_sum <= 0:
        return 0.0
    return math.exp(log_sum / weight_sum)


def compute_cas(results: List[ScoreResult]) -> Dict:
    """Compute the Cognitive Attention Score from individual task results."""
    # Group by task type
    by_type = {}
    for r in results:
        by_type.setdefault(r.task_type, []).append(r)

    # Compute mean primary metric per task type
    type_scores = {}
    type_details = {}
    for task_type, task_results in by_type.items():
        metric_name = PRIMARY_METRIC.get(task_type, "accuracy")
        values = [r.metrics.get(metric_name, 0.0) for r in task_results]
        mean_val = statistics.mean(values) if values else 0.0
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        type_scores[task_type] = mean_val

        # Per-difficulty breakdown
        by_diff = {}
        for r in task_results:
            by_diff.setdefault(r.difficulty, []).append(
                r.metrics.get(metric_name, 0.0)
            )
        diff_means = {d: statistics.mean(v) for d, v in by_diff.items()}

        type_details[task_type] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "n": len(values),
            "by_difficulty": {d: round(v, 4) for d, v in diff_means.items()},
            "metric_used": metric_name,
        }

    # Weighted composite
    cas = 0.0
    total_weight = 0.0
    for task_type, weight in WEIGHTS.items():
        if task_type in type_scores:
            cas += weight * type_scores[task_type]
            total_weight += weight

    # Normalize if not all task types present
    if total_weight > 0 and total_weight < 1.0:
        cas = cas / total_weight

    # Additional aggregate metrics
    all_metrics = {}
    for task_type, task_results in by_type.items():
        for r in task_results:
            for metric_name, metric_val in r.metrics.items():
                key = f"{task_type}_{metric_name}"
                all_metrics.setdefault(key, []).append(metric_val)

    aggregate = {
        metric: round(statistics.mean(vals), 4)
        for metric, vals in all_metrics.items()
    }

    # Geometric mean CAS (non-compensatory: zero on any task tanks composite)
    cas_geo = _geometric_weighted_mean(type_scores, WEIGHTS)

    return {
        "cas_score": round(cas, 4),
        "cas_geometric": round(cas_geo, 4),
        "per_task": type_details,
        "aggregate_metrics": aggregate,
        "total_instances": len(results),
        "task_types_evaluated": list(by_type.keys()),
    }


def format_cas_report(cas_result: Dict) -> str:
    """Format CAS results as a human-readable report."""
    lines = [
        "=" * 60,
        "  COGNITIVE ATTENTION SCORE (CAS) REPORT",
        "=" * 60,
        f"\n  CAS Score: {cas_result['cas_score']:.4f} / 1.0000",
        f"  Instances evaluated: {cas_result['total_instances']}",
        f"  Task types: {len(cas_result['task_types_evaluated'])}",
        "\n  Per-Task Breakdown:",
        "-" * 60,
    ]

    for task_type, details in cas_result["per_task"].items():
        weight = WEIGHTS.get(task_type, 0)
        lines.append(
            f"  {task_type:<22} "
            f"mean={details['mean']:.3f} "
            f"(±{details['std']:.3f}) "
            f"weight={weight:.3f} "
            f"n={details['n']}"
        )
        for diff, val in sorted(details["by_difficulty"].items()):
            lines.append(f"    {diff:<10} {val:.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)
