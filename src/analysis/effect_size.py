"""
Effect size analysis between models.

Computes Cohen's d and rank-biserial correlation for pairwise model comparisons.
"""

import math
import statistics
from typing import List, Dict, Tuple

from ..scorers.composite import PRIMARY_METRIC


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns positive d when group_a > group_b.

    Args:
        group_a: Scores for model A.
        group_b: Scores for model B.

    Returns:
        Cohen's d (positive = A better, negative = B better).
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1e-12

    return (mean_a - mean_b) / pooled_sd


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def rank_biserial(group_a: List[float], group_b: List[float]) -> float:
    """Compute rank-biserial correlation (non-parametric effect size).

    Proportion of pairwise comparisons where A > B, scaled to [-1, 1].
    """
    if not group_a or not group_b:
        return 0.0

    wins = sum(1 for a in group_a for b in group_b if a > b)
    ties = sum(1 for a in group_a for b in group_b if a == b)
    total = len(group_a) * len(group_b)

    return (wins + 0.5 * ties - total / 2) / (total / 2) if total > 0 else 0.0


def compute_pairwise_effects(
    results_by_model: Dict[str, List],
) -> Dict[str, Dict[str, object]]:
    """Compute pairwise effect sizes between all model pairs.

    Args:
        results_by_model: {model_name: [ScoreResult, ...]}.

    Returns:
        Dict keyed by "modelA_vs_modelB", each containing:
            overall_d, overall_interpretation, per_task: {task: {d, interpretation, rbc}}
    """
    models = sorted(results_by_model.keys())
    output = {}

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a, model_b = models[i], models[j]
            results_a = results_by_model[model_a]
            results_b = results_by_model[model_b]

            # Group by task type
            by_task_a: Dict[str, List[float]] = {}
            by_task_b: Dict[str, List[float]] = {}
            for r in results_a:
                metric = PRIMARY_METRIC.get(r.task_type, "accuracy")
                by_task_a.setdefault(r.task_type, []).append(r.metrics.get(metric, 0.0))
            for r in results_b:
                metric = PRIMARY_METRIC.get(r.task_type, "accuracy")
                by_task_b.setdefault(r.task_type, []).append(r.metrics.get(metric, 0.0))

            # Per-task effects
            per_task = {}
            all_tasks = set(by_task_a.keys()) | set(by_task_b.keys())
            for task in sorted(all_tasks):
                scores_a = by_task_a.get(task, [])
                scores_b = by_task_b.get(task, [])
                d = cohens_d(scores_a, scores_b)
                rbc = rank_biserial(scores_a, scores_b)
                per_task[task] = {
                    "cohens_d": round(d, 4),
                    "interpretation": interpret_cohens_d(d),
                    "rank_biserial": round(rbc, 4),
                }

            # Overall effect using all primary metric scores
            all_a = [v for vals in by_task_a.values() for v in vals]
            all_b = [v for vals in by_task_b.values() for v in vals]
            overall_d = cohens_d(all_a, all_b)

            key = f"{model_a}_vs_{model_b}"
            output[key] = {
                "model_a": model_a,
                "model_b": model_b,
                "overall_cohens_d": round(overall_d, 4),
                "overall_interpretation": interpret_cohens_d(overall_d),
                "overall_rank_biserial": round(rank_biserial(all_a, all_b), 4),
                "per_task": per_task,
            }

    return output
