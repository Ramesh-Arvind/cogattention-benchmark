"""
Few-Shot Rescue / In-Context Adaptation Rate.

Measures how quickly a model improves when given 1, 3, and 5 examples
of a task it initially fails at. The slope of improvement characterizes
the model's in-context learning rate — a proxy for genuine adaptation
vs static zero-shot capability.

This addresses Chollet's definition of AGI: intelligence is not just
zero-shot performance, but how quickly a system acquires new skills.
"""

import statistics
from typing import List, Dict, Optional, Tuple


def build_few_shot_prompt(
    task_prompt: str,
    examples: List[Dict[str, str]],
) -> str:
    """Build a few-shot prompt by prepending examples to the task.

    Args:
        task_prompt: The original zero-shot task prompt.
        examples: List of {"input": ..., "output": ...} dicts.

    Returns:
        Prompt with examples prepended.
    """
    if not examples:
        return task_prompt

    lines = ["Here are some examples of this task:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Input: {ex['input']}")
        lines.append(f"Output: {ex['output']}")
        lines.append("")

    lines.append("Now complete this task:\n")
    lines.append(task_prompt)
    return "\n".join(lines)


def compute_adaptation_rate(
    zero_shot_score: float,
    few_shot_scores: Dict[int, float],
) -> Dict[str, float]:
    """Compute in-context adaptation rate from zero-shot and few-shot scores.

    Args:
        zero_shot_score: Model accuracy with 0 examples.
        few_shot_scores: {n_examples: accuracy} e.g. {1: 0.4, 3: 0.6, 5: 0.75}

    Returns:
        Dict with:
            adaptation_rate: slope of improvement per example
            max_improvement: best few-shot score - zero-shot score
            rescue_ratio: max few-shot / zero-shot (how much few-shot helps)
            scores: {0: zero_shot, 1: ..., 3: ..., 5: ...}
            adaptation_category: "strong", "moderate", "weak", "none"
    """
    all_scores = {0: zero_shot_score}
    all_scores.update(few_shot_scores)

    sorted_n = sorted(all_scores.keys())
    sorted_scores = [all_scores[n] for n in sorted_n]

    # Linear regression: score = a + b * n_examples
    n = len(sorted_n)
    if n < 2:
        return {
            "adaptation_rate": 0.0,
            "max_improvement": 0.0,
            "rescue_ratio": 1.0,
            "scores": all_scores,
            "adaptation_category": "none",
        }

    x = [float(k) for k in sorted_n]
    y = sorted_scores
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)

    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0

    max_improvement = max(sorted_scores) - zero_shot_score
    rescue_ratio = max(sorted_scores) / max(zero_shot_score, 1e-6)

    # Categorize
    if slope > 0.05:
        category = "strong"
    elif slope > 0.02:
        category = "moderate"
    elif slope > 0.005:
        category = "weak"
    else:
        category = "none"

    return {
        "adaptation_rate": round(slope, 6),
        "max_improvement": round(max_improvement, 4),
        "rescue_ratio": round(rescue_ratio, 4),
        "scores": {k: round(v, 4) for k, v in all_scores.items()},
        "adaptation_category": category,
    }


def compute_model_adaptation_profile(
    results_by_shot: Dict[int, List],
    task_types: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Compute adaptation rate per task type across shot counts.

    Args:
        results_by_shot: {n_examples: [ScoreResult, ...]}
            where 0 = zero-shot, 1 = 1-shot, etc.
        task_types: Filter to these task types. Default: all.

    Returns:
        Dict keyed by task_type, each with adaptation metrics.
    """
    from ..scorers.composite import PRIMARY_METRIC

    # Group scores by task type and shot count
    task_scores: Dict[str, Dict[int, List[float]]] = {}

    for n_shot, results in results_by_shot.items():
        for r in results:
            if task_types and r.task_type not in task_types:
                continue
            metric = PRIMARY_METRIC.get(r.task_type, "accuracy")
            score = r.metrics.get(metric, 0.0)
            task_scores.setdefault(r.task_type, {}).setdefault(n_shot, []).append(score)

    output = {}
    for task_type, shot_scores in task_scores.items():
        # Average scores per shot count
        avg_by_shot = {n: statistics.mean(scores) for n, scores in shot_scores.items()}

        zero = avg_by_shot.pop(0, 0.0)
        output[task_type] = compute_adaptation_rate(zero, avg_by_shot)

    return output


def format_adaptation_report(profile: Dict[str, Dict]) -> str:
    """Format adaptation profile as human-readable report."""
    lines = [
        "=" * 60,
        "  IN-CONTEXT ADAPTATION RATE REPORT",
        "=" * 60,
        "",
    ]

    for task_type, metrics in sorted(profile.items()):
        rate = metrics["adaptation_rate"]
        cat = metrics["adaptation_category"]
        imp = metrics["max_improvement"]
        scores = metrics["scores"]
        score_str = ", ".join(f"{k}-shot={v:.3f}" for k, v in sorted(scores.items()))

        lines.append(f"  {task_type}")
        lines.append(f"    Adaptation rate: {rate:.4f}/example ({cat})")
        lines.append(f"    Max improvement: +{imp:.3f}")
        lines.append(f"    Scores: {score_str}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
