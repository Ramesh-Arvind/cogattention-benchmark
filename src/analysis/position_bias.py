"""
Position bias analysis — U-shaped attention pattern detection.

Bins items by target position in context and reports accuracy per bin,
following Liu et al. (2024) "Lost in the Middle".
"""

from typing import List, Dict
import statistics


def compute_position_bias(
    results: List,
    task_types: List[str] = None,
    n_bins: int = 5,
) -> Dict[str, object]:
    """Compute accuracy per position bin across results.

    Args:
        results: List of ScoreResult objects. Each should have a
            'target_position_ratio' in metrics (0.0=start, 1.0=end)
            or 'quintile_targets' in details.
        task_types: Filter to these task types. Default: all.
        n_bins: Number of position bins (default 5 = quintiles).

    Returns:
        Dict with:
            bins: list of {bin_label, accuracy_mean, accuracy_std, n}
            has_u_shape: bool (first & last bins > middle bins)
    """
    if task_types is not None:
        filtered = [r for r in results if r.task_type in task_types]
    else:
        filtered = list(results)

    if not filtered:
        return {"bins": [], "has_u_shape": False}

    # Collect (position_ratio, score) pairs
    pairs = []
    for r in filtered:
        pos = r.metrics.get("target_position_ratio")
        if pos is None:
            pos = r.details.get("target_position_ratio")
        if pos is None:
            continue

        primary = r.metrics.get("accuracy", r.metrics.get("recall", r.metrics.get("sas_score", 0.0)))
        pairs.append((float(pos), float(primary)))

    if not pairs:
        return {"bins": [], "has_u_shape": False}

    # Bin by position
    bin_width = 1.0 / n_bins
    bin_labels = []
    for i in range(n_bins):
        lo = round(i * bin_width, 2)
        hi = round((i + 1) * bin_width, 2)
        bin_labels.append(f"{lo:.0%}-{hi:.0%}")

    bin_scores: Dict[int, List[float]] = {i: [] for i in range(n_bins)}
    for pos, score in pairs:
        b = min(int(pos / bin_width), n_bins - 1)
        bin_scores[b].append(score)

    bins = []
    for i in range(n_bins):
        scores = bin_scores[i]
        if scores:
            bins.append({
                "bin_label": bin_labels[i],
                "accuracy_mean": round(statistics.mean(scores), 4),
                "accuracy_std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
                "n": len(scores),
            })
        else:
            bins.append({
                "bin_label": bin_labels[i],
                "accuracy_mean": None,
                "accuracy_std": None,
                "n": 0,
            })

    # Detect U-shape: first and last bin means > average of middle bins
    has_u_shape = False
    valid_means = [(i, b["accuracy_mean"]) for i, b in enumerate(bins) if b["accuracy_mean"] is not None]
    if len(valid_means) >= 3:
        first_mean = valid_means[0][1]
        last_mean = valid_means[-1][1]
        middle_means = [m for i, m in valid_means[1:-1]]
        if middle_means:
            avg_middle = sum(middle_means) / len(middle_means)
            has_u_shape = first_mean > avg_middle and last_mean > avg_middle

    return {
        "bins": bins,
        "has_u_shape": has_u_shape,
    }
