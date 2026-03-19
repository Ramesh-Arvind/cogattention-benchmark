"""
Bootstrapped confidence intervals for CAS and per-task scores.
"""

import numpy as np
from typing import List, Dict


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrapped confidence interval for the mean.

    Args:
        values: List of observed values.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper.
    """
    if not values:
        return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    arr = np.array(values, dtype=float)
    point_estimate = float(arr.mean())

    if len(arr) == 1:
        return {"point_estimate": point_estimate, "ci_lower": point_estimate, "ci_upper": point_estimate}

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": round(point_estimate, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
    }


def bootstrap_cas_ci(
    results: List,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrapped CIs for CAS and per-task scores.

    Args:
        results: List of ScoreResult objects.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with 'cas' key and per-task-type keys, each containing
        point_estimate, ci_lower, ci_upper.
    """
    from ..scorers.composite import compute_cas, PRIMARY_METRIC, WEIGHTS

    # Per-task CIs
    by_type: Dict[str, List[float]] = {}
    for r in results:
        metric_name = PRIMARY_METRIC.get(r.task_type, "accuracy")
        val = r.metrics.get(metric_name, 0.0)
        by_type.setdefault(r.task_type, []).append(val)

    output = {}
    for task_type, values in by_type.items():
        output[task_type] = bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)

    # CAS-level bootstrap: resample results, recompute CAS each time
    rng = np.random.RandomState(seed)
    n = len(results)
    cas_boots = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [results[j] for j in indices]
        cas_result = compute_cas(sample)
        cas_boots[i] = cas_result["cas_score"]

    cas_point = float(compute_cas(results)["cas_score"])
    ci_lower = float(np.percentile(cas_boots, 2.5))
    ci_upper = float(np.percentile(cas_boots, 97.5))

    output["cas"] = {
        "point_estimate": round(cas_point, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
    }

    return output
