"""
Degradation coefficient analysis — power-law fit for selective attention.

Error rate follows E(m) ~ a * m^delta where m is noise_ratio.
delta characterizes how vulnerable a model is to distractors.
"""

import math
import numpy as np
from typing import List, Dict, Optional


def compute_degradation_coefficient(
    results: List,
    task_type: str = "selective",
) -> Dict[str, float]:
    """Fit power-law degradation: ln(error_rate) = ln(a) + delta * ln(noise_ratio).

    Args:
        results: List of ScoreResult objects with noise_ratio in metrics.
        task_type: Filter results to this task type.

    Returns:
        Dict with keys: delta, r_squared, a, n_points.
        Returns delta=0.0, r_squared=0.0 if insufficient data or all-correct.
    """
    filtered = [r for r in results if r.task_type == task_type]
    if not filtered:
        return {"delta": 0.0, "r_squared": 0.0, "a": 0.0, "n_points": 0}

    # Group by noise_ratio, compute mean error rate per bin
    bins: Dict[float, List[float]] = {}
    for r in filtered:
        nr = r.metrics.get("noise_ratio", 0.0)
        if nr <= 0:
            continue
        # error_rate = 1 - primary_score
        primary = r.metrics.get("sas_score", r.metrics.get("accuracy", 0.0))
        error = 1.0 - primary
        bins.setdefault(nr, []).append(error)

    if len(bins) < 2:
        return {"delta": 0.0, "r_squared": 0.0, "a": 0.0, "n_points": len(bins)}

    noise_ratios = []
    error_rates = []
    for nr, errors in sorted(bins.items()):
        mean_err = sum(errors) / len(errors)
        if mean_err <= 0:
            continue  # skip zero-error bins (can't take log)
        noise_ratios.append(nr)
        error_rates.append(mean_err)

    if len(noise_ratios) < 2:
        return {"delta": 0.0, "r_squared": 0.0, "a": 0.0, "n_points": len(noise_ratios)}

    # Log-log linear regression: ln(error) = ln(a) + delta * ln(noise_ratio)
    ln_nr = np.array([math.log(x) for x in noise_ratios])
    ln_err = np.array([math.log(x) for x in error_rates])

    # Least squares fit
    n = len(ln_nr)
    mean_x = ln_nr.mean()
    mean_y = ln_err.mean()
    ss_xy = ((ln_nr - mean_x) * (ln_err - mean_y)).sum()
    ss_xx = ((ln_nr - mean_x) ** 2).sum()

    if ss_xx < 1e-12:
        return {"delta": 0.0, "r_squared": 0.0, "a": 0.0, "n_points": n}

    delta = float(ss_xy / ss_xx)
    ln_a = float(mean_y - delta * mean_x)
    a = math.exp(ln_a)

    # R-squared
    ss_tot = ((ln_err - mean_y) ** 2).sum()
    ss_res = ((ln_err - (ln_a + delta * ln_nr)) ** 2).sum()
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "delta": round(delta, 4),
        "r_squared": round(r_squared, 4),
        "a": round(a, 6),
        "n_points": n,
    }
