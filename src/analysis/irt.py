"""
Item Response Theory (IRT) analysis for the CogAttention benchmark.

Fits 2PL IRT models to extract item difficulty and discrimination parameters.
Uses scipy.optimize as fallback when girth is not installed.
"""

import warnings
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _neg_log_likelihood_2pl(
    params: np.ndarray,
    response_matrix: np.ndarray,
) -> float:
    """Negative log-likelihood for 2PL IRT model.

    params layout: [theta_1..theta_M, a_1..a_N, b_1..b_N]
    where M = n_models, N = n_items.
    """
    n_models, n_items = response_matrix.shape
    theta = params[:n_models]
    a = params[n_models:n_models + n_items]
    b = params[n_models + n_items:]

    # P(correct) = sigmoid(a_j * (theta_i - b_j))
    # shape: (n_models, n_items)
    logit = a[None, :] * (theta[:, None] - b[None, :])
    p = _sigmoid(logit)
    p = np.clip(p, 1e-8, 1 - 1e-8)

    # Binary cross-entropy
    ll = (response_matrix * np.log(p) + (1 - response_matrix) * np.log(1 - p)).sum()
    return -ll


def fit_irt_model(
    response_matrix: np.ndarray,
    model_names: Optional[List[str]] = None,
    item_ids: Optional[List[str]] = None,
    model_type: str = "2PL",
    max_iter: int = 2000,
) -> Dict[str, object]:
    """Fit a 2PL IRT model to a response matrix.

    Args:
        response_matrix: (n_models x n_items) array of scores in [0, 1].
            Binary (0/1) is ideal but continuous works as approximate.
        model_names: Labels for rows. Default: Model_0, Model_1, ...
        item_ids: Labels for columns. Default: Item_0, Item_1, ...
        model_type: "2PL" (discrimination + difficulty). "1PL" constrains a=1.
        max_iter: Max optimizer iterations.

    Returns:
        Dict with:
            theta: {model_name: ability_estimate}
            difficulty: {item_id: b_parameter}
            discrimination: {item_id: a_parameter}
            convergence: bool
            neg_log_likelihood: float
    """
    n_models, n_items = response_matrix.shape

    if model_names is None:
        model_names = [f"Model_{i}" for i in range(n_models)]
    if item_ids is None:
        item_ids = [f"Item_{j}" for j in range(n_items)]

    # Initialize parameters
    # theta: from mean score per model, z-scored
    mean_scores = response_matrix.mean(axis=1)
    theta_init = (mean_scores - mean_scores.mean()) / (mean_scores.std() + 1e-8)

    # b (difficulty): from mean score per item, inverted
    mean_item = response_matrix.mean(axis=0)
    b_init = -(mean_item - mean_item.mean()) / (mean_item.std() + 1e-8)

    # a (discrimination): start at 1.0
    a_init = np.ones(n_items)

    if model_type == "1PL":
        a_init = np.ones(n_items)  # fixed

    params0 = np.concatenate([theta_init, a_init, b_init])

    # Bounds: a > 0.1 for stability
    bounds = (
        [(None, None)] * n_models +  # theta unconstrained
        [(0.1, 5.0)] * n_items +     # a in [0.1, 5.0]
        [(None, None)] * n_items      # b unconstrained
    )

    if model_type == "1PL":
        # Fix a=1 by tightening bounds
        bounds = (
            [(None, None)] * n_models +
            [(1.0, 1.0)] * n_items +
            [(None, None)] * n_items
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            _neg_log_likelihood_2pl,
            params0,
            args=(response_matrix,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-8},
        )

    theta = result.x[:n_models]
    a = result.x[n_models:n_models + n_items]
    b = result.x[n_models + n_items:]

    return {
        "theta": {name: round(float(t), 4) for name, t in zip(model_names, theta)},
        "difficulty": {iid: round(float(bi), 4) for iid, bi in zip(item_ids, b)},
        "discrimination": {iid: round(float(ai), 4) for iid, ai in zip(item_ids, a)},
        "convergence": result.success,
        "neg_log_likelihood": round(float(result.fun), 4),
        "model_type": model_type,
    }


def build_response_matrix(
    results_by_model: Dict[str, List],
    primary_metric_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a response matrix from per-model ScoreResult lists.

    Args:
        results_by_model: {model_name: [ScoreResult, ...]}.
        primary_metric_map: {task_type: metric_name}. Default uses PRIMARY_METRIC.

    Returns:
        (response_matrix, model_names, item_ids)
    """
    if primary_metric_map is None:
        from ..scorers.composite import PRIMARY_METRIC
        primary_metric_map = PRIMARY_METRIC

    model_names = sorted(results_by_model.keys())

    # Collect all unique item IDs
    all_item_ids = set()
    for results in results_by_model.values():
        for r in results:
            all_item_ids.add(r.task_id)
    item_ids = sorted(all_item_ids)
    item_to_idx = {iid: j for j, iid in enumerate(item_ids)}

    matrix = np.zeros((len(model_names), len(item_ids)))

    for i, model in enumerate(model_names):
        for r in results_by_model[model]:
            j = item_to_idx.get(r.task_id)
            if j is not None:
                metric = primary_metric_map.get(r.task_type, "accuracy")
                matrix[i, j] = r.metrics.get(metric, 0.0)

    return matrix, model_names, item_ids
