"""
Human baseline task sampler.

Selects a representative subset of benchmark items for human evaluation
via Prolific, MTurk, or manual testing.
"""

import random
from typing import List, Dict, Optional
from ..generators.base import TaskInstance, DIFFICULTY_LEVELS


def sample_human_tasks(
    generators: Dict[str, callable],
    n_per_task: int = 5,
    difficulties: Optional[List[str]] = None,
    seed: int = 42,
) -> List[TaskInstance]:
    """Sample representative tasks for human evaluation.

    Args:
        generators: {task_type: generator_fn} mapping.
        n_per_task: Number of items to sample per task type.
        difficulties: Which difficulties to sample from. Default: Easy, Medium, Hard.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled TaskInstance objects.
    """
    if difficulties is None:
        difficulties = ["Easy", "Medium", "Hard"]

    rng = random.Random(seed)
    sampled = []

    for task_type, gen_fn in sorted(generators.items()):
        dataset = gen_fn(seed=2026)

        # Filter to requested difficulties
        candidates = [d for d in dataset if d.difficulty in difficulties]
        if not candidates:
            continue

        # Sample n_per_task, stratified by difficulty
        per_diff = max(1, n_per_task // len(difficulties))
        for diff in difficulties:
            diff_items = [d for d in candidates if d.difficulty == diff]
            n_sample = min(per_diff, len(diff_items))
            sampled.extend(rng.sample(diff_items, n_sample))

    return sampled


def estimate_human_time(tasks: List[TaskInstance]) -> Dict[str, float]:
    """Estimate time required for human evaluation.

    Based on cognitive psychology norms:
    - Simple extraction: ~30 seconds per item
    - Long-context vigilance: ~2-3 minutes per item
    - Rule-shift classification: ~1 minute per item

    Returns:
        Dict with total_minutes, per_item_avg_seconds, n_items.
    """
    time_estimates = {
        "capacity": 45,
        "interference": 60,
        "sustained": 150,
        "stream_segregation": 90,
        "selective": 45,
        "stroop": 30,
        "flanker": 20,
        "shifting": 60,
        "inhibition_return": 60,
        "anomaly": 90,
    }

    total_seconds = sum(time_estimates.get(t.task_type, 60) for t in tasks)
    n_items = len(tasks)

    return {
        "total_minutes": round(total_seconds / 60, 1),
        "per_item_avg_seconds": round(total_seconds / n_items, 1) if n_items > 0 else 0,
        "n_items": n_items,
    }
