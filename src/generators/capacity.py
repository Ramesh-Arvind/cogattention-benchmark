"""
Task A: Thread Tracking (Attention Capacity)

Tests how many concurrent items a system can track through pairwise swaps.
Cognitive construct: §7.3.1 Attention Capacity
Paradigm source: Multiple Object Tracking (Pylyshyn) + PI-LLM

Difficulty scales by number of people (N) and number of swaps.
Ground truth is 100% programmatically computed.
"""

import random
from typing import List, Dict, Tuple
from .base import (
    TaskInstance, FIRST_NAMES, COLORS, OBJECTS, DIFFICULTY_LEVELS,
    sample_unique, set_seed,
)


# ─── Difficulty parameters ──────────────────────────────────────────
DIFFICULTY_CONFIG = {
    "Easy":   {"n_people": 2, "n_swaps": 2},
    "Medium": {"n_people": 3, "n_swaps": 4},
    "Hard":   {"n_people": 4, "n_swaps": 7},
    "Expert":   {"n_people": 5, "n_swaps": 12},
    "Frontier": {"n_people": 8, "n_swaps": 25},
}

ITEMS_PER_DIFFICULTY = 8  # 8 per level × 4 levels = 32 total


def _compute_ground_truth(
    people: List[str],
    items: List[str],
    swaps: List[Tuple[int, int]],
) -> Dict[str, str]:
    """Replay swap sequence and return final person→item mapping."""
    holdings = dict(zip(people, items))
    for i, j in swaps:
        p1, p2 = people[i], people[j]
        holdings[p1], holdings[p2] = holdings[p2], holdings[p1]
    return holdings


def _generate_swap_sequence(
    n_people: int,
    n_swaps: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Generate a valid swap sequence (no consecutive identical swaps when possible)."""
    # Build all possible swaps
    all_pairs = [(i, j) for i in range(n_people) for j in range(i + 1, n_people)]
    swaps = []
    prev = (-1, -1)
    for _ in range(n_swaps):
        candidates = [(i, j) for i, j in all_pairs
                       if (i, j) != prev and (j, i) != prev]
        # Fallback: if only one possible swap (n_people=2), allow repeats
        if not candidates:
            candidates = all_pairs
        pair = rng.choice(candidates)
        swaps.append(pair)
        prev = pair
    return swaps


def generate_capacity_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    """Generate a single Thread Tracking instance."""
    rng = random.Random(seed + instance_idx * 1000 + hash(difficulty))
    config = DIFFICULTY_CONFIG[difficulty]
    n_people = config["n_people"]
    n_swaps = config["n_swaps"]

    # Sample unique names and items
    people = [FIRST_NAMES[i] for i in rng.sample(range(len(FIRST_NAMES)), n_people)]
    colors = [COLORS[i] for i in rng.sample(range(len(COLORS)), n_people)]
    objects_ = [OBJECTS[i] for i in rng.sample(range(len(OBJECTS)), n_people)]
    items = [f"{c} {o}" for c, o in zip(colors, objects_)]

    # Generate swaps
    swaps = _generate_swap_sequence(n_people, n_swaps, rng)

    # Compute ground truth
    gt = _compute_ground_truth(people, items, swaps)

    # Build prompt
    start_lines = "\n".join(
        f"- {p} holds a {item}" for p, item in zip(people, items)
    )
    swap_lines = "\n".join(
        f"{idx+1}. {people[i]} and {people[j]} swap items."
        for idx, (i, j) in enumerate(swaps)
    )

    prompt = (
        f"You are given {n_people} people, each holding a unique item. "
        f"After a series of swaps, report who holds each item.\n\n"
        f"Starting positions:\n{start_lines}\n\n"
        f"Swaps:\n{swap_lines}\n\n"
        f"After all swaps, list each person and their current item.\n"
        f"Format your answer EXACTLY as:\n"
        f"ANSWER:\n"
        + "\n".join(f"- [Person]: [item]" for _ in range(n_people))
    )

    # Gold answer as dict and formatted string
    gold_formatted = "\n".join(f"- {p}: {gt[p]}" for p in people)

    return TaskInstance(
        task_id=f"capacity_{difficulty.lower()}_{instance_idx:03d}",
        task_type="capacity",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gt,
        metadata={
            "n_people": n_people,
            "n_swaps": n_swaps,
            "people": people,
            "initial_items": dict(zip(people, items)),
            "swaps": [(people[i], people[j]) for i, j in swaps],
            "gold_formatted": gold_formatted,
        },
    )


def generate_capacity_dataset(seed: int = 2026) -> List[TaskInstance]:
    """Generate all 32 Thread Tracking instances."""
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_capacity_instance(
                instance_idx=len(instances),
                difficulty=diff,
                seed=seed,
            )
            instances.append(inst)
    return instances


# ─── Self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = generate_capacity_dataset()
    print(f"Generated {len(dataset)} Thread Tracking instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        print(f"\n  {diff}: {len(subset)} instances")
        sample = subset[0]
        print(f"    N={sample.metadata['n_people']}, "
              f"Swaps={sample.metadata['n_swaps']}")
        print(f"    Gold: {sample.metadata['gold_formatted']}")
    # Verify ground truth correctness
    print("\n--- Verification ---")
    for inst in dataset:
        gt = inst.gold_answer
        people = inst.metadata["people"]
        initial = inst.metadata["initial_items"]
        # Re-compute
        holdings = dict(initial)
        for p1_name, p2_name in inst.metadata["swaps"]:
            holdings[p1_name], holdings[p2_name] = holdings[p2_name], holdings[p1_name]
        assert holdings == gt, f"MISMATCH in {inst.task_id}"
    print("All ground truth verified!")
