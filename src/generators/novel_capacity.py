"""
Task A-Novel: Proactive Interference Chain (Attention Capacity)

NOVEL CONTRIBUTION: No existing LLM benchmark tests this.
Source: PI-LLM (Wang & Sun, ICML 2025 Workshop, arXiv:2506.08184)

Streams K sequential updates to the same key. The model must report
only the FINAL value. Accuracy declines log-linearly as K increases
(proactive interference — an architectural limitation).
"""

import random
from typing import List
from .base import TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES

ITEMS_PER_DIFFICULTY = 8

DIFFICULTY_CONFIG = {
    "Easy":   {"n_updates": 3,  "n_keys": 1, "n_decoys": 0},
    "Medium": {"n_updates": 6,  "n_keys": 2, "n_decoys": 0},
    "Hard":   {"n_updates": 12, "n_keys": 3, "n_decoys": 1},
    "Expert":   {"n_updates": 25, "n_keys": 4, "n_decoys": 2},
    "Frontier": {"n_updates": 50, "n_keys": 8, "n_decoys": 4},
}

# Semantic fields for key-value pairs (all same domain = max interference)
VALUE_POOLS = {
    "city": CITIES,
    "person": FIRST_NAMES,
    "number": [str(i) for i in range(100, 1000)],
}


def generate_interference_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1031 + DIFFICULTY_LEVELS.index(difficulty) * 4201)
    config = DIFFICULTY_CONFIG[difficulty]

    n_keys = config["n_keys"]
    n_updates = config["n_updates"]

    # Choose value domain
    domain = rng.choice(list(VALUE_POOLS.keys()))
    pool = VALUE_POOLS[domain]

    # Generate key names
    key_names = rng.sample(["location", "contact", "reference", "assignment",
                             "destination", "delegate", "coordinator", "liaison",
                             "registry", "dispatch"], n_keys)

    # Generate update sequences for each key
    update_sequences = {}
    final_values = {}
    for key in key_names:
        values = [rng.choice(pool) for _ in range(n_updates)]
        # Ensure no two consecutive values are identical
        for i in range(1, len(values)):
            while values[i] == values[i-1]:
                values[i] = rng.choice(pool)
        update_sequences[key] = values
        final_values[key] = values[-1]

    # Interleave updates across keys
    all_updates = []
    for step in range(n_updates):
        for key in key_names:
            all_updates.append((key, update_sequences[key][step], step + 1))

    # Build prompt
    lines = [
        "You will receive a series of updates to one or more records. "
        "Each update REPLACES the previous value. "
        "After reading ALL updates, report ONLY the FINAL value for each record.",
        "",
        "Updates (in order):",
    ]

    for key, value, step in all_updates:
        lines.append(f"  Update: {key} is now set to '{value}'")

    lines.append("")
    lines.append("What is the FINAL value of each record?")
    lines.append("ANSWER:")
    for key in key_names:
        lines.append(f"- {key}: [final value]")

    # Decoy questions: ask about prior values to test interference resistance
    n_decoys = config["n_decoys"]
    decoy_questions = []
    if n_decoys > 0:
        lines.append("")
        lines.append("Also answer these verification questions (Yes or No):")
        decoy_idx = 0
        for key in key_names:
            prior = update_sequences[key][:-1]
            if prior and decoy_idx < n_decoys:
                # Pick a prior value that is NOT the final value
                decoy_val = rng.choice(prior)
                while decoy_val == final_values[key] and len(set(prior)) > 1:
                    decoy_val = rng.choice(prior)
                lines.append(
                    f"V{decoy_idx + 1}. Was '{decoy_val}' ever assigned to {key}? [Yes/No]"
                )
                decoy_questions.append({
                    "key": key, "value": decoy_val,
                    "answer": "Yes",
                })
                decoy_idx += 1

    prompt = "\n".join(lines)

    return TaskInstance(
        task_id=f"interference_{difficulty.lower()}_{instance_idx:03d}",
        task_type="interference",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=final_values,
        metadata={
            "n_keys": n_keys,
            "n_updates_per_key": n_updates,
            "total_updates": len(all_updates),
            "domain": domain,
            "key_names": key_names,
            "final_values": final_values,
            "all_prior_values": {k: v[:-1] for k, v in update_sequences.items()},
            "decoy_questions": decoy_questions,
        },
    )


def generate_interference_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_interference_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_interference_dataset()
    print(f"Generated {len(dataset)} Proactive Interference instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_keys']} keys × "
              f"{s.metadata['n_updates_per_key']} updates = "
              f"{s.metadata['total_updates']} total")
        print(f"    Final values: {s.gold_answer}")
