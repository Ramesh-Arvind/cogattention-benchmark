#!/usr/bin/env python3
"""
Generate a Prolific/Qualtrics survey from CogAttention benchmark items.

Samples 30 items (6 per ability cluster, stratified by difficulty Easy/Medium/Hard),
exports to Qualtrics-compatible CSV and JSON, and exports a separate scoring key.

Usage:
    python3 human_baseline/generate_survey.py [--output-dir human_baseline/survey_output]
"""

import argparse
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.generators.capacity import generate_capacity_dataset
from src.generators.novel_capacity import generate_interference_dataset
from src.generators.sustained import generate_sustained_dataset
from src.generators.novel_sustained import generate_stream_dataset
from src.generators.selective import generate_selective_dataset
from src.generators.novel_selective import generate_stroop_dataset
from src.generators.shifting import generate_shifting_dataset
from src.generators.inhibition_return import generate_ior_dataset
from src.generators.anomaly import generate_anomaly_dataset
from src.generators.flanker import generate_flanker_dataset
from src.generators.base import TaskInstance

from src.human_baseline.task_sampler import sample_human_tasks, estimate_human_time
from src.human_baseline.format_for_survey import (
    format_for_qualtrics,
    format_scoring_key,
)


# ---------------------------------------------------------------------------
# Ability clusters: map cluster name -> list of (task_type, generator_fn)
# ---------------------------------------------------------------------------
ABILITY_CLUSTERS = {
    "capacity": [
        ("capacity", generate_capacity_dataset),
        ("interference", generate_interference_dataset),
    ],
    "sustained": [
        ("sustained", generate_sustained_dataset),
        ("stream_segregation", generate_stream_dataset),
    ],
    "selective": [
        ("selective", generate_selective_dataset),
        ("stroop", generate_stroop_dataset),
    ],
    "shifting": [
        ("shifting", generate_shifting_dataset),
        ("inhibition_return", generate_ior_dataset),
    ],
    "anomaly": [
        ("anomaly", generate_anomaly_dataset),
    ],
}

ITEMS_PER_CLUSTER = 6
DIFFICULTIES = ["Easy", "Medium", "Hard"]
ITEMS_PER_DIFFICULTY = 2  # 2 per difficulty x 3 difficulties = 6 per cluster
SEED = 42


def sample_cluster_items(cluster_name, gen_pairs, seed=SEED):
    """Sample ITEMS_PER_CLUSTER items from a cluster, stratified by difficulty.

    Within each cluster, items are drawn round-robin from available task types
    so that each task type is represented.
    """
    import random
    rng = random.Random(seed)

    # Generate all items from all generators in the cluster
    all_items = []
    for task_type, gen_fn in gen_pairs:
        dataset = gen_fn(seed=2026)
        for item in dataset:
            if item.difficulty in DIFFICULTIES:
                all_items.append(item)

    # Stratify by difficulty
    by_diff = {d: [] for d in DIFFICULTIES}
    for item in all_items:
        by_diff[item.difficulty].append(item)

    sampled = []
    for diff in DIFFICULTIES:
        candidates = by_diff[diff]
        if not candidates:
            continue
        rng.shuffle(candidates)
        # Take ITEMS_PER_DIFFICULTY, trying to balance across task types
        task_types_in_cluster = [tt for tt, _ in gen_pairs]
        selected = []
        # Round-robin: pick one from each task type, then fill remaining
        for tt in task_types_in_cluster:
            tt_items = [c for c in candidates if c.task_type == tt and c not in selected]
            if tt_items and len(selected) < ITEMS_PER_DIFFICULTY:
                selected.append(tt_items[0])
        # Fill remaining slots if needed
        remaining = [c for c in candidates if c not in selected]
        while len(selected) < ITEMS_PER_DIFFICULTY and remaining:
            selected.append(remaining.pop(0))
        sampled.extend(selected[:ITEMS_PER_DIFFICULTY])

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Generate CogAttention human baseline survey")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "human_baseline", "survey_output"),
        help="Directory for output files",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  CogAttention Human Baseline Survey Generator")
    print("=" * 60)

    # ---- Sample items per cluster ----
    all_sampled = []
    for cluster_name in ["capacity", "sustained", "selective", "shifting", "anomaly"]:
        gen_pairs = ABILITY_CLUSTERS[cluster_name]
        items = sample_cluster_items(cluster_name, gen_pairs, seed=args.seed)
        print(f"\n  Cluster: {cluster_name}")
        print(f"    Sampled {len(items)} items")
        for item in items:
            print(f"      [{item.difficulty:>6}] {item.task_type}: {item.task_id}")
        all_sampled.extend(items)

    print(f"\n  Total items sampled: {len(all_sampled)}")

    # ---- Estimate completion time ----
    time_est = estimate_human_time(all_sampled)
    print(f"\n  Estimated completion time:")
    print(f"    Total: {time_est['total_minutes']:.1f} minutes")
    print(f"    Per item avg: {time_est['per_item_avg_seconds']:.0f} seconds")
    print(f"    Items: {time_est['n_items']}")

    # ---- Export Qualtrics CSV ----
    csv_path = format_for_qualtrics(
        all_sampled,
        os.path.join(args.output_dir, "qualtrics_survey.csv"),
        format="csv",
    )
    print(f"\n  Qualtrics CSV: {csv_path}")

    # ---- Export JSON (for programmatic import) ----
    json_path = format_for_qualtrics(
        all_sampled,
        os.path.join(args.output_dir, "qualtrics_survey.json"),
        format="json",
    )
    print(f"  Qualtrics JSON: {json_path}")

    # ---- Export scoring key ----
    key_path = format_scoring_key(
        all_sampled,
        os.path.join(args.output_dir, "scoring_key.json"),
    )
    print(f"  Scoring key:    {key_path}")

    # ---- Export survey metadata ----
    metadata = {
        "n_items": len(all_sampled),
        "items_per_cluster": ITEMS_PER_CLUSTER,
        "difficulties_included": DIFFICULTIES,
        "seed": args.seed,
        "estimated_minutes": time_est["total_minutes"],
        "estimated_per_item_seconds": time_est["per_item_avg_seconds"],
        "clusters": {},
    }
    for cluster_name in ABILITY_CLUSTERS:
        cluster_items = [i for i in all_sampled
                         if i.task_type in [tt for tt, _ in ABILITY_CLUSTERS[cluster_name]]]
        metadata["clusters"][cluster_name] = {
            "n_items": len(cluster_items),
            "task_types": list(set(i.task_type for i in cluster_items)),
            "difficulties": list(set(i.difficulty for i in cluster_items)),
        }

    meta_path = os.path.join(args.output_dir, "survey_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata:       {meta_path}")

    # ---- Export item-to-cluster mapping (for scoring) ----
    item_cluster_map = {}
    for cluster_name, gen_pairs in ABILITY_CLUSTERS.items():
        task_types = [tt for tt, _ in gen_pairs]
        for item in all_sampled:
            if item.task_type in task_types:
                item_cluster_map[item.task_id] = cluster_name

    map_path = os.path.join(args.output_dir, "item_cluster_map.json")
    with open(map_path, "w") as f:
        json.dump(item_cluster_map, f, indent=2)
    print(f"  Cluster map:    {map_path}")

    print("\n" + "=" * 60)
    print("  Survey generation complete.")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
