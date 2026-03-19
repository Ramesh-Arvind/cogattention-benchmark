#!/usr/bin/env python3
"""Main evaluation harness for local LLM validation.

Usage examples:
    # Smoke test (1 instance, single task, smallest model)
    python3 src/run_local_eval.py --model phi3 --pilot-n 1 --tasks capacity

    # Pilot run (2 per difficulty per task, all models sequentially)
    python3 src/run_local_eval.py --model all --pilot-n 2

    # Full 256-instance run
    python3 src/run_local_eval.py --model all --full
"""

import argparse
import gc
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from src.eval_config import MODEL_CONFIGS, get_task_registry
from src.generators.base import DIFFICULTY_LEVELS, TaskInstance
from src.inference_engine import InferenceEngine
from src.scorers.base import ScoreResult
from src.scorers.composite import compute_cas, format_cas_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_local_eval")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(model_name: str) -> str:
    return os.path.join(LOGS_DIR, f"checkpoint_{model_name}.json")


def load_checkpoint(model_name: str) -> Dict:
    path = checkpoint_path(model_name)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded checkpoint for %s: %d completed", model_name, len(data.get("completed_ids", [])))
        return data
    return {"completed_ids": [], "results": []}


def save_checkpoint(model_name: str, state: Dict):
    path = checkpoint_path(model_name)
    with open(path, "w") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Memory profiler (daemon thread)
# ---------------------------------------------------------------------------

def _memory_profiler(model_name: str, interval: float = 30.0, stop_event: threading.Event = None):
    """Log CPU + GPU memory usage periodically."""
    log_path = os.path.join(LOGS_DIR, f"memory_profile_{model_name}.log")
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except ImportError:
        process = None

    with open(log_path, "a") as f:
        while not (stop_event and stop_event.is_set()):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = [f"[{ts}]"]

            # CPU RAM
            if process:
                mem = process.memory_info()
                lines.append(f"  CPU RSS: {mem.rss / 1e9:.2f} GB")

            # GPU RAM
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    lines.append(f"  GPU {i}: alloc={alloc:.2f} GB, reserved={reserved:.2f} GB")

            f.write("\n".join(lines) + "\n")
            f.flush()
            if stop_event:
                stop_event.wait(interval)
            else:
                time.sleep(interval)


def start_memory_profiler(model_name: str) -> threading.Event:
    stop = threading.Event()
    t = threading.Thread(target=_memory_profiler, args=(model_name, 30.0, stop), daemon=True)
    t.start()
    return stop


# ---------------------------------------------------------------------------
# Dataset generation & pilot selection
# ---------------------------------------------------------------------------

def generate_all_instances(seed: int, task_filter: Optional[List[str]] = None) -> List[TaskInstance]:
    """Generate all benchmark instances via the task registry."""
    registry = get_task_registry()
    all_instances: List[TaskInstance] = []
    for task_type, (gen_fn, _) in registry.items():
        if task_filter and task_type not in task_filter:
            continue
        logger.info("Generating instances for task: %s", task_type)
        instances = gen_fn(seed=seed)
        all_instances.extend(instances)
        logger.info("  -> %d instances", len(instances))
    logger.info("Total instances generated: %d", len(all_instances))
    return all_instances


def select_pilot_subset(instances: List[TaskInstance], n_per_difficulty: int) -> List[TaskInstance]:
    """Select first N instances per difficulty per task type."""
    if n_per_difficulty <= 0:
        return instances  # return all

    buckets: Dict[str, Dict[str, List[TaskInstance]]] = defaultdict(lambda: defaultdict(list))
    for inst in instances:
        buckets[inst.task_type][inst.difficulty].append(inst)

    selected = []
    for task_type in sorted(buckets.keys()):
        for diff in DIFFICULTY_LEVELS:
            candidates = buckets[task_type].get(diff, [])
            selected.extend(candidates[:n_per_difficulty])

    logger.info("Pilot subset: %d instances (n_per_difficulty=%d)", len(selected), n_per_difficulty)
    return selected


# ---------------------------------------------------------------------------
# Evaluation loop for one model
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    instances: List[TaskInstance],
    resume: bool = True,
) -> List[Dict]:
    """Run inference + scoring for all instances on one model."""
    config = MODEL_CONFIGS[model_name]
    registry = get_task_registry()

    # Checkpoint
    state = load_checkpoint(model_name) if resume else {"completed_ids": [], "results": []}
    completed_ids = set(state["completed_ids"])
    results = state["results"]

    pending = [inst for inst in instances if inst.task_id not in completed_ids]
    if not pending:
        logger.info("All instances already completed for %s.", model_name)
        return results

    logger.info("Evaluating %s: %d pending (%d already done)", model_name, len(pending), len(completed_ids))

    # Init engine (MUST happen before any CUDA calls to avoid fork issues with tp>1)
    engine = InferenceEngine(config)

    # Start memory profiler (after engine init to avoid CUDA fork issues)
    mem_stop = start_memory_profiler(model_name)

    try:
        for inst in tqdm(pending, desc=f"eval-{model_name}"):
            # Get scorer
            _, scorer_fn = registry[inst.task_type]

            # Inference
            response_text, gen_meta = engine.generate(inst.prompt, inst.task_id)

            # Score
            score_result: ScoreResult = scorer_fn(inst, response_text)
            score_result.raw_response = response_text

            result_dict = score_result.to_dict()
            result_dict["model"] = model_name
            result_dict["generation_meta"] = gen_meta
            result_dict["raw_response"] = response_text[:2000]  # truncate for storage

            results.append(result_dict)
            completed_ids.add(inst.task_id)

            # Save checkpoint after every item
            state["completed_ids"] = list(completed_ids)
            state["results"] = results
            save_checkpoint(model_name, state)

    finally:
        mem_stop.set()
        engine.shutdown()

    return results


# ---------------------------------------------------------------------------
# Cross-model comparison helpers
# ---------------------------------------------------------------------------

def build_score_results(results: List[Dict]) -> List[ScoreResult]:
    """Reconstruct ScoreResult objects from result dicts for compute_cas."""
    score_results = []
    for r in results:
        sr = ScoreResult(r["task_id"], r["task_type"], r["difficulty"])
        # Copy all numeric metrics (skip known non-metric keys)
        skip = {"task_id", "task_type", "difficulty", "details", "model", "generation_meta"}
        for k, v in r.items():
            if k not in skip and isinstance(v, (int, float)):
                sr.add_metric(k, v)
        if "details" in r:
            sr.details = r["details"]
        score_results.append(sr)
    return score_results


def save_model_results(model_name: str, results: List[Dict]):
    """Save per-model results and CAS report."""
    # Raw results
    out_path = os.path.join(RESULTS_DIR, f"pilot_{model_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved results: %s", out_path)

    # CAS
    score_results = build_score_results(results)
    cas = compute_cas(score_results)
    cas_path = os.path.join(RESULTS_DIR, f"pilot_{model_name}_cas.json")
    with open(cas_path, "w") as f:
        json.dump(cas, f, indent=2, default=str)

    report = format_cas_report(cas)
    logger.info("\n%s CAS Report:\n%s", model_name, report)

    return cas


def generate_cross_model_summary(all_cas: Dict[str, Dict]):
    """Generate combined pilot_metrics.json with all models."""
    metrics = {"models": {}, "comparison": {}}

    for model_name, cas in all_cas.items():
        metrics["models"][model_name] = {
            "cas_score": cas["cas_score"],
            "total_instances": cas["total_instances"],
            "per_task": {
                t: {"mean": info["mean"], "std": info["std"]}
                for t, info in cas["per_task"].items()
            },
        }

    # Per-task cross-model spread
    task_types = set()
    for cas in all_cas.values():
        task_types.update(cas["per_task"].keys())

    comparison = {}
    for task in sorted(task_types):
        scores = []
        for model_name, cas in all_cas.items():
            if task in cas["per_task"]:
                scores.append((model_name, cas["per_task"][task]["mean"]))
        if len(scores) >= 2:
            vals = [s for _, s in scores]
            spread = max(vals) - min(vals)
            comparison[task] = {
                "spread": round(spread, 4),
                "scores": {m: round(v, 4) for m, v in scores},
            }

    metrics["comparison"] = comparison

    out_path = os.path.join(RESULTS_DIR, "pilot_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved cross-model metrics: %s", out_path)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local LLM evaluation harness")
    parser.add_argument(
        "--model",
        type=str,
        default="phi3",
        help="Model alias (qwen72b, llama8b, phi3) or 'all'",
    )
    parser.add_argument(
        "--pilot-n",
        type=int,
        default=2,
        help="Instances per difficulty per task for pilot (default 2)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 256 instances (overrides --pilot-n)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task types to evaluate (default: all)",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    args = parser.parse_args()

    task_filter = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    # Determine models to run
    if args.model == "all":
        model_names = list(MODEL_CONFIGS.keys())
    else:
        if args.model not in MODEL_CONFIGS:
            logger.error("Unknown model: %s. Available: %s", args.model, list(MODEL_CONFIGS.keys()))
            sys.exit(1)
        model_names = [args.model]

    # Generate instances
    all_instances = generate_all_instances(args.seed, task_filter)

    # Select subset
    if args.full:
        instances = all_instances
    else:
        instances = select_pilot_subset(all_instances, args.pilot_n)

    logger.info("Models: %s | Instances: %d | Resume: %s", model_names, len(instances), args.resume)

    # Run each model sequentially
    all_cas: Dict[str, Dict] = {}
    for model_name in model_names:
        logger.info("=" * 60)
        logger.info("Starting evaluation: %s", model_name)
        logger.info("=" * 60)

        results = evaluate_model(model_name, instances, resume=args.resume)
        cas = save_model_results(model_name, results)
        all_cas[model_name] = cas

        # Force cleanup between models
        gc.collect()
        torch.cuda.empty_cache()

    # Cross-model comparison
    if len(all_cas) > 1:
        generate_cross_model_summary(all_cas)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
