"""
Scorer for Task C: Distractor Filtering + Semantic Stroop + Flanker.
"""

import re
from .base import (
    ScoreResult, extract_answer_block, extract_list_items,
    extract_numbered_answers, compute_recall, compute_precision,
    compute_intrusion_rate, fuzzy_value_match,
)
from ..generators.base import TaskInstance


def score_selective(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Distractor Filtering response."""
    result = ScoreResult(instance.task_id, "selective", instance.difficulty)
    result.raw_response = response

    gold_signals = instance.gold_answer  # List[str] - signal values
    distractors = instance.metadata["distractor_values"]

    # Extract reported values
    found_items = extract_list_items(response)

    # Core metrics
    recall = compute_recall(found_items, gold_signals)
    precision = compute_precision(found_items, gold_signals)
    intrusion = compute_intrusion_rate(found_items, distractors)

    # SAS = Selective Attention Score = Recall × (1 - Intrusion)
    sas = recall * (1 - intrusion)

    # Noise ratio from generator metadata (for degradation analysis)
    noise_ratio = instance.metadata.get("noise_ratio", 0.0)

    result.add_metric("signal_recall", recall)
    result.add_metric("attention_precision", precision)
    result.add_metric("distractor_intrusion", intrusion)
    result.add_metric("sas_score", sas)
    result.add_metric("noise_ratio", noise_ratio)
    result.details = {
        "found_items": found_items,
        "gold_signals": gold_signals,
        "distractor_values": distractors,
        "n_found": len(found_items),
        "noise_ratio": noise_ratio,
    }

    return result


def score_stroop(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Semantic Stroop response."""
    result = ScoreResult(instance.task_id, "stroop", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # Dict[str, str]: {"1": correct_answer, ...}
    trap_answers = instance.metadata["trap_answers"]
    items = instance.metadata["items"]

    answers = extract_numbered_answers(response)

    correct = 0
    stroop_errors = 0  # Gave the "trap" answer (corrected the error)
    total = len(gold)
    details = {}

    for idx_str, gold_val in gold.items():
        pred = answers.get(idx_str, "")
        trap = trap_answers.get(idx_str)

        if fuzzy_value_match(pred, gold_val):
            correct += 1
            details[idx_str] = {"status": "correct", "predicted": pred, "gold": gold_val}
        elif trap and fuzzy_value_match(pred, trap):
            stroop_errors += 1
            details[idx_str] = {
                "status": "stroop_error",
                "predicted": pred,
                "gold": gold_val,
                "trap": trap,
            }
        else:
            details[idx_str] = {
                "status": "other_error",
                "predicted": pred,
                "gold": gold_val,
            }

    accuracy = correct / total if total > 0 else 0.0
    stroop_error_rate = stroop_errors / total if total > 0 else 0.0
    # Stroop resistance = 1 - stroop_error_rate (higher = better at suppressing correction reflex)
    stroop_resistance = 1.0 - stroop_error_rate

    result.add_metric("accuracy", accuracy)
    result.add_metric("stroop_error_rate", stroop_error_rate)
    result.add_metric("stroop_resistance", stroop_resistance)
    result.add_metric("correct_count", correct)
    result.add_metric("stroop_errors", stroop_errors)
    result.details = details

    return result


def score_flanker(instance, response: str) -> ScoreResult:
    """Score a Flanker response."""
    result = ScoreResult(instance.task_id, "flanker", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # string value
    answer_block = extract_answer_block(response)

    passed = fuzzy_value_match(answer_block, str(gold))

    result.add_metric("accuracy", 1.0 if passed else 0.0)
    result.details = {
        "predicted": answer_block[:200],
        "gold": gold,
        "correct": passed,
        "n_flankers": instance.metadata["n_flankers"],
        "target_position": instance.metadata["target_position"],
    }

    return result
