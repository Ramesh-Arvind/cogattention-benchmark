"""
Scorer for Task B: Vigilance Probe + Stream Segregation + Context Dilution + Semantic NIAH + Multi-hop.
"""

import re
from typing import Dict, List
from .base import (
    ScoreResult, extract_answer_block, extract_list_items,
    extract_numbered_answers, compute_recall, compute_precision,
    compute_intrusion_rate, fuzzy_value_match,
)
from ..generators.base import TaskInstance


def score_sustained(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Vigilance Probe response."""
    result = ScoreResult(instance.task_id, "sustained", instance.difficulty)
    result.raw_response = response

    gold_targets = instance.gold_answer  # List[str]
    nearmisses = instance.metadata["nearmisses"]
    quintile_targets = instance.metadata["quintile_targets"]

    # Extract reported items
    found_items = extract_list_items(response)

    # Core metrics
    recall = compute_recall(found_items, gold_targets)
    precision = compute_precision(found_items, gold_targets)
    intrusion = compute_intrusion_rate(found_items, nearmisses)

    # Vigilance decrement: compare recall in first quintile vs last quintile
    q_recalls = {}
    for q_str, q_targets in quintile_targets.items():
        if q_targets:
            q_found = sum(1 for t in q_targets
                         if any(fuzzy_value_match(f, t) for f in found_items))
            q_recalls[q_str] = q_found / len(q_targets)
        else:
            q_recalls[q_str] = None

    # Decrement = early recall - late recall (positive = degradation)
    early_recall = q_recalls.get("0")
    late_recall = q_recalls.get("4")
    if early_recall is not None and late_recall is not None:
        vigilance_decrement = early_recall - late_recall
    else:
        vigilance_decrement = 0.0

    result.add_metric("recall", recall)
    result.add_metric("precision", precision)
    result.add_metric("intrusion_rate", intrusion)
    result.add_metric("vigilance_decrement", vigilance_decrement)
    result.add_metric("f1", 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0)
    result.details = {
        "found_items": found_items,
        "gold_targets": gold_targets,
        "quintile_recalls": q_recalls,
        "n_found": len(found_items),
        "n_gold": len(gold_targets),
    }

    return result


def score_stream(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Stream Segregation response."""
    result = ScoreResult(instance.task_id, "stream_segregation", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # Dict: {"1": first_number, optionally "2": "yes"}
    has_breakthrough = instance.metadata["has_breakthrough"]

    answers = extract_numbered_answers(response)

    # Question 1: first number in stream A
    q1_correct = False
    gold_num = gold.get("1", "unknown")
    pred_q1 = answers.get("1", "")

    # Fallback: if numbered extraction failed, search the full response for the gold number
    if not pred_q1:
        answer_block = extract_answer_block(response)
        # Try to find any number in the response
        nums_found = re.findall(r'\b\d+(?:\.\d+)?\b', answer_block)
        if nums_found:
            pred_q1 = answer_block  # Use full block for fuzzy matching

    if gold_num and gold_num != "unknown":
        q1_correct = fuzzy_value_match(pred_q1, gold_num)

    # Question 2: breakthrough detection (if applicable)
    q2_correct = None
    if has_breakthrough:
        pred_q2 = answers.get("2", "").lower()
        q2_correct = "yes" in pred_q2

    stream_accuracy = 1.0 if q1_correct else 0.0
    breakthrough_detected = 1.0 if q2_correct else (0.0 if q2_correct is not None else None)

    result.add_metric("stream_accuracy", stream_accuracy)
    if breakthrough_detected is not None:
        result.add_metric("breakthrough_detection", breakthrough_detected)
        # Weighted average instead of multiplication to avoid floor effect
        result.add_metric("dual_score", 0.6 * stream_accuracy + 0.4 * breakthrough_detected)
    else:
        result.add_metric("dual_score", stream_accuracy)

    result.details = {
        "q1_predicted": pred_q1,
        "q1_gold": gold_num,
        "q1_correct": q1_correct,
        "q2_predicted": answers.get("2"),
        "q2_correct": q2_correct,
    }

    return result


def score_dilution(instance, response: str) -> ScoreResult:
    """Score a Context Dilution response."""
    result = ScoreResult(instance.task_id, "context_dilution", instance.difficulty)
    result.raw_response = response
    gold = str(instance.gold_answer)
    answer_block = extract_answer_block(response)
    passed = fuzzy_value_match(answer_block, gold)
    result.add_metric("accuracy", 1.0 if passed else 0.0)
    result.details = {
        "predicted": answer_block[:200],
        "gold": gold,
        "correct": passed,
        "word_count": instance.metadata["word_count"],
        "is_shuffled": instance.metadata["is_shuffled"],
        "needle_depth": instance.metadata["needle_depth"],
    }
    return result


def score_sniah(instance, response: str) -> ScoreResult:
    """Score a Semantic NIAH response."""
    result = ScoreResult(instance.task_id, "semantic_niah", instance.difficulty)
    result.raw_response = response
    gold = str(instance.gold_answer)
    answer_block = extract_answer_block(response)
    passed = fuzzy_value_match(answer_block, gold)
    result.add_metric("accuracy", 1.0 if passed else 0.0)
    result.details = {
        "predicted": answer_block[:200],
        "gold": gold,
        "correct": passed,
        "n_paragraphs": instance.metadata["n_paragraphs"],
        "needle_depth": instance.metadata["needle_depth_ratio"],
        "word_count": instance.metadata["word_count"],
    }
    return result


def score_multihop(instance, response: str) -> ScoreResult:
    """Score a Multi-hop Scattered Reasoning response."""
    result = ScoreResult(instance.task_id, "multihop", instance.difficulty)
    result.raw_response = response
    gold = str(instance.gold_answer)
    answer_block = extract_answer_block(response)
    passed = fuzzy_value_match(answer_block, gold)
    result.add_metric("accuracy", 1.0 if passed else 0.0)
    result.details = {
        "predicted": answer_block[:200],
        "gold": gold,
        "correct": passed,
        "n_hops": instance.metadata["n_hops"],
        "has_distractors": instance.metadata["has_distractors"],
        "word_count": instance.metadata["word_count"],
    }
    return result
