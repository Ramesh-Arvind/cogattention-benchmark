"""
Scorer for Task A: Thread Tracking + Proactive Interference + Attentional Blink.
"""

import re
from typing import Dict
from .base import (
    ScoreResult, extract_person_item_pairs, extract_answer_block,
    fuzzy_value_match, extract_list_items,
)
from ..generators.base import TaskInstance


def score_capacity(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Thread Tracking response."""
    result = ScoreResult(instance.task_id, "capacity", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # Dict[person_name, item_descriptor]
    people = instance.metadata["people"]

    # Extract person-item pairs from response
    predicted = extract_person_item_pairs(response)

    # Score each person
    correct = 0
    details = {}
    for person in people:
        gold_item = gold[person]
        # Find this person in predicted (case-insensitive match)
        pred_item = None
        for pred_name, pred_val in predicted.items():
            if pred_name.lower() == person.lower():
                pred_item = pred_val
                break

        if pred_item and fuzzy_value_match(pred_item, gold_item):
            correct += 1
            details[person] = {"status": "correct", "predicted": pred_item, "gold": gold_item}
        else:
            details[person] = {
                "status": "incorrect",
                "predicted": pred_item or "[not found]",
                "gold": gold_item,
            }

    n_people = len(people)
    accuracy = correct / n_people if n_people > 0 else 0.0

    result.add_metric("accuracy", accuracy)
    result.add_metric("correct_count", correct)
    result.add_metric("total_people", n_people)
    result.details = details

    return result


def score_interference(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Proactive Interference response."""
    result = ScoreResult(instance.task_id, "interference", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # Dict[key_name, final_value]
    key_names = instance.metadata["key_names"]
    prior_values = instance.metadata["all_prior_values"]

    answer_block = extract_answer_block(response)

    correct = 0
    perseveration = 0  # gave an old value instead of the latest
    details = {}

    for key in key_names:
        gold_val = gold[key]
        # Find this key in the response
        pattern = rf"(?i){re.escape(key)}\s*[:\-=]\s*['\"]?(.+?)['\"]?\s*(?:\n|$)"
        match = re.search(pattern, answer_block)
        pred_val = match.group(1).strip().rstrip('.') if match else None

        if pred_val and fuzzy_value_match(pred_val, gold_val):
            correct += 1
            details[key] = {"status": "correct", "predicted": pred_val, "gold": gold_val}
        else:
            # Check if it's a prior value (perseveration)
            is_prior = False
            if pred_val and key in prior_values:
                for old_val in prior_values[key]:
                    if fuzzy_value_match(pred_val, old_val):
                        is_prior = True
                        perseveration += 1
                        break
            details[key] = {
                "status": "perseveration" if is_prior else "incorrect",
                "predicted": pred_val or "[not found]",
                "gold": gold_val,
            }

    n_keys = len(key_names)
    accuracy = correct / n_keys if n_keys > 0 else 0.0
    perseveration_rate = perseveration / n_keys if n_keys > 0 else 0.0

    result.add_metric("accuracy", accuracy)
    result.add_metric("perseveration_rate", perseveration_rate)
    result.add_metric("correct_count", correct)
    result.add_metric("total_keys", n_keys)
    result.details = details

    return result


def score_blink(instance, response: str) -> ScoreResult:
    """Score an Attentional Blink response."""
    result = ScoreResult(instance.task_id, "blink", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # {"T1": ..., "T2": ...}

    answer_block = extract_answer_block(response)

    # Extract T1 and T2 from response
    t1_match = re.search(r"(?i)T1\s*[:\-=]\s*(.+?)(?:\n|$)", answer_block)
    t2_match = re.search(r"(?i)T2\s*[:\-=]\s*(.+?)(?:\n|$)", answer_block)

    pred_t1 = t1_match.group(1).strip().rstrip(".") if t1_match else ""
    pred_t2 = t2_match.group(1).strip().rstrip(".") if t2_match else ""

    t1_correct = fuzzy_value_match(pred_t1, gold["T1"]) if pred_t1 else False
    t2_correct = fuzzy_value_match(pred_t2, gold["T2"]) if pred_t2 else False

    # T2|T1 accuracy (T2 correct given T1 correct) — the blink metric
    t2_given_t1 = t2_correct if t1_correct else False

    result.add_metric("t1_accuracy", 1.0 if t1_correct else 0.0)
    result.add_metric("t2_accuracy", 1.0 if t2_correct else 0.0)
    result.add_metric("t2_given_t1", 1.0 if t2_given_t1 else 0.0)
    result.add_metric("accuracy", (int(t1_correct) + int(t2_correct)) / 2.0)
    result.details = {
        "pred_t1": pred_t1,
        "pred_t2": pred_t2,
        "gold_t1": gold["T1"],
        "gold_t2": gold["T2"],
        "t1_correct": t1_correct,
        "t2_correct": t2_correct,
        "lag": instance.metadata["lag"],
    }

    return result
