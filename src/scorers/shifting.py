"""
Scorer for Task D: Rule Shift + Inhibition of Return (Attention Shifting).
"""

import re
from .base import ScoreResult, extract_numbered_answers, extract_answer_block, fuzzy_value_match
from ..generators.base import TaskInstance


def score_shifting(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Rule Shift response."""
    result = ScoreResult(instance.task_id, "shifting", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # Dict[str, str]: {"1": answer, "2": answer, ...}
    # Frontier uses switch_points (list) for triple-rule; others use switch_point (int)
    if "switch_points" in instance.metadata:
        switch_point = instance.metadata["switch_points"][0]  # first switch
    else:
        switch_point = instance.metadata["switch_point"]
    pre_answers = instance.metadata["pre_answers"]
    post_answers = instance.metadata["post_answers"]
    perseveration_answers = instance.metadata["perseveration_answers"]

    answers = extract_numbered_answers(response)

    # Collect all pre-switch correct answers for residue detection
    pre_switch_correct_values = set()
    for word, ans in pre_answers:
        pre_switch_correct_values.add(ans.strip().lower())

    pre_correct = 0
    post_correct = 0
    perseveration_errors = 0
    residue_errors = 0
    random_errors = 0
    details = {}

    for idx_str, gold_val in gold.items():
        idx = int(idx_str)
        pred = answers.get(idx_str, "")
        is_pre = idx <= switch_point

        if fuzzy_value_match(pred, gold_val):
            if is_pre:
                pre_correct += 1
            else:
                post_correct += 1
            details[idx_str] = {"status": "correct", "predicted": pred,
                                "gold": gold_val, "phase": "pre" if is_pre else "post"}
        else:
            # Check perseveration (post-switch items answered with old rule)
            is_persev = False
            is_residue = False
            if not is_pre:
                post_idx = idx - switch_point - 1
                if 0 <= post_idx < len(perseveration_answers):
                    old_rule_answer = perseveration_answers[post_idx][1]
                    if fuzzy_value_match(pred, old_rule_answer):
                        is_persev = True
                        perseveration_errors += 1

                # Check residue: answer matches ANY pre-switch correct answer
                # (but not already flagged as perseveration)
                if not is_persev and pred.strip():
                    pred_lower = pred.strip().lower()
                    if pred_lower in pre_switch_correct_values:
                        is_residue = True
                        residue_errors += 1

                # Random error: wrong answer that is neither perseveration nor residue
                if not is_persev and not is_residue and not is_pre and pred.strip():
                    random_errors += 1
                elif not is_persev and not is_residue and not is_pre and not pred.strip():
                    random_errors += 1  # Empty answer counts as random error

            if is_persev:
                status = "perseveration"
            elif is_residue:
                status = "residue"
            elif not is_pre:
                status = "random_error"
            else:
                status = "incorrect"

            details[idx_str] = {
                "status": status,
                "predicted": pred,
                "gold": gold_val,
                "phase": "pre" if is_pre else "post",
            }

    n_pre = len(pre_answers)
    n_post = len(post_answers)
    pre_accuracy = pre_correct / n_pre if n_pre > 0 else 0.0
    post_accuracy = post_correct / n_post if n_post > 0 else 0.0
    switch_cost = pre_accuracy - post_accuracy  # positive = switching hurts
    perseveration_rate = perseveration_errors / n_post if n_post > 0 else 0.0
    overall_accuracy = (pre_correct + post_correct) / (n_pre + n_post)

    # Post-switch error breakdown
    post_errors_total = n_post - post_correct
    residue_rate = residue_errors / n_post if n_post > 0 else 0.0
    random_error_rate = random_errors / n_post if n_post > 0 else 0.0

    result.add_metric("overall_accuracy", overall_accuracy)
    result.add_metric("pre_switch_accuracy", pre_accuracy)
    result.add_metric("post_switch_accuracy", post_accuracy)
    result.add_metric("switch_cost", switch_cost)
    result.add_metric("perseveration_rate", perseveration_rate)
    result.add_metric("perseveration_errors", perseveration_errors)
    result.add_metric("residue_error_count", residue_errors)
    result.add_metric("residue_rate", residue_rate)
    result.add_metric("random_error_count", random_errors)
    result.add_metric("random_error_rate", random_error_rate)
    result.details = details

    return result


def score_ior(instance, response: str) -> ScoreResult:
    """Score an Inhibition of Return response."""
    result = ScoreResult(instance.task_id, "inhibition_return", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer  # {"1": answer, "2": answer, ...}
    questions = instance.metadata["questions"]

    answers = extract_numbered_answers(response)

    correct_by_phase = {"initial": 0, "shift": 0, "return": 0}
    total_by_phase = {"initial": 0, "shift": 0, "return": 0}
    details = {}

    for q in questions:
        qi = str(q["question_num"])
        phase = q["phase"]
        gold_val = gold[qi]
        pred = answers.get(qi, "")

        total_by_phase[phase] += 1
        is_correct = fuzzy_value_match(pred, gold_val) if pred else False
        if is_correct:
            correct_by_phase[phase] += 1

        details[qi] = {
            "phase": phase,
            "passage": q["passage"],
            "predicted": pred,
            "gold": gold_val,
            "correct": is_correct,
        }

    total_correct = sum(correct_by_phase.values())
    total_items = sum(total_by_phase.values())
    overall_accuracy = total_correct / total_items if total_items > 0 else 0.0

    initial_acc = correct_by_phase["initial"] / total_by_phase["initial"] if total_by_phase["initial"] > 0 else 0.0
    return_acc = correct_by_phase["return"] / total_by_phase["return"] if total_by_phase["return"] > 0 else 0.0
    return_penalty = initial_acc - return_acc  # positive = return costs accuracy

    result.add_metric("overall_accuracy", overall_accuracy)
    result.add_metric("initial_accuracy", initial_acc)
    result.add_metric("return_accuracy", return_acc)
    result.add_metric("return_penalty", return_penalty)
    result.details = details

    return result
