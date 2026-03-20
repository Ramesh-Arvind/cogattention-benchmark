"""
Scorer for Task G: Visual Inattentional Blindness.

Metrics:
- primary_accuracy: count matches within tolerance (±1 for Hard+)
- detection_rate: unexpected stimulus correctly noticed/denied
- false_alarm_rate: claiming unusual when nothing is there
- ib_composite: weighted 0.4 * primary + 0.6 * detection
"""

import re
from .base import ScoreResult, extract_numbered_answers
from ..generators.base import TaskInstance


def score_visual_inattentional(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Visual Inattentional Blindness response."""
    result = ScoreResult(instance.task_id, "visual_inattentional", instance.difficulty)
    result.raw_response = response

    gold = instance.gold_answer
    gold_count = gold["count"]
    unexpected_present = gold["unexpected_present"]
    unexpected_desc = gold.get("unexpected_description", "")

    answers = extract_numbered_answers(response)

    # --- Primary task: count accuracy ---
    pred_count_str = answers.get("1", "")
    primary_score = 0.0
    pred_count = None
    try:
        nums = re.findall(r'\d+', pred_count_str)
        if nums:
            pred_count = int(nums[0])
            # Tolerance: ±0 for Easy/Medium, ±1 for Hard+
            tolerance = 0 if instance.difficulty in ("Easy", "Medium") else 1
            if abs(pred_count - gold_count) <= tolerance:
                primary_score = 1.0
    except (ValueError, TypeError):
        pass

    # --- Secondary task: unexpected stimulus detection ---
    pred_unusual = answers.get("2", "")
    detection_score = 0.0
    false_alarm = 0.0

    if pred_unusual:
        lower_resp = pred_unusual.lower()
        negatives = [
            "nothing unusual", "nothing out of place", "no unusual",
            "didn't notice", "did not notice", "nothing strange",
            "nothing abnormal", "no anomal", "nothing", "none",
        ]
        is_negative = any(neg in lower_resp for neg in negatives)

        if unexpected_present:
            # Should detect: check for any description of the stimulus
            if not is_negative:
                # Check for keywords matching the unexpected type
                type_keywords = {
                    "star": ["star", "pointed", "asterisk"],
                    "arrow": ["arrow", "pointer", "chevron", "pointing"],
                    "cross": ["cross", "plus", "+", "intersection"],
                    "dot_pattern": ["dot", "cluster", "spots", "speckle"],
                    "gradient_patch": ["gradient", "patch", "fade", "blur", "shadow"],
                }
                ux_type = instance.metadata.get("unexpected_type", "")
                keywords = type_keywords.get(ux_type, [])
                # Also accept any generic "unusual" detection
                if keywords and any(kw in lower_resp for kw in keywords):
                    detection_score = 1.0
                elif len(lower_resp) > 5:
                    # Partial credit for noticing something, even if description is off
                    detection_score = 0.5
        else:
            # No unexpected stimulus: correct if negative response
            if is_negative:
                detection_score = 1.0
            else:
                false_alarm = 1.0

    # Composite score: same weighting as anomaly scorer
    ib_composite = 0.4 * primary_score + 0.6 * detection_score

    result.add_metric("primary_accuracy", primary_score)
    result.add_metric("detection_rate", detection_score)
    result.add_metric("false_alarm_rate", false_alarm)
    result.add_metric("ib_composite", ib_composite)
    result.details = {
        "pred_count": pred_count,
        "gold_count": gold_count,
        "primary_correct": primary_score == 1.0,
        "pred_unusual": pred_unusual,
        "unexpected_present": unexpected_present,
        "unexpected_description": unexpected_desc,
        "detection_correct": detection_score == 1.0,
        "false_alarm": false_alarm == 1.0,
        "saliency": instance.metadata.get("saliency", ""),
    }

    return result
