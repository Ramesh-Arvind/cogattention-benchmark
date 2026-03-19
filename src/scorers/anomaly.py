"""
Scorer for Task E: Anomaly Detection (Stimulus-Driven Attention).
"""

import re
from .base import ScoreResult, extract_numbered_answers, extract_answer_block
from ..generators.base import TaskInstance


def score_anomaly(instance: TaskInstance, response: str) -> ScoreResult:
    """Score an Anomaly Detection response."""
    result = ScoreResult(instance.task_id, "anomaly", instance.difficulty)
    result.raw_response = response

    gold_primary = instance.gold_answer["primary"]
    anomaly_type = instance.gold_answer["anomaly_type"]
    detection_keywords = instance.metadata["detection_keywords"]

    answers = extract_numbered_answers(response)

    # Question 1: primary task accuracy
    pred_primary = answers.get("1", "")
    primary_correct = False

    # Try numeric comparison
    try:
        pred_num = float(re.sub(r'[,$]', '', pred_primary))
        gold_num = float(gold_primary)
        primary_correct = abs(pred_num - gold_num) < 0.01
    except (ValueError, TypeError):
        # String comparison
        primary_correct = str(gold_primary).lower() in pred_primary.lower()

    # Question 2: anomaly detection
    pred_anomaly = answers.get("2", "")
    anomaly_detected = False

    if pred_anomaly:
        lower_resp = pred_anomaly.lower()
        # Check if "nothing unusual" (negative response)
        negatives = ["nothing unusual", "nothing out of place", "no unusual",
                     "didn't notice", "did not notice", "nothing strange",
                     "nothing abnormal", "no anomal"]
        is_negative = any(neg in lower_resp for neg in negatives)

        if not is_negative:
            # Check for detection keywords
            anomaly_detected = any(kw.lower() in lower_resp for kw in detection_keywords)
            # Also check if the response describes the anomaly type
            if not anomaly_detected:
                type_hints = {
                    "language_switch": ["french", "spanish", "german", "italian",
                                        "foreign", "language", "non-english"],
                    "code_block": ["code", "programming", "function", "sql",
                                   "script", "variable", "syntax"],
                    "factual_absurdity": ["incorrect", "wrong", "false", "error",
                                          "impossible", "inaccurate", "absurd"],
                    "numerical_anomaly": ["large", "amount", "expensive", "unusual",
                                          "million", "extraordinary", "outlier"],
                    "name_inconsistency": ["name", "changed", "inconsistent",
                                           "different", "same person", "alias"],
                }
                hints = type_hints.get(anomaly_type, [])
                anomaly_detected = any(h in lower_resp for h in hints)

    primary_score = 1.0 if primary_correct else 0.0
    detection_score = 1.0 if anomaly_detected else 0.0
    # Weighted average: primary task (40%) + anomaly detection (60%)
    # Using multiplication causes floor effect (0 × anything = 0)
    dual_task_score = 0.4 * primary_score + 0.6 * detection_score

    # Cognitive load effect: did primary task accuracy suffer when anomaly was detected?
    # (Can only be analyzed across multiple instances — stored for aggregation)

    result.add_metric("primary_accuracy", primary_score)
    result.add_metric("anomaly_detection", detection_score)
    result.add_metric("dual_task_score", dual_task_score)
    result.details = {
        "pred_primary": pred_primary,
        "gold_primary": gold_primary,
        "primary_correct": primary_correct,
        "pred_anomaly": pred_anomaly,
        "anomaly_detected": anomaly_detected,
        "anomaly_type": anomaly_type,
        "anomaly_saliency": instance.metadata["anomaly_saliency"],
    }

    return result
