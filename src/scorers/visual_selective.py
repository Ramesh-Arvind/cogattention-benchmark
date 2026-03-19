"""
Scorer for Task F: Visual Stroop (Multimodal VLM Extension).

Scores whether VLMs can report ink color rather than reading the word.
Classifies errors as architectural prior failures (OCR pathway overpowering
color perception) vs random errors.
"""

from .base import ScoreResult, extract_answer_block, extract_numbered_answers, fuzzy_value_match
from ..generators.base import TaskInstance


COLOR_ALIASES = {
    "red": ["red", "crimson", "scarlet"],
    "blue": ["blue", "navy", "azure"],
    "green": ["green", "lime", "emerald"],
    "yellow": ["yellow", "gold", "golden"],
    "orange": ["orange", "amber"],
    "purple": ["purple", "violet", "magenta"],
    "pink": ["pink", "rose", "fuchsia"],
    "brown": ["brown", "tan", "chocolate"],
}


def _match_color(predicted: str, target: str) -> bool:
    """Check if predicted color matches target, allowing common aliases."""
    pred_lower = predicted.strip().lower()
    target_lower = target.strip().lower()

    if pred_lower == target_lower:
        return True

    aliases = COLOR_ALIASES.get(target_lower, [target_lower])
    return pred_lower in aliases


def score_visual_stroop(instance: TaskInstance, response: str) -> ScoreResult:
    """Score a Visual Stroop response."""
    result = ScoreResult(instance.task_id, "visual_stroop", instance.difficulty)
    result.raw_response = response

    n_items = instance.metadata["n_items"]
    trap_answers = instance.metadata["trap_answers"]
    items = instance.metadata["items"]

    correct = 0
    stroop_errors = 0  # Model read the word instead of reporting ink color
    random_errors = 0
    details = {}

    if n_items == 1:
        # Single item — extract from answer block
        gold = instance.gold_answer  # string
        trap = trap_answers.get("1", "")
        pred = extract_answer_block(response).strip().lower()

        if _match_color(pred, gold):
            correct = 1
            details["1"] = {"status": "correct", "predicted": pred, "gold": gold}
        elif _match_color(pred, trap):
            stroop_errors = 1
            details["1"] = {"status": "stroop_error", "predicted": pred, "gold": gold, "trap": trap}
        else:
            random_errors = 1
            details["1"] = {"status": "random_error", "predicted": pred, "gold": gold}
    else:
        # Multiple items — extract numbered answers
        gold = instance.gold_answer  # Dict[str, str]
        answers = extract_numbered_answers(response)

        for idx_str, gold_val in gold.items():
            pred = answers.get(idx_str, "").strip().lower()
            trap = trap_answers.get(idx_str, "")

            if _match_color(pred, gold_val):
                correct += 1
                details[idx_str] = {"status": "correct", "predicted": pred, "gold": gold_val}
            elif _match_color(pred, trap):
                stroop_errors += 1
                details[idx_str] = {
                    "status": "stroop_error", "predicted": pred,
                    "gold": gold_val, "trap": trap,
                }
            else:
                random_errors += 1
                details[idx_str] = {"status": "random_error", "predicted": pred, "gold": gold_val}

    total = n_items
    accuracy = correct / total if total > 0 else 0.0
    stroop_error_rate = stroop_errors / total if total > 0 else 0.0
    stroop_resistance = 1.0 - stroop_error_rate

    result.add_metric("accuracy", accuracy)
    result.add_metric("stroop_error_rate", stroop_error_rate)
    result.add_metric("stroop_resistance", stroop_resistance)
    result.add_metric("correct_count", correct)
    result.add_metric("stroop_errors", stroop_errors)
    result.add_metric("random_errors", random_errors)
    result.details = details

    return result
