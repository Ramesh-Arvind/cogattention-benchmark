"""
Kaggle Benchmarks SDK assertion logic for CogAttention benchmark.

Maps each task type's scoring to kbench assertions with fine-grained
per-element assertions for continuous scoring.

All task functions share the same 5-param signature:
    def task_fn(llm, prompt, gold_json, task_id, difficulty)

gold_json is a JSON string containing everything needed for assertions.
"""

import json
import re


# ── Inline scoring helpers (subset of src/scorers/base.py) ────────────
# These are also embedded in notebooks so they work on Kaggle.

def extract_answer_block(response: str) -> str:
    """Extract text after 'ANSWER:' marker from model response."""
    for pat in [r"ANSWER:\s*(.*)", r"Answer:\s*(.*)", r"answer:\s*(.*)"]:
        match = re.search(pat, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()


def extract_numbered_answers(response: str) -> dict:
    """Extract numbered answers like '1. value' or '1: value'."""
    answer_block = extract_answer_block(response)
    results = {}
    matches = re.findall(
        r"(\d+)\s*[.):\-]\s*(.+?)(?=\n\d+\s*[.):\-]|\Z)",
        answer_block, re.DOTALL,
    )
    for num, val in matches:
        results[num] = val.strip().rstrip(".")
    return results


def extract_list_items(response: str) -> list:
    """Extract items from a list in the answer block."""
    answer_block = extract_answer_block(response)
    bullets = re.findall(r"[-•]\s*(.+?)(?:\n|$)", answer_block)
    if bullets:
        return [b.strip().rstrip(".") for b in bullets]
    numeric_items = re.findall(
        r'[\$]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:°[CF]|mg/L|%|\$))?',
        answer_block,
    )
    if numeric_items and len(numeric_items) >= 2:
        return [x.strip() for x in numeric_items]
    if "," in answer_block:
        items = [x.strip().rstrip(".") for x in answer_block.split(",")]
        return [x for x in items if x]
    lines = [l.strip().rstrip(".") for l in answer_block.split("\n") if l.strip()]
    return lines if lines else ([answer_block] if answer_block else [])


def extract_person_item_pairs(response: str) -> dict:
    """Extract 'Person: item' pairs."""
    answer_block = extract_answer_block(response)
    results = {}
    for pat in [
        r"[-•]?\s*(\w+)\s*:\s*(.+?)(?:\n|$)",
        r"[-•]?\s*(\w+)\s+holds?\s+(?:a\s+)?(.+?)(?:\n|$)",
    ]:
        matches = re.findall(pat, answer_block, re.IGNORECASE)
        if matches:
            for name, item in matches:
                results[name.strip()] = item.strip().rstrip(".")
            break
    return results


def fuzzy_value_match(predicted: str, gold: str) -> bool:
    """Check if predicted matches gold (case-insensitive, substring, numeric)."""
    pred_clean = re.sub(r"\s+", " ", predicted.strip().lower())
    gold_clean = re.sub(r"\s+", " ", gold.strip().lower())
    if pred_clean == gold_clean:
        return True
    if gold_clean in pred_clean:
        return True
    try:
        pred_num = float(re.sub(r"[,$%°]", "", predicted))
        gold_num = float(re.sub(r"[,$%°]", "", gold))
        return pred_num == gold_num
    except (ValueError, TypeError):
        pass
    return False


# ── Build gold_json for each task type ────────────────────────────────

def build_gold_json(instance) -> str:
    """Serialize a TaskInstance's gold data into a JSON string for SDK tasks."""
    t = instance.task_type
    meta = instance.metadata

    if t == "capacity":
        data = {
            "answers": instance.gold_answer,
            "people": meta["people"],
        }
    elif t == "interference":
        data = {
            "final_values": instance.gold_answer,
            "key_names": meta["key_names"],
        }
    elif t == "sustained":
        data = {
            "targets": instance.gold_answer,
            "nearmisses": meta["nearmisses"],
        }
    elif t == "stream_segregation":
        data = {
            "first_number": instance.gold_answer.get("1", "unknown"),
            "has_breakthrough": meta["has_breakthrough"],
        }
    elif t == "selective":
        data = {
            "signals": instance.gold_answer,
            "distractors": meta["distractor_values"],
        }
    elif t == "stroop":
        data = {
            "answers": instance.gold_answer,
            "traps": meta["trap_answers"],
        }
    elif t == "shifting":
        data = {
            "answers": instance.gold_answer,
        }
    elif t == "anomaly":
        data = {
            "primary_answer": str(instance.gold_answer["primary"]),
            "detection_keywords": meta["detection_keywords"],
            "anomaly_type": instance.gold_answer["anomaly_type"],
        }
    elif t == "blink":
        data = {
            "t1": instance.gold_answer["T1"],
            "t2": instance.gold_answer["T2"],
        }
    elif t == "flanker":
        data = {
            "gold_value": str(instance.gold_answer),
        }
    elif t == "inhibition_return":
        data = {
            "answers": instance.gold_answer,
            "questions": meta["questions"],
        }
    elif t == "context_dilution":
        data = {
            "gold_value": str(instance.gold_answer),
        }
    elif t == "semantic_niah":
        data = {
            "gold_value": str(instance.gold_answer),
        }
    elif t == "multihop":
        data = {
            "gold_value": str(instance.gold_answer),
        }
    else:
        raise ValueError(f"Unknown task type: {t}")

    return json.dumps(data, ensure_ascii=False)


# ── Assertion functions per task type ─────────────────────────────────
# Each returns a list of (pattern, expectation, passed) tuples.
# In notebooks, we use assert_contains_regex with patterns that do
# contextual matching (person→item, idx→answer, etc).

def _escape_for_regex(s: str) -> str:
    """Escape a string for use in regex, but allow flexible whitespace."""
    return re.escape(s).replace(r"\ ", r"\s+")


def assert_capacity(response: str, gold_json: str) -> list:
    """Per-person assertions for capacity (thread tracking)."""
    gold = json.loads(gold_json)
    results = []
    predicted = extract_person_item_pairs(response)
    for person in gold["people"]:
        gold_item = gold["answers"][person]
        # Pattern: person name followed by the gold item somewhere on the same line
        pattern = rf"(?i){re.escape(person)}\s*[:.\-]\s*.*{_escape_for_regex(gold_item)}"
        pred_item = None
        for pred_name, pred_val in predicted.items():
            if pred_name.lower() == person.lower():
                pred_item = pred_val
                break
        passed = pred_item is not None and fuzzy_value_match(pred_item, gold_item)
        expectation = f"{person} should hold '{gold_item}'"
        results.append((pattern, expectation, passed))
    return results


def assert_interference(response: str, gold_json: str) -> list:
    """Per-key assertions for interference."""
    gold = json.loads(gold_json)
    results = []
    answer_block = extract_answer_block(response)
    for key in gold["key_names"]:
        gold_val = gold["final_values"][key]
        pattern = rf"(?i){re.escape(key)}\s*[:\-=]\s*.*{_escape_for_regex(gold_val)}"
        key_pat = rf"(?i){re.escape(key)}\s*[:\-=]\s*['\"]?(.+?)['\"]?\s*(?:\n|$)"
        match = re.search(key_pat, answer_block)
        pred_val = match.group(1).strip().rstrip(".") if match else ""
        passed = bool(pred_val) and fuzzy_value_match(pred_val, gold_val)
        expectation = f"Final value of '{key}' should be '{gold_val}'"
        results.append((pattern, expectation, passed))
    return results


def assert_sustained(response: str, gold_json: str) -> list:
    """Per-target assertions for sustained."""
    gold = json.loads(gold_json)
    results = []
    # Use simple case-insensitive substring check on full response
    # rather than extract_list_items (which can be fooled by hyphens)
    lower_response = response.lower()
    for target in gold["targets"]:
        pattern = rf"(?i){_escape_for_regex(target)}"
        passed = target.lower() in lower_response
        expectation = f"Should find target '{target}'"
        results.append((pattern, expectation, passed))
    return results


def assert_stream_segregation(response: str, gold_json: str) -> list:
    """1-2 assertions for stream segregation."""
    gold = json.loads(gold_json)
    results = []
    answers = extract_numbered_answers(response)

    gold_num = gold["first_number"]
    if gold_num and gold_num != "unknown":
        pattern = rf"\b{re.escape(gold_num)}\b"
        pred_q1 = answers.get("1", "")
        if not pred_q1:
            ab = extract_answer_block(response)
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", ab)
            if nums:
                pred_q1 = ab
        passed = fuzzy_value_match(pred_q1, gold_num) if pred_q1 else False
        expectation = f"First number in stream A should be '{gold_num}'"
        results.append((pattern, expectation, passed))

    if gold["has_breakthrough"]:
        pattern = r"(?i)\byes\b"
        pred_q2 = answers.get("2", "").lower()
        passed = "yes" in pred_q2
        expectation = "Should detect ALERT breakthrough in stream B"
        results.append((pattern, expectation, passed))

    return results


def assert_selective(response: str, gold_json: str) -> list:
    """Per-signal assertions for selective."""
    gold = json.loads(gold_json)
    results = []
    # Use substring check on response — extract_list_items can fail on
    # negative numbers and special chars like °C, mg/L
    lower_response = response.lower()
    for signal in gold["signals"]:
        pattern = rf"(?i){_escape_for_regex(signal)}"
        passed = signal.lower() in lower_response
        expectation = f"Should extract signal value '{signal}'"
        results.append((pattern, expectation, passed))
    return results


def assert_stroop(response: str, gold_json: str) -> list:
    """Per-item assertions for stroop."""
    gold = json.loads(gold_json)
    results = []
    answers = extract_numbered_answers(response)
    for idx_str, gold_val in gold["answers"].items():
        pattern = rf"(?i){re.escape(idx_str)}\s*[.):\-]\s*.*{_escape_for_regex(gold_val)}"
        pred = answers.get(idx_str, "")
        passed = fuzzy_value_match(pred, gold_val) if pred else False
        trap = gold["traps"].get(idx_str)
        trap_note = f" (trap: '{trap}')" if trap else ""
        expectation = f"Item {idx_str} should be '{gold_val}'{trap_note}"
        results.append((pattern, expectation, passed))
    return results


def assert_shifting(response: str, gold_json: str) -> list:
    """Per-item assertions for shifting."""
    gold = json.loads(gold_json)
    results = []
    answers = extract_numbered_answers(response)
    for idx_str, gold_val in gold["answers"].items():
        pattern = rf"(?i){re.escape(idx_str)}\s*[.):\-]\s*.*{_escape_for_regex(gold_val)}"
        pred = answers.get(idx_str, "")
        passed = fuzzy_value_match(pred, gold_val) if pred else False
        expectation = f"Item {idx_str} should be classified as '{gold_val}'"
        results.append((pattern, expectation, passed))
    return results


def assert_anomaly(response: str, gold_json: str) -> list:
    """2 assertions for anomaly."""
    gold = json.loads(gold_json)
    results = []
    answers = extract_numbered_answers(response)

    # Primary answer
    gold_primary = gold["primary_answer"]
    pred_primary = answers.get("1", "")
    try:
        pred_num = float(re.sub(r"[,$]", "", pred_primary))
        gold_num = float(gold_primary)
        primary_passed = abs(pred_num - gold_num) < 0.01
    except (ValueError, TypeError):
        primary_passed = (
            gold_primary.lower() in pred_primary.lower() if pred_primary else False
        )
    pattern_primary = rf"(?i)1\s*[.):\-]\s*.*{re.escape(gold_primary)}"
    expectation_primary = f"Primary answer should be '{gold_primary}'"
    results.append((pattern_primary, expectation_primary, primary_passed))

    # Anomaly detection
    pred_anomaly = answers.get("2", "")
    anomaly_passed = False
    if pred_anomaly:
        lower_resp = pred_anomaly.lower()
        negatives = [
            "nothing unusual", "nothing out of place", "no unusual",
            "didn't notice", "did not notice", "nothing strange",
            "nothing abnormal", "no anomal",
        ]
        is_negative = any(neg in lower_resp for neg in negatives)
        if not is_negative:
            anomaly_passed = any(
                kw.lower() in lower_resp for kw in gold["detection_keywords"]
            )
            if not anomaly_passed:
                type_hints = {
                    "language_switch": [
                        "french", "spanish", "german", "italian",
                        "foreign", "language", "non-english",
                    ],
                    "code_block": [
                        "code", "programming", "function", "sql",
                        "script", "variable", "syntax",
                    ],
                    "factual_absurdity": [
                        "incorrect", "wrong", "false", "error",
                        "impossible", "inaccurate", "absurd",
                    ],
                    "numerical_anomaly": [
                        "large", "amount", "expensive", "unusual",
                        "million", "extraordinary", "outlier",
                    ],
                    "name_inconsistency": [
                        "name", "changed", "inconsistent",
                        "different", "same person", "alias",
                    ],
                }
                hints = type_hints.get(gold["anomaly_type"], [])
                anomaly_passed = any(h in lower_resp for h in hints)

    kw_sample = gold["detection_keywords"][0] if gold["detection_keywords"] else "anomaly"
    pattern_anomaly = rf"(?i){re.escape(kw_sample)}"
    expectation_anomaly = f"Should detect {gold['anomaly_type']} anomaly"
    results.append((pattern_anomaly, expectation_anomaly, anomaly_passed))

    return results


def assert_blink(response: str, gold_json: str) -> list:
    """Per-target assertions for attentional blink."""
    gold = json.loads(gold_json)
    results = []
    answer_block = extract_answer_block(response)

    for target_key in ["t1", "t2"]:
        gold_val = gold[target_key]
        label = target_key.upper()
        pat = rf"(?i){re.escape(label)}\s*[:\-=]\s*(.+?)(?:\n|$)"
        match = re.search(pat, answer_block)
        pred = match.group(1).strip().rstrip(".") if match else ""
        passed = fuzzy_value_match(pred, gold_val) if pred else False
        pattern = rf"(?i){re.escape(label)}\s*[:\-=]\s*.*{_escape_for_regex(gold_val)}"
        expectation = f"{label} should be '{gold_val}'"
        results.append((pattern, expectation, passed))

    return results


def assert_flanker(response: str, gold_json: str) -> list:
    """Single assertion for flanker."""
    gold = json.loads(gold_json)
    results = []
    answer_block = extract_answer_block(response)
    gold_val = gold["gold_value"]
    passed = fuzzy_value_match(answer_block, gold_val)
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    expectation = f"Should extract target value '{gold_val}'"
    results.append((pattern, expectation, passed))
    return results


def assert_context_dilution(response: str, gold_json: str) -> list:
    """Single assertion for context dilution."""
    gold = json.loads(gold_json)
    answer_block = extract_answer_block(response)
    gold_val = gold["gold_value"]
    passed = fuzzy_value_match(answer_block, gold_val)
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    return [(pattern, f"Should find '{gold_val}' despite context length", passed)]


def assert_semantic_niah(response: str, gold_json: str) -> list:
    """Single assertion for semantic NIAH."""
    gold = json.loads(gold_json)
    answer_block = extract_answer_block(response)
    gold_val = gold["gold_value"]
    passed = fuzzy_value_match(answer_block, gold_val)
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    return [(pattern, f"Should find semantic needle '{gold_val}'", passed)]


def assert_multihop(response: str, gold_json: str) -> list:
    """Single assertion for multi-hop scattered reasoning."""
    gold = json.loads(gold_json)
    answer_block = extract_answer_block(response)
    gold_val = gold["gold_value"]
    passed = fuzzy_value_match(answer_block, gold_val)
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    return [(pattern, f"Should chain facts to find '{gold_val}'", passed)]


def assert_inhibition_return(response: str, gold_json: str) -> list:
    """Per-question assertions for inhibition of return."""
    gold = json.loads(gold_json)
    results = []
    answers = extract_numbered_answers(response)
    for idx_str, gold_val in gold["answers"].items():
        pattern = rf"(?i){re.escape(idx_str)}\s*[.):\-]\s*.*{_escape_for_regex(gold_val)}"
        pred = answers.get(idx_str, "")
        passed = fuzzy_value_match(pred, gold_val) if pred else False
        # Find the phase for this question
        phase = "unknown"
        for q in gold["questions"]:
            if str(q["question_num"]) == idx_str:
                phase = q["phase"]
                break
        expectation = f"Q{idx_str} ({phase}) should be '{gold_val}'"
        results.append((pattern, expectation, passed))
    return results


# ── Dispatcher ────────────────────────────────────────────────────────

ASSERTION_DISPATCH = {
    "capacity": assert_capacity,
    "interference": assert_interference,
    "blink": assert_blink,
    "sustained": assert_sustained,
    "stream_segregation": assert_stream_segregation,
    "context_dilution": assert_context_dilution,
    "semantic_niah": assert_semantic_niah,
    "multihop": assert_multihop,
    "selective": assert_selective,
    "stroop": assert_stroop,
    "flanker": assert_flanker,
    "shifting": assert_shifting,
    "inhibition_return": assert_inhibition_return,
    "anomaly": assert_anomaly,
}


def run_assertions(task_type: str, response: str, gold_json: str) -> list:
    """Run assertions for a task type. Returns list of (pattern, expectation, passed)."""
    fn = ASSERTION_DISPATCH.get(task_type)
    if fn is None:
        raise ValueError(f"No assertion function for task type: {task_type}")
    return fn(response, gold_json)


# ── Source code for embedding in notebooks ────────────────────────────

INLINE_HELPERS_SOURCE = r'''
import json
import re

def extract_answer_block(response):
    for pat in [r"ANSWER:\s*(.*)", r"Answer:\s*(.*)", r"answer:\s*(.*)"]:
        match = re.search(pat, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return response.strip()

def extract_numbered_answers(response):
    answer_block = extract_answer_block(response)
    results = {}
    matches = re.findall(
        r"(\d+)\s*[.):\-]\s*(.+?)(?=\n\d+\s*[.):\-]|\Z)",
        answer_block, re.DOTALL,
    )
    for num, val in matches:
        results[num] = val.strip().rstrip(".")
    return results

def extract_list_items(response):
    answer_block = extract_answer_block(response)
    bullets = re.findall(r"[-\u2022]\s*(.+?)(?:\n|$)", answer_block)
    if bullets:
        return [b.strip().rstrip(".") for b in bullets]
    numeric_items = re.findall(
        r'[\$]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:\xb0[CF]|mg/L|%|\$))?',
        answer_block,
    )
    if numeric_items and len(numeric_items) >= 2:
        return [x.strip() for x in numeric_items]
    if "," in answer_block:
        items = [x.strip().rstrip(".") for x in answer_block.split(",")]
        return [x for x in items if x]
    lines = [l.strip().rstrip(".") for l in answer_block.split("\n") if l.strip()]
    return lines if lines else ([answer_block] if answer_block else [])

def extract_person_item_pairs(response):
    answer_block = extract_answer_block(response)
    results = {}
    for pat in [
        r"[-\u2022]?\s*(\w+)\s*:\s*(.+?)(?:\n|$)",
        r"[-\u2022]?\s*(\w+)\s+holds?\s+(?:a\s+)?(.+?)(?:\n|$)",
    ]:
        matches = re.findall(pat, answer_block, re.IGNORECASE)
        if matches:
            for name, item in matches:
                results[name.strip()] = item.strip().rstrip(".")
            break
    return results

def fuzzy_value_match(predicted, gold):
    pred_clean = re.sub(r"\s+", " ", predicted.strip().lower())
    gold_clean = re.sub(r"\s+", " ", gold.strip().lower())
    if pred_clean == gold_clean:
        return True
    if gold_clean in pred_clean:
        return True
    try:
        pred_num = float(re.sub(r"[,$%\xb0]", "", predicted))
        gold_num = float(re.sub(r"[,$%\xb0]", "", gold))
        return pred_num == gold_num
    except (ValueError, TypeError):
        pass
    return False

def _escape_for_regex(s):
    return re.escape(s).replace(r"\ ", r"\s+")
'''


def _get_assertion_runner_source(task_types: list) -> str:
    """Generate assertion runner functions for embedding in notebooks.

    These use kbench.assertions.assert_contains_regex directly.
    The approach: build a contextual regex pattern and match against response.
    For correct answers, the pattern will naturally match. For incorrect ones, it won't.
    """
    runners = {}

    runners["capacity"] = r'''
def run_assertions_capacity(response, gold, kbench):
    for person in gold["people"]:
        gold_item = gold["answers"][person]
        pattern = rf"(?i){re.escape(person)}\s*[:.\\-]\s*.*{_escape_for_regex(gold_item)}"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"{person} should hold '{gold_item}'"
        )
'''

    runners["interference"] = r'''
def run_assertions_interference(response, gold, kbench):
    for key in gold["key_names"]:
        gold_val = gold["final_values"][key]
        pattern = rf"(?i){re.escape(key)}\s*[:\-=]\s*.*{_escape_for_regex(gold_val)}"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"Final value of '{key}' should be '{gold_val}'"
        )
'''

    runners["sustained"] = r'''
def run_assertions_sustained(response, gold, kbench):
    for target in gold["targets"]:
        pattern = rf"(?i){_escape_for_regex(target)}"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"Should find target '{target}'"
        )
'''

    runners["stream_segregation"] = r'''
def run_assertions_stream_segregation(response, gold, kbench):
    gold_num = gold["first_number"]
    if gold_num and gold_num != "unknown":
        pattern = rf"\b{re.escape(gold_num)}\b"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"First number in stream A should be '{gold_num}'"
        )
    if gold["has_breakthrough"]:
        kbench.assertions.assert_contains_regex(
            r"(?i)\byes\b", response,
            expectation="Should detect ALERT breakthrough in stream B"
        )
'''

    runners["selective"] = r'''
def run_assertions_selective(response, gold, kbench):
    for signal in gold["signals"]:
        pattern = rf"(?i){_escape_for_regex(signal)}"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"Should extract signal value '{signal}'"
        )
'''

    runners["stroop"] = r'''
def run_assertions_stroop(response, gold, kbench):
    for idx_str, gold_val in gold["answers"].items():
        pattern = rf"(?i){re.escape(idx_str)}\s*[.):\-]\s*.*{_escape_for_regex(gold_val)}"
        trap = gold["traps"].get(idx_str)
        trap_note = f" (trap: '{trap}')" if trap else ""
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"Item {idx_str} should be '{gold_val}'{trap_note}"
        )
'''

    runners["shifting"] = r'''
def run_assertions_shifting(response, gold, kbench):
    for idx_str, gold_val in gold["answers"].items():
        pattern = rf"(?i){re.escape(idx_str)}\s*[.):\-]\s*.*{_escape_for_regex(gold_val)}"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"Item {idx_str} should be classified as '{gold_val}'"
        )
'''

    runners["anomaly"] = r'''
def run_assertions_anomaly(response, gold, kbench):
    gold_primary = gold["primary_answer"]
    pattern_primary = rf"(?i)1\s*[.):\-]\s*.*{re.escape(gold_primary)}"
    kbench.assertions.assert_contains_regex(
        pattern_primary, response,
        expectation=f"Primary answer should be '{gold_primary}'"
    )
    kw_sample = gold["detection_keywords"][0] if gold["detection_keywords"] else "anomaly"
    pattern_anomaly = rf"(?i){re.escape(kw_sample)}"
    kbench.assertions.assert_contains_regex(
        pattern_anomaly, response,
        expectation=f"Should detect {gold['anomaly_type']} anomaly"
    )
'''

    runners["context_dilution"] = r'''
def run_assertions_context_dilution(response, gold, kbench):
    gold_val = gold["gold_value"]
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    kbench.assertions.assert_contains_regex(
        pattern, response,
        expectation=f"Should find '{gold_val}' despite context length"
    )
'''

    runners["semantic_niah"] = r'''
def run_assertions_semantic_niah(response, gold, kbench):
    gold_val = gold["gold_value"]
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    kbench.assertions.assert_contains_regex(
        pattern, response,
        expectation=f"Should find semantic needle '{gold_val}'"
    )
'''

    runners["multihop"] = r'''
def run_assertions_multihop(response, gold, kbench):
    gold_val = gold["gold_value"]
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    kbench.assertions.assert_contains_regex(
        pattern, response,
        expectation=f"Should chain facts to find '{gold_val}'"
    )
'''

    runners["blink"] = r'''
def run_assertions_blink(response, gold, kbench):
    for target_key in ["t1", "t2"]:
        gold_val = gold[target_key]
        label = target_key.upper()
        pattern = rf"(?i){re.escape(label)}\s*[:\-=]\s*.*{_escape_for_regex(gold_val)}"
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"{label} should be '{gold_val}'"
        )
'''

    runners["flanker"] = r'''
def run_assertions_flanker(response, gold, kbench):
    gold_val = gold["gold_value"]
    pattern = rf"(?i){_escape_for_regex(gold_val)}"
    kbench.assertions.assert_contains_regex(
        pattern, response,
        expectation=f"Should extract target value '{gold_val}'"
    )
'''

    runners["inhibition_return"] = r'''
def run_assertions_inhibition_return(response, gold, kbench):
    for idx_str, gold_val in gold["answers"].items():
        pattern = rf"(?i){re.escape(idx_str)}\s*[.):\-]\s*.*{_escape_for_regex(gold_val)}"
        phase = "unknown"
        for q in gold["questions"]:
            if str(q["question_num"]) == idx_str:
                phase = q["phase"]
                break
        kbench.assertions.assert_contains_regex(
            pattern, response,
            expectation=f"Q{idx_str} ({phase}) should be '{gold_val}'"
        )
'''

    parts = []
    for tt in task_types:
        if tt in runners:
            parts.append(runners[tt])
    return "\n".join(parts)
