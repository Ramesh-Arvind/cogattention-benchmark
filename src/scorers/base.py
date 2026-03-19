"""
Base scoring utilities for CogAttention benchmark.
Handles response parsing, regex extraction, and common scoring patterns.
"""

import re
from typing import List, Optional, Dict, Any


def extract_answer_block(response: str) -> str:
    """Extract text after 'ANSWER:' marker from model response."""
    patterns = [
        r"ANSWER:\s*(.*)",
        r"Answer:\s*(.*)",
        r"answer:\s*(.*)",
    ]
    for pat in patterns:
        match = re.search(pat, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Fallback: return entire response
    return response.strip()


def extract_numbered_answers(response: str) -> Dict[str, str]:
    """Extract numbered answers like '1. value' or '1: value' from response."""
    answer_block = extract_answer_block(response)
    results = {}
    # Match patterns: "1. answer", "1: answer", "1) answer"
    matches = re.findall(
        r"(\d+)\s*[.):\-]\s*(.+?)(?=\n\d+\s*[.):\-]|\Z)",
        answer_block,
        re.DOTALL
    )
    for num, val in matches:
        results[num] = val.strip().rstrip('.')
    return results


def extract_list_items(response: str) -> List[str]:
    """Extract items from a comma-separated or bullet list in the answer block."""
    answer_block = extract_answer_block(response)

    # Try bullet points first: "- item"
    bullets = re.findall(r"[-•]\s*(.+?)(?:\n|$)", answer_block)
    if bullets:
        return [b.strip().rstrip('.') for b in bullets]

    # Try to extract numeric values with units (handles commas in numbers like 3,201.33)
    # Match patterns: $1,234.56, 1,234.56, 12.3°C, 45.6 mg/L
    numeric_items = re.findall(
        r'[\$]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:°[CF]|mg/L|%|\$))?',
        answer_block
    )
    if numeric_items and len(numeric_items) >= 2:
        return [x.strip() for x in numeric_items]

    # Try comma-separated (fallback for non-numeric items)
    if "," in answer_block:
        items = [x.strip().rstrip('.') for x in answer_block.split(",")]
        return [x for x in items if x]

    # Try newline-separated
    lines = [l.strip().rstrip('.') for l in answer_block.split("\n") if l.strip()]
    if lines:
        return lines

    return [answer_block] if answer_block else []


def extract_person_item_pairs(response: str) -> Dict[str, str]:
    """Extract 'Person: item' or 'Person holds item' pairs."""
    answer_block = extract_answer_block(response)
    results = {}
    # Match "- Name: item" or "Name: item" or "Name holds item"
    patterns = [
        r"[-•]?\s*(\w+)\s*:\s*(.+?)(?:\n|$)",
        r"[-•]?\s*(\w+)\s+holds?\s+(?:a\s+)?(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        matches = re.findall(pat, answer_block, re.IGNORECASE)
        if matches:
            for name, item in matches:
                results[name.strip()] = item.strip().rstrip('.')
            break
    return results


def fuzzy_value_match(predicted: str, gold: str, tolerance: float = 0.0) -> bool:
    """Check if predicted value matches gold, with optional numeric tolerance."""
    # Exact match (case-insensitive, whitespace-normalized)
    pred_clean = re.sub(r'\s+', ' ', predicted.strip().lower())
    gold_clean = re.sub(r'\s+', ' ', gold.strip().lower())

    if pred_clean == gold_clean:
        return True

    # Check if gold is contained in predicted
    if gold_clean in pred_clean:
        return True

    # Numeric comparison with tolerance
    try:
        pred_num = float(re.sub(r'[,$%°]', '', predicted))
        gold_num = float(re.sub(r'[,$%°]', '', gold))
        if tolerance > 0:
            return abs(pred_num - gold_num) <= tolerance
        return pred_num == gold_num
    except (ValueError, TypeError):
        pass

    return False


def compute_recall(found: List[str], gold: List[str]) -> float:
    """Compute recall: what fraction of gold items were found."""
    if not gold:
        return 1.0
    matched = sum(1 for g in gold if any(fuzzy_value_match(f, g) for f in found))
    return matched / len(gold)


def compute_precision(found: List[str], gold: List[str]) -> float:
    """Compute precision: what fraction of found items are correct."""
    if not found:
        return 0.0
    matched = sum(1 for f in found if any(fuzzy_value_match(f, g) for g in gold))
    return matched / len(found)


def compute_intrusion_rate(found: List[str], distractors: List[str]) -> float:
    """Compute intrusion rate: what fraction of distractors leaked through."""
    if not distractors:
        return 0.0
    intruded = sum(1 for d in distractors if any(fuzzy_value_match(f, d) for f in found))
    return intruded / len(distractors)


class ScoreResult:
    """Container for scoring results with metadata."""

    def __init__(self, task_id: str, task_type: str, difficulty: str):
        self.task_id = task_id
        self.task_type = task_type
        self.difficulty = difficulty
        self.metrics: Dict[str, float] = {}
        self.details: Dict[str, Any] = {}
        self.raw_response: str = ""

    def add_metric(self, name: str, value: float):
        self.metrics[name] = round(value, 4)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            **self.metrics,
            "details": self.details,
        }
