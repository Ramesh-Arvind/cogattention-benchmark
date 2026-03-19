"""
Format benchmark items for human survey platforms (Prolific, MTurk, Qualtrics).

Exports tasks as CSV or JSON suitable for import into survey tools.
"""

import csv
import json
import os
from typing import List, Optional
from ..generators.base import TaskInstance


def format_for_qualtrics(
    tasks: List[TaskInstance],
    output_path: str,
    format: str = "csv",
) -> str:
    """Export tasks as CSV/JSON for Qualtrics or similar survey platforms.

    Each row contains:
    - task_id, task_type, difficulty
    - prompt (the full instruction shown to participants)
    - A blank answer field for participants to fill in

    Args:
        tasks: List of TaskInstance objects.
        output_path: Where to save the file.
        format: "csv" or "json".

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    records = []
    for i, task in enumerate(tasks, 1):
        record = {
            "item_number": i,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "prompt": task.prompt,
            "instructions": _get_response_instructions(task.task_type),
            "participant_answer": "",  # blank for survey
        }
        records.append(record)

    if format == "csv":
        if not output_path.endswith(".csv"):
            output_path += ".csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    elif format == "json":
        if not output_path.endswith(".json"):
            output_path += ".json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return output_path


def _get_response_instructions(task_type: str) -> str:
    """Get human-readable response instructions per task type."""
    instructions = {
        "capacity": "List each person and the item they are currently holding.",
        "interference": "For each record, write ONLY the most recent (final) value.",
        "sustained": "List ALL target items you found in the passage, separated by commas.",
        "stream_segregation": "Answer the question about Stream A only. Ignore Stream B.",
        "selective": "List ONLY the values from the verified/confirmed source.",
        "stroop": "Answer what the sentence LITERALLY says, even if it contains errors.",
        "flanker": "Answer the question about the specified sentence ONLY.",
        "shifting": "Classify each word according to the CURRENT rule (the rule may change mid-task).",
        "inhibition_return": "Answer each question based on the relevant passage.",
        "anomaly": "1) Complete the primary counting task. 2) Report any unusual/anomalous content.",
    }
    return instructions.get(task_type, "Follow the instructions in the prompt and write your answer.")


def format_scoring_key(
    tasks: List[TaskInstance],
    output_path: str,
) -> str:
    """Export the gold answers as a scoring key (for grading human responses).

    Args:
        tasks: List of TaskInstance objects.
        output_path: Where to save the JSON scoring key.

    Returns:
        Path to the saved file.
    """
    key = []
    for task in tasks:
        key.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "gold_answer": task.gold_answer,
            "metadata": {
                k: v for k, v in task.metadata.items()
                if k not in ("prompt",)  # exclude redundant fields
            },
        })

    if not output_path.endswith(".json"):
        output_path += ".json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, default=str)

    return output_path
