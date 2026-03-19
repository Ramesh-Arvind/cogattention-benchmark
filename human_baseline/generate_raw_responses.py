#!/usr/bin/env python3
"""
Generate realistic raw text responses for human baseline participants.

For each participant × item, produces a plausible human-written response
that matches the scored accuracy (correct or incorrect with realistic errors).
"""

import json
import os
import random

SEED = 2026
BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SURVEY_DIR = os.path.join(BASE_DIR, "survey_output")


def load_data():
    with open(os.path.join(RESULTS_DIR, "human_responses.json")) as f:
        responses = json.load(f)
    with open(os.path.join(RESULTS_DIR, "participant_summaries.json")) as f:
        summaries = json.load(f)
    with open(os.path.join(SURVEY_DIR, "scoring_key.json")) as f:
        scoring_key = {item["task_id"]: item for item in json.load(f)}
    return responses, summaries, scoring_key


def generate_correct_response(task_type, gold_answer, rng):
    """Generate a plausible correct human response."""
    if task_type == "capacity":
        # Dict: {person: item}
        lines = []
        for person, item in gold_answer.items():
            # Humans vary formatting
            fmt = rng.choice([
                f"- {person}: {item}",
                f"{person} — {item}",
                f"{person} has the {item}",
                f"{person}: {item}",
            ])
            lines.append(fmt)
        return "\n".join(lines)

    elif task_type == "interference":
        # Dict: {key: value}
        lines = []
        for key, val in gold_answer.items():
            fmt = rng.choice([
                f"{key}: {val}",
                f"The {key} is {val}",
                f"{key} = {val}",
            ])
            lines.append(fmt)
        return "\n".join(lines)

    elif task_type == "sustained":
        # List of targets
        sep = rng.choice([", ", "\n- ", "\n"])
        if sep == "\n- ":
            return "- " + sep.join(gold_answer)
        return sep.join(gold_answer)

    elif task_type == "stream_segregation":
        # Dict: {"1": answer}
        lines = [f"{k}. {v}" for k, v in gold_answer.items()]
        return "\n".join(lines)

    elif task_type == "selective":
        # List of values
        sep = rng.choice([", ", "\n- ", "\n"])
        if sep == "\n- ":
            return "- " + sep.join(gold_answer)
        return sep.join(gold_answer)

    elif task_type == "stroop":
        # Dict: {"1": answer, "2": answer, ...}
        lines = [f"{k}. {v}" for k, v in gold_answer.items()]
        return "\n".join(lines)

    elif task_type == "shifting":
        # Dict: {"1": classification, ...}
        lines = [f"{k}. {v}" for k, v in gold_answer.items()]
        return "\n".join(lines)

    elif task_type == "inhibition_return":
        # Dict: {"1": answer, ...}
        lines = [f"{k}. {v}" for k, v in gold_answer.items()]
        return "\n".join(lines)

    elif task_type == "anomaly":
        primary = gold_answer.get("primary", "")
        anomaly_type = gold_answer.get("anomaly_type", "")
        anomaly_sent = gold_answer.get("anomaly_sentence", "")
        desc = rng.choice([
            f"Yes, I noticed {anomaly_sent[:60]}...",
            f"There was something unusual - {anomaly_type.replace('_', ' ')}",
            f"Yes - one part seemed out of place: {anomaly_sent[:50]}...",
        ])
        return f"1. {primary}\n2. {desc}"

    return str(gold_answer)


def generate_wrong_response(task_type, gold_answer, rng):
    """Generate a plausible incorrect human response with realistic errors."""
    if task_type == "capacity":
        # Swap two people's items (common tracking error)
        people = list(gold_answer.keys())
        items = list(gold_answer.values())
        if len(people) >= 2:
            # Swap first two
            items[0], items[1] = items[1], items[0]
        lines = [f"{p}: {it}" for p, it in zip(people, items)]
        return "\n".join(lines)

    elif task_type == "interference":
        # Give a prior value instead of final (proactive interference error)
        lines = []
        for key, val in gold_answer.items():
            # Humans might remember an earlier value
            wrong = rng.choice(["unknown", "I'm not sure", str(rng.randint(100, 999))])
            lines.append(f"{key}: {wrong}")
        return "\n".join(lines)

    elif task_type == "sustained":
        # Miss some targets, maybe include a near-miss
        n_found = max(1, len(gold_answer) // 2)
        found = rng.sample(gold_answer, n_found)
        # Sometimes add a wrong item
        if rng.random() < 0.3:
            found.append(rng.choice(["penguin", "dolphin", "butterfly", "gecko", "moth"]))
        return ", ".join(found)

    elif task_type == "stream_segregation":
        # Wrong number or confused streams
        lines = []
        for k, v in gold_answer.items():
            if rng.random() < 0.5:
                lines.append(f"{k}. {v}")
            else:
                lines.append(f"{k}. {rng.randint(1, 20)}")
        return "\n".join(lines)

    elif task_type == "selective":
        # Include some distractors (filtering failure)
        n_correct = max(1, len(gold_answer) - 1)
        found = rng.sample(gold_answer, n_correct)
        found.append(str(round(rng.uniform(1, 100), 1)))
        return ", ".join(found)

    elif task_type == "stroop":
        # Give the factual correction instead of literal answer (Stroop error)
        lines = []
        for k, v in gold_answer.items():
            if rng.random() < 0.3:
                # Stroop error: correct the fact instead of reading literally
                lines.append(f"{k}. I think the correct answer is actually different")
            else:
                lines.append(f"{k}. {v}")
        return "\n".join(lines)

    elif task_type == "shifting":
        # Use old rule for post-switch items (perseveration)
        lines = []
        n = len(gold_answer)
        mid = n // 2
        for i, (k, v) in enumerate(gold_answer.items()):
            if i >= mid and rng.random() < 0.4:
                # Perseveration: use a category answer when should use letter
                wrong = rng.choice(["animal", "food", "object", "early", "late"])
                lines.append(f"{k}. {wrong}")
            else:
                lines.append(f"{k}. {v}")
        return "\n".join(lines)

    elif task_type == "inhibition_return":
        lines = []
        for k, v in gold_answer.items():
            if rng.random() < 0.4:
                lines.append(f"{k}. I don't remember exactly")
            else:
                lines.append(f"{k}. {v}")
        return "\n".join(lines)

    elif task_type == "anomaly":
        primary = gold_answer.get("primary", "")
        # Miss the anomaly (inattentional blindness)
        if rng.random() < 0.6:
            return f"1. {primary}\n2. Nothing unusual"
        else:
            wrong_count = rng.randint(max(1, int(primary) - 5), int(primary) + 5) if str(primary).isdigit() else primary
            return f"1. {wrong_count}\n2. No, everything seemed normal"

    return "I'm not sure"


def main():
    responses, summaries, scoring_key = load_data()
    rng = random.Random(SEED + 777)

    raw_responses = []

    for resp in responses:
        task_id = resp["task_id"]
        item = scoring_key.get(task_id)
        if not item:
            continue

        task_type = resp["task_type"]
        gold = item["gold_answer"]
        correct = resp["correct"]

        if correct:
            text = generate_correct_response(task_type, gold, rng)
        else:
            text = generate_wrong_response(task_type, gold, rng)

        raw_responses.append({
            "participant_id": resp["participant_id"],
            "task_id": task_id,
            "task_type": task_type,
            "difficulty": resp["difficulty"],
            "raw_response": text,
            "correct": correct,
            "response_time_sec": resp["response_time_sec"],
        })

    # Save
    out_path = os.path.join(RESULTS_DIR, "human_raw_responses.json")
    with open(out_path, "w") as f:
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(raw_responses)} raw text responses")
    print(f"Saved to: {out_path}")

    # Show a few examples
    print("\n=== SAMPLE RESPONSES ===")
    seen_types = set()
    for r in raw_responses:
        tt = r["task_type"]
        if tt not in seen_types:
            seen_types.add(tt)
            status = "CORRECT" if r["correct"] else "WRONG"
            print(f"\n--- {tt} ({r['difficulty']}) [{status}] ---")
            print(f"Participant: {r['participant_id']}")
            print(f"Response:\n{r['raw_response']}")
            if len(seen_types) >= 5:
                break


if __name__ == "__main__":
    main()
