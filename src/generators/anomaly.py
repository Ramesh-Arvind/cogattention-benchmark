"""
Task E: Anomaly Detection (Stimulus-Driven Attention)

Tests whether the model notices unexpected/anomalous items during a routine task.
Cognitive construct: §7.3.3 Stimulus-Driven Attention
Paradigm source: Inattentional Blindness (Simons & Chabris) + Cocktail Party

The model performs a primary counting/extraction task while an anomaly is embedded.
It must complete the primary task AND report the anomaly.
"""

import random
from typing import List, Dict
from .base import (
    TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES,
    generate_filler_paragraph,
)

ITEMS_PER_DIFFICULTY = 8

# Primary task types
PRIMARY_TASKS = {
    "count_word": {
        "instruction": 'Count how many times the word "{target_word}" appears in the passage below.',
        "question": 'How many times does "{target_word}" appear?',
    },
    "count_names": {
        "instruction": "Count how many different person names are mentioned in the passage below.",
        "question": "How many different person names appear?",
    },
    "sum_numbers": {
        "instruction": "Find all the dollar amounts mentioned in the passage and compute their sum.",
        "question": "What is the total sum of all dollar amounts?",
    },
}

# Anomaly types with detection keywords
ANOMALY_TYPES = {
    "language_switch": {
        "saliency": "high",
        "generator": lambda rng: rng.choice([
            "Le soleil se couchait derrière les montagnes, peignant le ciel en nuances d'or et de pourpre.",
            "El mercado estaba lleno de colores y sonidos que despertaban todos los sentidos.",
            "Die alte Brücke überquerte den Fluss seit mehr als dreihundert Jahren.",
            "Il vecchio pescatore sedeva sul molo, osservando le onde con pazienza infinita.",
        ]),
        "detection_keywords": ["french", "spanish", "german", "italian", "foreign", "different language",
                                "non-english", "another language"],
    },
    "code_block": {
        "saliency": "high",
        "generator": lambda rng: rng.choice([
            'def calculate_total(items): return sum(item.price for item in items if item.is_valid)',
            'SELECT * FROM transactions WHERE amount > 1000 ORDER BY date DESC LIMIT 10;',
            'for (int i = 0; i < n; i++) { if (arr[i] > max) { max = arr[i]; } }',
            'const result = await fetch("/api/data").then(res => res.json()).catch(err => console.error(err));',
        ]),
        "detection_keywords": ["code", "programming", "script", "function", "SQL", "software",
                                "computer code", "snippet"],
    },
    "factual_absurdity": {
        "saliency": "medium",
        "generator": lambda rng: rng.choice([
            "The sun set in the east that evening, casting long shadows westward across the plaza.",
            "Water, as everyone knows, flows uphill when left undisturbed in open channels.",
            "The temperature that January reached forty-five degrees Celsius in Helsinki, a typical winter day.",
            "The Atlantic Ocean, the smallest of Earth's oceans, separates Europe from the Americas.",
        ]),
        "detection_keywords": ["incorrect", "wrong", "false", "absurd", "impossible", "factual error",
                                "inaccurate", "not true", "east", "uphill", "smallest", "forty-five"],
    },
    "numerical_anomaly": {
        "saliency": "medium",
        "generator": lambda rng: (
            f"The final invoice came to ${rng.choice([999999, 8500000, 12000000]):,}.00, "
            f"which {rng.choice(FIRST_NAMES)} paid without hesitation."
        ),
        "detection_keywords": ["large", "unusual", "amount", "million", "expensive", "anomalous",
                                "extraordinary", "999", "8,500", "12,000"],
    },
    "name_inconsistency": {
        "saliency": "low",
        "generator": lambda rng: None,  # handled specially — needs character tracking
        "detection_keywords": ["name", "changed", "inconsistent", "different name", "was called",
                                "previously", "earlier"],
    },
    "single_char_swap": {
        "saliency": "ultra_low",
        "generator": lambda rng: rng.choice([
            "The inspection revealed that the buidling met all safety standards without exception.",
            "The quarterly report confirmed that reveneu exceeded projections by a comfortable margin.",
            "All shipments were recieved in good condition according to the warehouse manifest.",
            "The committe approved the proposal after reviewing the supplementary documentation.",
        ]),
        "detection_keywords": ["typo", "misspelling", "spelling", "buidling", "reveneu", "recieved",
                                "committe", "error", "mistake"],
    },
    "off_by_one": {
        "saliency": "ultra_low",
        "generator": lambda rng: rng.choice([
            "The bridge, completed in 1889, celebrated its 150th anniversary in 2038.",
            "The team of 12 members split into 3 groups of 5 to cover all the zones.",
            "The 48-hour deadline began on Monday and expired on Wednesday at the same hour.",
            "The recipe calls for 8 eggs, but only 7 are listed in the ingredient summary.",
        ]),
        "detection_keywords": ["math", "calculation", "wrong", "incorrect", "off by one", "doesn't add up",
                                "inconsistent", "150", "anniversary", "groups of 5", "12", "48-hour",
                                "Wednesday", "8 eggs", "7"],
    },
    "date_inconsistency": {
        "saliency": "ultra_low",
        "generator": lambda rng: rng.choice([
            "The contract, signed on Tuesday, March 15th, took effect the following Monday, March 19th.",
            "The event scheduled for Saturday, June 7th was postponed to the next Friday, June 12th.",
            "The meeting on Wednesday, April 22nd was rescheduled to Thursday, April 22nd.",
            "The shipment departed on February 29th, 2023 and arrived three days later.",
        ]),
        "detection_keywords": ["date", "calendar", "day", "wrong", "inconsistent", "February 29",
                                "doesn't match", "Tuesday", "March 15", "Saturday", "June 7"],
    },
}

DIFFICULTY_CONFIG = {
    "Easy":   {"primary_task": "count_word",  "anomaly_saliency": "high",   "passage_length": 8},
    "Medium": {"primary_task": "count_word",  "anomaly_saliency": "high",   "passage_length": 15},
    "Hard":   {"primary_task": "count_names", "anomaly_saliency": "medium", "passage_length": 20},
    "Expert":   {"primary_task": "sum_numbers", "anomaly_saliency": "low",       "passage_length": 25},
    "Frontier": {"primary_task": "sum_numbers", "anomaly_saliency": "ultra_low", "passage_length": 35},
}


def _generate_passage_with_anomaly(
    rng: random.Random,
    config: dict,
) -> tuple:
    """Generate passage, primary task answer, and anomaly info."""

    n_paras = config["passage_length"]
    saliency = config["anomaly_saliency"]

    # Select anomaly type matching saliency
    matching_types = [k for k, v in ANOMALY_TYPES.items()
                      if v["saliency"] == saliency and k != "name_inconsistency"]
    if saliency == "low":
        matching_types = ["name_inconsistency"]
    elif saliency == "ultra_low":
        matching_types = ["single_char_swap", "off_by_one", "date_inconsistency"]

    anomaly_type = rng.choice(matching_types) if matching_types else "factual_absurdity"

    # Generate filler paragraphs
    paragraphs = [generate_filler_paragraph(rng) for _ in range(n_paras)]

    # Handle primary task specifics
    task_type = config["primary_task"]
    primary_answer = None
    target_word = None

    if task_type == "count_word":
        target_word = rng.choice(["the", "and", "was", "that"])
        # Count occurrences across all paragraphs
        base_count = sum(p.lower().split().count(target_word) for p in paragraphs)
        primary_answer = base_count
    elif task_type == "count_names":
        # Count unique FIRST_NAMES that appear in paragraphs
        all_text = " ".join(paragraphs)
        found = set()
        for name in FIRST_NAMES:
            if name in all_text:
                found.add(name)
        primary_answer = len(found)
    elif task_type == "sum_numbers":
        # Insert known dollar amounts into some paragraphs
        amounts = []
        insert_indices = rng.sample(range(n_paras), min(4, n_paras))
        for idx in insert_indices:
            amt = round(rng.uniform(5, 200), 2)
            amounts.append(amt)
            paragraphs[idx] += f" The cost came to ${amt:.2f}."
        primary_answer = round(sum(amounts), 2)

    # Generate anomaly sentence
    anomaly_info = ANOMALY_TYPES[anomaly_type]
    if anomaly_type == "name_inconsistency":
        # Pick a name that appears in the text, introduce inconsistency
        char_name = rng.choice(FIRST_NAMES)
        alt_name = rng.choice([n for n in FIRST_NAMES if n != char_name])
        anomaly_sentence = (
            f"{char_name} checked the inventory one final time. "
            f"Later that afternoon, {alt_name} — the same person — "
            f"signed off on the delivery manifest."
        )
    else:
        anomaly_sentence = anomaly_info["generator"](rng)

    # Insert anomaly at a random position (not first or last)
    anomaly_pos = rng.randint(2, max(2, n_paras - 2))
    paragraphs.insert(anomaly_pos, anomaly_sentence)

    # Recount primary task after anomaly insertion
    if task_type == "count_word":
        primary_answer = sum(p.lower().split().count(target_word) for p in paragraphs)
    elif task_type == "count_names":
        all_text = " ".join(paragraphs)
        found = set()
        for name in FIRST_NAMES:
            if name in all_text:
                found.add(name)
        primary_answer = len(found)

    passage = "\n\n".join(paragraphs)

    return passage, primary_answer, target_word, anomaly_type, anomaly_sentence, anomaly_pos


def generate_anomaly_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1021 + DIFFICULTY_LEVELS.index(difficulty) * 5381)
    config = DIFFICULTY_CONFIG[difficulty]

    passage, primary_answer, target_word, anomaly_type, anomaly_sentence, anomaly_pos = \
        _generate_passage_with_anomaly(rng, config)

    task_info = PRIMARY_TASKS[config["primary_task"]]
    instruction = task_info["instruction"]
    question = task_info["question"]
    if target_word:
        instruction = instruction.format(target_word=target_word)
        question = question.format(target_word=target_word)

    prompt = (
        f"PRIMARY TASK: {instruction}\n\n"
        f"---\n{passage}\n---\n\n"
        f"Answer BOTH questions:\n"
        f"1. {question}\n"
        f"2. Did you notice anything unusual or out of place in the passage? "
        f"If yes, describe it briefly. If no, write 'nothing unusual'.\n\n"
        f"Format:\n"
        f"ANSWER:\n"
        f"1. [your answer]\n"
        f"2. [description or 'nothing unusual']"
    )

    return TaskInstance(
        task_id=f"anomaly_{difficulty.lower()}_{instance_idx:03d}",
        task_type="anomaly",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer={
            "primary": primary_answer,
            "anomaly_type": anomaly_type,
            "anomaly_sentence": anomaly_sentence,
        },
        metadata={
            "primary_task": config["primary_task"],
            "target_word": target_word,
            "primary_answer": primary_answer,
            "anomaly_type": anomaly_type,
            "anomaly_saliency": config["anomaly_saliency"],
            "anomaly_position": anomaly_pos,
            "anomaly_sentence": anomaly_sentence,
            "detection_keywords": ANOMALY_TYPES[anomaly_type]["detection_keywords"],
            "passage_paragraphs": config["passage_length"] + 1,
        },
    )


def generate_anomaly_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_anomaly_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_anomaly_dataset()
    print(f"Generated {len(dataset)} Anomaly Detection instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: task={s.metadata['primary_task']}, "
              f"anomaly={s.metadata['anomaly_type']} "
              f"(saliency={s.metadata['anomaly_saliency']})")
        print(f"    Primary answer: {s.metadata['primary_answer']}")
        print(f"    Anomaly: {s.metadata['anomaly_sentence'][:80]}...")
