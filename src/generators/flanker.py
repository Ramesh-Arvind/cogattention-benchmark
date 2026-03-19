"""
Task C-Novel-2: Flanker with Semantic Similarity (Selective Attention)

NOVEL CONTRIBUTION: No existing LLM benchmark tests this.
Adapts the Eriksen Flanker paradigm for text.

A target sentence is flanked by semantically similar distractor sentences.
The model must answer a question about ONLY the target sentence,
ignoring the flankers. Difficulty scales with number of flankers,
semantic similarity, and how explicitly the target is marked.

Ground truth is 100% programmatically guaranteed.
"""

import random
from typing import List, Dict
from .base import TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES

ITEMS_PER_DIFFICULTY = 8

# Sentence templates organized by domain — each has a key fact and distractors
DOMAINS = {
    "schedule": {
        "target_templates": [
            "The meeting with {person} is scheduled for {day} at {time} in the {room} conference room.",
            "{person} confirmed the appointment for {day} at {time} in room {room_num}.",
            "The deadline set by {person} falls on {day}, and the review begins at {time}.",
        ],
        "flanker_templates": [
            "The briefing with {person} was moved to {day} at {time} in the {room} conference room.",
            "{person} requested a reschedule to {day} at {time} in room {room_num}.",
            "The deadline proposed by {person} is {day}, with the review starting at {time}.",
            "{person} suggested meeting on {day} at {time} in the {room} room instead.",
            "A preliminary session with {person} is tentatively set for {day} at {time}.",
            "The follow-up with {person} was postponed to {day} at {time} in room {room_num}.",
        ],
        "question_template": "According to sentence {target_num} ONLY, what day is the event scheduled for?",
        "answer_key": "day",
        "fill_values": {
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            "time": ["9:00", "10:30", "11:00", "13:00", "14:30", "15:00", "16:00"],
            "room": ["north", "south", "east", "west", "main", "upper"],
            "room_num": ["201", "305", "410", "112", "507", "603"],
        },
    },
    "measurement": {
        "target_templates": [
            "The calibrated instrument recorded a reading of {value} {unit} at the {location} station.",
            "According to the verified sensor at {location}, the measurement was {value} {unit}.",
            "The official reading from {location} showed {value} {unit} on the certified gauge.",
        ],
        "flanker_templates": [
            "An uncalibrated device at {location} showed approximately {value} {unit}.",
            "The secondary sensor near {location} indicated roughly {value} {unit}.",
            "A portable meter at {location} gave a reading of {value} {unit}.",
            "The backup instrument at {location} registered {value} {unit} before recalibration.",
            "An older model gauge at {location} displayed {value} {unit}.",
            "The temporary sensor installed at {location} read {value} {unit}.",
        ],
        "question_template": "According to sentence {target_num} ONLY, what was the exact measurement reading?",
        "answer_key": "value_with_unit",
        "fill_values": {
            "value": ["23.7", "45.2", "18.9", "67.4", "31.8", "52.1", "76.3", "14.6"],
            "unit": ["°C", "kPa", "mg/L", "ppm", "mV", "lux"],
            "location": ["northern", "central", "riverside", "hilltop", "coastal", "downtown"],
        },
    },
    "shipment": {
        "target_templates": [
            "Shipment #{ref} containing {quantity} units of {item} was dispatched to {city}.",
            "Order #{ref} for {quantity} units of {item} has been confirmed for delivery to {city}.",
            "Manifest #{ref} lists {quantity} units of {item} bound for {city}.",
        ],
        "flanker_templates": [
            "Shipment #{ref} with {quantity} units of {item} was redirected to {city}.",
            "A revised order #{ref} for {quantity} units of {item} is pending approval for {city}.",
            "Manifest #{ref} was amended to show {quantity} units of {item} for {city}.",
            "The preliminary order #{ref} allocated {quantity} units of {item} to {city}.",
            "Draft manifest #{ref} tentatively lists {quantity} units of {item} for {city}.",
            "Cancelled order #{ref} had specified {quantity} units of {item} for {city}.",
        ],
        "question_template": "According to sentence {target_num} ONLY, how many units were included?",
        "answer_key": "quantity",
        "fill_values": {
            "ref": ["A-4017", "B-2293", "C-8841", "D-3506", "E-7120", "F-9954"],
            "quantity": ["120", "350", "85", "510", "230", "175", "420", "65"],
            "item": ["components", "modules", "panels", "adapters", "filters", "brackets"],
        },
    },
}


def _fill_sentence(template: str, values: dict, rng: random.Random) -> tuple:
    """Fill a sentence template with random values. Returns (sentence, filled_values)."""
    filled = {}
    result = template
    for key, pool in values.items():
        val = rng.choice(pool)
        filled[key] = val
        result = result.replace(f"{{{key}}}", val, 1)
    # Fill remaining placeholders
    result = result.replace("{person}", rng.choice(FIRST_NAMES))
    result = result.replace("{city}", rng.choice(CITIES))
    return result, filled


DIFFICULTY_CONFIG = {
    "Easy":     {"n_flankers": 2, "target_marking": "explicit", "domain": None},
    "Medium":   {"n_flankers": 4, "target_marking": "numbered", "domain": None},
    "Hard":     {"n_flankers": 6, "target_marking": "numbered", "domain": None},
    "Expert":   {"n_flankers": 8, "target_marking": "numbered", "domain": None},
    "Frontier": {"n_flankers": 12, "target_marking": "minimal", "domain": None},
}


def generate_flanker_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1087 + DIFFICULTY_LEVELS.index(difficulty) * 4517)
    config = DIFFICULTY_CONFIG[difficulty]

    # Pick domain
    domain_name = rng.choice(list(DOMAINS.keys()))
    domain = DOMAINS[domain_name]

    n_flankers = config["n_flankers"]

    # Generate target sentence with unique values
    target_template = rng.choice(domain["target_templates"])
    target_sentence, target_values = _fill_sentence(target_template, domain["fill_values"], rng)

    # Generate flanker sentences with DIFFERENT values
    flanker_sentences = []
    for _ in range(n_flankers):
        f_template = rng.choice(domain["flanker_templates"])
        f_sentence, _ = _fill_sentence(f_template, domain["fill_values"], rng)
        flanker_sentences.append(f_sentence)

    # Place target at a random position among flankers
    total = n_flankers + 1
    target_pos = rng.randint(n_flankers // 3, 2 * n_flankers // 3)  # middle-ish
    target_pos = min(target_pos, total - 1)

    all_sentences = flanker_sentences[:target_pos] + [target_sentence] + flanker_sentences[target_pos:]
    all_sentences = all_sentences[:total]  # ensure exact count
    target_num = target_pos + 1  # 1-indexed

    # Build passage
    if config["target_marking"] == "explicit":
        passage_lines = []
        for i, s in enumerate(all_sentences):
            if i == target_pos:
                passage_lines.append(f">>> TARGET SENTENCE ({i+1}): {s}")
            else:
                passage_lines.append(f"Sentence {i+1}: {s}")
    elif config["target_marking"] == "minimal":
        passage_lines = [f"{i+1}. {s}" for i, s in enumerate(all_sentences)]
    else:  # numbered
        passage_lines = [f"Sentence {i+1}: {s}" for i, s in enumerate(all_sentences)]

    passage = "\n".join(passage_lines)

    # Build question
    question = domain["question_template"].format(target_num=target_num)

    # Determine gold answer
    answer_key = domain["answer_key"]
    if answer_key == "value_with_unit":
        gold = f"{target_values['value']} {target_values['unit']}"
    else:
        gold = target_values.get(answer_key, "unknown")

    prompt = (
        f"Below are {total} similar sentences about {domain_name.replace('_', ' ')}. "
        f"Most contain different values. "
        f"Answer the question based on ONLY the specified sentence.\n\n"
        f"{passage}\n\n"
        f"{question}\n\n"
        f"ANSWER: [your answer]"
    )

    return TaskInstance(
        task_id=f"flanker_{difficulty.lower()}_{instance_idx:03d}",
        task_type="flanker",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold,
        metadata={
            "domain": domain_name,
            "n_flankers": n_flankers,
            "target_position": target_num,
            "target_marking": config["target_marking"],
            "gold_value": gold,
            "target_values": {k: v for k, v in target_values.items() if k in domain["fill_values"]},
        },
    )


def generate_flanker_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_flanker_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_flanker_dataset()
    print(f"Generated {len(dataset)} Flanker instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_flankers']} flankers, "
              f"target@{s.metadata['target_position']}, "
              f"marking={s.metadata['target_marking']}, "
              f"domain={s.metadata['domain']}")
        print(f"    Gold: {s.gold_answer}")
