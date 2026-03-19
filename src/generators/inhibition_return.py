"""
Task D-Novel: Inhibition of Return (Attention Shifting)

NOVEL CONTRIBUTION: No existing LLM benchmark tests this.
Adapts Inhibition of Return (IOR) from visual attention research.

Three short passages (A, B, C) are presented. Three questions follow:
  Q1: about passage X (initial attend)
  Q2: about passage Y (shift away)
  Q3: about passage X again (return)

Measures "return penalty": does accuracy on Q3 (return) degrade
compared to Q1 (initial attention)?

Ground truth is 100% programmatically guaranteed.
"""

import random
from typing import List, Dict
from .base import (
    TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES,
    generate_filler_paragraph,
)

ITEMS_PER_DIFFICULTY = 8

# Passage topics with embedded facts for question generation
PASSAGE_TOPICS = [
    {
        "topic": "warehouse_inventory",
        "template": (
            "The warehouse at {location} received a shipment on {day}. "
            "It contained {quantity} crates of {item}, each weighing {weight} kilograms. "
            "The shipment was logged by {person} under reference number {ref}. "
            "{filler} "
            "The total value of the shipment was ${value}."
        ),
        "questions": [
            ("What day did the shipment arrive?", "day"),
            ("How many crates were in the shipment?", "quantity"),
            ("Who logged the shipment?", "person"),
            ("What was the reference number?", "ref"),
            ("What was the total value?", "value"),
            ("How much did each crate weigh?", "weight"),
        ],
        "fill_values": {
            "location": ["the north dock", "the east wing", "building 7", "the central depot"],
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "quantity": ["24", "36", "48", "15", "60", "72"],
            "item": ["electronics", "textiles", "machinery parts", "chemicals", "ceramics"],
            "weight": ["12", "18", "25", "8", "32", "15"],
            "ref": ["WH-4017", "WH-2293", "WH-8841", "WH-3506", "WH-7120"],
            "value": ["4,200", "7,850", "12,300", "3,600", "9,100", "15,400"],
        },
    },
    {
        "topic": "weather_report",
        "template": (
            "The weather station at {location} recorded conditions on {day}. "
            "Temperature reached {temp}°C with humidity at {humidity}%. "
            "Wind speed was measured at {wind} km/h from the {direction}. "
            "{filler} "
            "Barometric pressure stood at {pressure} hPa."
        ),
        "questions": [
            ("What was the temperature?", "temp"),
            ("What was the humidity percentage?", "humidity"),
            ("What was the wind speed?", "wind"),
            ("What direction was the wind from?", "direction"),
            ("What was the barometric pressure?", "pressure"),
            ("What day were conditions recorded?", "day"),
        ],
        "fill_values": {
            "location": ["the hilltop", "the coastal station", "the airport", "the valley floor"],
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "temp": ["18.5", "23.2", "31.7", "12.4", "27.8", "8.9"],
            "humidity": ["45", "62", "78", "33", "55", "87"],
            "wind": ["15", "28", "42", "8", "35", "52"],
            "direction": ["north", "south", "east", "west", "northeast", "southwest"],
            "pressure": ["1013", "1008", "1021", "997", "1015", "1003"],
        },
    },
    {
        "topic": "lab_experiment",
        "template": (
            "Experiment {exp_id} was conducted by {person} in lab {lab_num}. "
            "The sample was heated to {temp}°C for {duration} minutes. "
            "The resulting concentration was {concentration} mg/L. "
            "{filler} "
            "The pH of the solution measured {ph} at completion."
        ),
        "questions": [
            ("What was the experiment ID?", "exp_id"),
            ("Who conducted the experiment?", "person"),
            ("What temperature was the sample heated to?", "temp"),
            ("How many minutes was the heating duration?", "duration"),
            ("What was the resulting concentration?", "concentration"),
            ("What was the final pH?", "ph"),
        ],
        "fill_values": {
            "exp_id": ["EX-401", "EX-229", "EX-884", "EX-350", "EX-712"],
            "lab_num": ["B-12", "C-05", "A-18", "D-03", "B-22"],
            "temp": ["65", "80", "120", "45", "95", "150"],
            "duration": ["15", "30", "45", "60", "90", "120"],
            "concentration": ["12.4", "28.7", "45.1", "8.9", "33.6", "56.2"],
            "ph": ["6.2", "7.1", "5.8", "8.4", "6.9", "4.5"],
        },
    },
]


def _build_passage(topic_data: dict, rng: random.Random) -> tuple:
    """Build a passage and return (passage_text, fact_dict)."""
    template = topic_data["template"]
    fill = topic_data["fill_values"]

    facts = {}
    result = template
    for key, pool in fill.items():
        val = rng.choice(pool)
        facts[key] = val
        result = result.replace(f"{{{key}}}", val, 1)

    # Fill person and filler
    person = rng.choice(FIRST_NAMES)
    facts["person"] = person
    result = result.replace("{person}", person, 1)
    filler = generate_filler_paragraph(rng)
    result = result.replace("{filler}", filler, 1)

    return result, facts


DIFFICULTY_CONFIG = {
    "Easy":     {"n_passages": 3, "n_filler_between": 0, "return_delay": 1},
    "Medium":   {"n_passages": 3, "n_filler_between": 1, "return_delay": 1},
    "Hard":     {"n_passages": 4, "n_filler_between": 2, "return_delay": 2},
    "Expert":   {"n_passages": 4, "n_filler_between": 3, "return_delay": 2},
    "Frontier": {"n_passages": 5, "n_filler_between": 5, "return_delay": 3},
}


def generate_ior_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1093 + DIFFICULTY_LEVELS.index(difficulty) * 5003)
    config = DIFFICULTY_CONFIG[difficulty]

    n_passages = config["n_passages"]

    # Pick distinct topics for each passage
    topics = rng.sample(PASSAGE_TOPICS, min(n_passages, len(PASSAGE_TOPICS)))
    # If we need more passages than topics, reuse with different values
    while len(topics) < n_passages:
        topics.append(rng.choice(PASSAGE_TOPICS))

    # Build passages
    passages = []
    passage_facts = []
    for topic in topics:
        text, facts = _build_passage(topic, rng)
        passages.append(text)
        passage_facts.append((topic, facts))

    # Generate filler paragraphs between passages
    n_filler = config["n_filler_between"]
    fillers = [generate_filler_paragraph(rng) for _ in range(n_filler * (n_passages - 1))]

    # Build document with passages labeled
    doc_parts = []
    labels = [chr(ord('A') + i) for i in range(n_passages)]
    filler_idx = 0
    for i, (passage, label) in enumerate(zip(passages, labels)):
        doc_parts.append(f"--- Passage {label} ---\n{passage}")
        if i < n_passages - 1 and fillers:
            for _ in range(n_filler):
                if filler_idx < len(fillers):
                    doc_parts.append(fillers[filler_idx])
                    filler_idx += 1

    document = "\n\n".join(doc_parts)

    # Design question sequence: Q1 about X, Q2 about Y, Q3 about X again
    # Pick the "return" passage (X) and "distractor" passage(s) (Y)
    return_passage_idx = rng.randint(0, n_passages - 1)
    other_indices = [i for i in range(n_passages) if i != return_passage_idx]

    # Pick questions — Q1 and Q3 from the SAME passage, Q2 from different passage
    return_topic, return_facts = passage_facts[return_passage_idx]
    available_qs = list(return_topic["questions"])
    rng.shuffle(available_qs)
    q1_text, q1_key = available_qs[0]
    q3_text, q3_key = available_qs[1]  # Different question about same passage

    # Q2 from a different passage
    q2_passage_idx = rng.choice(other_indices)
    q2_topic, q2_facts = passage_facts[q2_passage_idx]
    q2_available = list(q2_topic["questions"])
    rng.shuffle(q2_available)
    q2_text, q2_key = q2_available[0]

    # For harder difficulties, add more intervening questions
    extra_qs = []
    if config["return_delay"] > 1:
        for _ in range(config["return_delay"] - 1):
            eq_idx = rng.choice(other_indices)
            eq_topic, eq_facts = passage_facts[eq_idx]
            eq_available = list(eq_topic["questions"])
            rng.shuffle(eq_available)
            eq_text, eq_key = eq_available[0]
            extra_qs.append((eq_idx, eq_text, eq_key, eq_facts))

    # Build question sequence
    all_questions = []
    all_questions.append((return_passage_idx, q1_text, q1_key, return_facts, "initial"))
    all_questions.append((q2_passage_idx, q2_text, q2_key, q2_facts, "shift"))
    for eq_idx, eq_text, eq_key, eq_facts in extra_qs:
        all_questions.append((eq_idx, eq_text, eq_key, eq_facts, "shift"))
    all_questions.append((return_passage_idx, q3_text, q3_key, return_facts, "return"))

    # Build prompt
    q_lines = []
    gold_answers = {}
    q_metadata = []
    for qi, (p_idx, q_text, q_key, facts, phase) in enumerate(all_questions, 1):
        label = labels[p_idx]
        q_lines.append(f"Q{qi}. (About Passage {label}) {q_text}")
        answer = facts.get(q_key, "unknown")
        gold_answers[str(qi)] = answer
        q_metadata.append({
            "question_num": qi,
            "passage": label,
            "passage_idx": p_idx,
            "question": q_text,
            "answer_key": q_key,
            "gold": answer,
            "phase": phase,
        })

    q_section = "\n".join(q_lines)

    prompt = (
        f"Read the following {n_passages} passages carefully, then answer each question.\n"
        f"Each question specifies which passage it refers to.\n\n"
        f"{document}\n\n"
        f"Answer each question:\n{q_section}\n\n"
        f"ANSWER:\n"
        + "\n".join(f"{qi}. [your answer]" for qi in range(1, len(all_questions) + 1))
    )

    return TaskInstance(
        task_id=f"ior_{difficulty.lower()}_{instance_idx:03d}",
        task_type="inhibition_return",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answers,
        metadata={
            "n_passages": n_passages,
            "n_filler_between": n_filler,
            "return_passage": labels[return_passage_idx],
            "questions": q_metadata,
            "n_questions": len(all_questions),
            "return_delay": config["return_delay"],
        },
    )


def generate_ior_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_ior_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_ior_dataset()
    print(f"Generated {len(dataset)} Inhibition of Return instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_passages']} passages, "
              f"{s.metadata['n_filler_between']} fillers between, "
              f"{s.metadata['n_questions']} questions, "
              f"return_delay={s.metadata['return_delay']}")
        for q in s.metadata['questions']:
            print(f"    Q{q['question_num']} [{q['phase']}] Passage {q['passage']}: "
                  f"{q['question'][:50]}... → {q['gold']}")
