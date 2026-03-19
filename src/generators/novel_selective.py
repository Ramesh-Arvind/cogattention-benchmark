"""
Task C-Novel: Semantic Stroop (Selective Attention)

NOVEL CONTRIBUTION: No existing LLM benchmark tests this.
Tests whether LLMs can suppress their "correction reflex" when
encountering embedded factual errors — they must answer the ASKED
question, not correct the error.

This is a text analogue of the classic Stroop interference task.
"""

import random
from typing import List
from .base import TaskInstance, DIFFICULTY_LEVELS

ITEMS_PER_DIFFICULTY = 8

# (sentence_with_error, asked_question, correct_answer, stroop_trap_answer)
STROOP_ITEMS = [
    # Geography errors
    {
        "sentence": "Paris is the capital of {wrong_country}",
        "question": "What country is mentioned in the sentence?",
        "correct": "{wrong_country}",
        "trap": "France",  # model may correct the error
        "wrong_options": ["Germany", "Spain", "Italy", "Brazil", "Japan", "Australia"],
        "category": "geography",
    },
    {
        "sentence": "The {wrong_river} is the longest river in {continent}",
        "question": "What river is mentioned?",
        "correct": "{wrong_river}",
        "trap": None,  # trap is correcting to actual longest river
        "wrong_options": ["Thames", "Seine", "Danube", "Rhine", "Volga"],
        "continents": ["Africa", "South America", "Asia", "Europe"],
        "category": "geography",
    },
    # Math errors
    {
        "sentence": "According to the report, {a} + {b} = {wrong_sum}",
        "question": "What number does the report state as the sum?",
        "correct": "{wrong_sum}",
        "trap": None,  # trap is giving the actual sum
        "category": "math",
    },
    # Science errors
    {
        "sentence": "Water boils at {wrong_temp} degrees Celsius at sea level",
        "question": "What temperature is mentioned in the sentence?",
        "correct": "{wrong_temp}",
        "trap": "100",
        "wrong_temps": ["85", "92", "110", "75", "120", "150"],
        "category": "science",
    },
    # Historical errors
    {
        "sentence": "The {event} occurred in {wrong_year}",
        "question": "What year is mentioned in the sentence?",
        "correct": "{wrong_year}",
        "trap": None,
        "events": ["French Revolution", "Moon landing", "Fall of the Berlin Wall",
                    "signing of the Magna Carta"],
        "wrong_years": ["1801", "1955", "1997", "1302", "1776", "1848"],
        "category": "history",
    },
]

DIFFICULTY_CONFIG = {
    "Easy":   {"n_items": 3, "context": "isolated",  "instruction_clarity": "explicit"},
    "Medium": {"n_items": 4, "context": "paragraph",  "instruction_clarity": "clear"},
    "Hard":   {"n_items": 6, "context": "narrative",   "instruction_clarity": "subtle"},
    "Expert":   {"n_items": 8,  "context": "multi_error", "instruction_clarity": "minimal"},
    "Frontier": {"n_items": 15, "context": "multi_error", "instruction_clarity": "hidden"},
}


def _generate_stroop_item(rng: random.Random) -> dict:
    """Generate a single Stroop interference item."""
    template = rng.choice(STROOP_ITEMS)
    cat = template["category"]

    if cat == "geography" and "wrong_country" in template["sentence"]:
        wrong = rng.choice(template["wrong_options"])
        sentence = template["sentence"].format(wrong_country=wrong)
        correct = wrong
        trap = template["trap"]
        question = template["question"]
    elif cat == "geography" and "wrong_river" in template["sentence"]:
        wrong_river = rng.choice(template["wrong_options"])
        continent = rng.choice(template["continents"])
        sentence = template["sentence"].format(wrong_river=wrong_river, continent=continent)
        correct = wrong_river
        trap = "the actual longest river"
        question = template["question"]
    elif cat == "math":
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        real_sum = a + b
        wrong_sum = real_sum + rng.choice([-3, -2, -1, 1, 2, 3]) * rng.randint(1, 5)
        sentence = template["sentence"].format(a=a, b=b, wrong_sum=wrong_sum)
        correct = str(wrong_sum)
        trap = str(real_sum)
        question = template["question"]
    elif cat == "science":
        wrong_temp = rng.choice(template["wrong_temps"])
        sentence = template["sentence"].format(wrong_temp=wrong_temp)
        correct = wrong_temp
        trap = template["trap"]
        question = template["question"]
    elif cat == "history":
        event = rng.choice(template["events"])
        wrong_year = rng.choice(template["wrong_years"])
        sentence = template["sentence"].format(event=event, wrong_year=wrong_year)
        correct = wrong_year
        trap = "the actual year"
        question = template["question"]
    else:
        sentence = "Error in generation"
        correct = "N/A"
        trap = "N/A"
        question = "N/A"

    return {
        "sentence": sentence,
        "question": question,
        "correct": correct,
        "trap": trap,
        "category": cat,
    }


def generate_stroop_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1039 + DIFFICULTY_LEVELS.index(difficulty) * 3571)
    config = DIFFICULTY_CONFIG[difficulty]

    items = [_generate_stroop_item(rng) for _ in range(config["n_items"])]

    # Build prompt based on instruction clarity
    if config["instruction_clarity"] == "explicit":
        preamble = (
            "IMPORTANT: For each sentence below, answer the question about "
            "what the sentence LITERALLY SAYS. Do NOT correct any factual errors — "
            "report exactly what is stated in the sentence, even if it is wrong.\n"
        )
    elif config["instruction_clarity"] == "clear":
        preamble = (
            "Answer each question based ONLY on what the sentence states. "
            "Do not correct errors in the sentences.\n"
        )
    elif config["instruction_clarity"] == "subtle":
        preamble = (
            "For each item, answer the question about the sentence as written.\n"
        )
    elif config["instruction_clarity"] == "minimal":
        preamble = "Answer each question.\n"
    else:  # hidden — instruction is deeply embedded and indirect
        preamble = (
            "You will read a series of statements. Each one contains a factual claim. "
            "Some claims are accurate, others are not. For each item, a question is posed. "
            "Note: the questions are about the literal content of the sentences, not about "
            "real-world correctness. Keep this distinction in mind as you proceed.\n"
        )

    lines = [preamble]
    for idx, item in enumerate(items, 1):
        lines.append(f'{idx}. Sentence: "{item["sentence"]}"')
        lines.append(f'   Question: {item["question"]}')
        lines.append("")

    lines.append("ANSWER:")
    for idx in range(1, len(items) + 1):
        lines.append(f"{idx}. [your answer]")

    prompt = "\n".join(lines)

    gold_answer = {str(i+1): item["correct"] for i, item in enumerate(items)}
    trap_answers = {str(i+1): item["trap"] for i, item in enumerate(items)}

    return TaskInstance(
        task_id=f"stroop_{difficulty.lower()}_{instance_idx:03d}",
        task_type="stroop",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answer,
        metadata={
            "n_items": config["n_items"],
            "instruction_clarity": config["instruction_clarity"],
            "items": items,
            "trap_answers": trap_answers,
            "categories": [item["category"] for item in items],
        },
    )


def generate_stroop_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_stroop_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_stroop_dataset()
    print(f"Generated {len(dataset)} Semantic Stroop instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_items']} items, clarity={s.metadata['instruction_clarity']}")
        for idx, item in enumerate(s.metadata['items'], 1):
            print(f"    {idx}. '{item['sentence'][:60]}...'")
            print(f"       Correct: {item['correct']} | Trap: {item['trap']}")
