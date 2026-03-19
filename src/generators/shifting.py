"""
Task D: Rule Shift (Attention Shifting)

Tests ability to switch attentional focus when the task rule changes mid-stream.
Cognitive construct: §7.3.2 Attention Shifting
Paradigm source: WCST + Task Interference (EMNLP 2024)

The model classifies items by Rule 1, then a rule change occurs, and it must
apply Rule 2. Perseveration errors (using old rule post-switch) are measured.
"""

import random
from typing import List, Dict, Tuple
from .base import TaskInstance, DIFFICULTY_LEVELS

ITEMS_PER_DIFFICULTY = 8

# Classification rules with deterministic answers
WORD_BANK = [
    # (word, category, first_letter_group, syllables, length_group)
    ("tiger", "animal", "late", 2, "short"),
    ("hammer", "object", "early", 2, "medium"),
    ("banana", "food", "early", 3, "medium"),
    ("eagle", "animal", "early", 2, "short"),
    ("pencil", "object", "late", 2, "medium"),
    ("orange", "food", "late", 2, "medium"),
    ("salmon", "food", "late", 2, "medium"),
    ("wrench", "object", "late", 1, "medium"),
    ("falcon", "animal", "early", 2, "medium"),
    ("mango", "food", "early", 2, "short"),
    ("candle", "object", "early", 2, "medium"),
    ("lobster", "animal", "early", 2, "long"),
    ("walnut", "food", "late", 2, "medium"),
    ("basket", "object", "early", 2, "medium"),
    ("donkey", "animal", "early", 2, "medium"),
    ("cherry", "food", "early", 2, "medium"),
    ("mirror", "object", "early", 2, "medium"),
    ("parrot", "animal", "late", 2, "medium"),
    ("pepper", "food", "late", 2, "medium"),
    ("needle", "object", "late", 2, "medium"),
    ("rabbit", "animal", "late", 2, "medium"),
    ("turnip", "food", "late", 2, "medium"),
    ("shovel", "object", "late", 2, "medium"),
    ("pigeon", "animal", "late", 2, "medium"),
]

RULES = {
    "category": {
        "instruction": "Classify each word by CATEGORY: animal, food, or object",
        "get_answer": lambda w: w[1],  # category field
        "answers": ["animal", "food", "object"],
    },
    "first_letter": {
        "instruction": "Classify each word by FIRST LETTER: 'early' if A-M, 'late' if N-Z",
        "get_answer": lambda w: w[2],  # first_letter_group field
        "answers": ["early", "late"],
    },
    "syllables": {
        "instruction": "Classify each word by SYLLABLE COUNT: '1' for one syllable, '2' for two, '3+' for three or more",
        "get_answer": lambda w: "3+" if w[3] >= 3 else str(w[3]),
        "answers": ["1", "2", "3+"],
    },
}

# Rule pairs by similarity level
RULE_PAIRS = {
    "low":    [("category", "first_letter"), ("first_letter", "category")],
    "medium": [("category", "syllables"), ("syllables", "category")],
    "high":   [("first_letter", "syllables"), ("syllables", "first_letter")],
}

DIFFICULTY_CONFIG = {
    "Easy":   {"pre_items": 3, "post_items": 3, "similarity": "low",    "warning": "header"},
    "Medium": {"pre_items": 4, "post_items": 4, "similarity": "low",    "warning": "inline"},
    "Hard":   {"pre_items": 5, "post_items": 5, "similarity": "medium", "warning": "subtle"},
    "Expert":   {"pre_items": 5, "post_items": 5,  "similarity": "high",   "warning": "buried"},
    "Frontier": {"pre_items": 5, "post_items": 20, "similarity": "high",   "warning": "none",
                 "triple_rule": True},
}


def _format_warning(warning_type: str, new_rule_instruction: str) -> str:
    if warning_type == "header":
        return f"\n--- RULE CHANGE ---\nNEW RULE: {new_rule_instruction}\n"
    elif warning_type == "inline":
        return f"\nRule change: {new_rule_instruction}\n"
    elif warning_type == "subtle":
        return f"\n(Note: from this point, {new_rule_instruction.lower()})\n"
    elif warning_type == "buried":
        return f"\nContinue with the following adjustment — {new_rule_instruction.lower()}\n"
    else:  # none — no explicit warning at all
        return f"\n{new_rule_instruction}\n"


def generate_shifting_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1009 + DIFFICULTY_LEVELS.index(difficulty) * 6151)
    config = DIFFICULTY_CONFIG[difficulty]

    triple_rule = config.get("triple_rule", False)

    if triple_rule:
        # Triple rule change: Rule1→Rule2→Rule3
        all_rule_names = list(RULES.keys())
        rng.shuffle(all_rule_names)
        rule1_name, rule2_name, rule3_name = all_rule_names[:3]
        rule1, rule2, rule3 = RULES[rule1_name], RULES[rule2_name], RULES[rule3_name]

        pre_items = config["pre_items"]
        post_items = config["post_items"]
        # Split post items into two segments for Rule2 and Rule3
        mid_items = post_items // 2
        final_items = post_items - mid_items

        total_items = pre_items + post_items
        selected_words = rng.sample(WORD_BANK, min(total_items, len(WORD_BANK)))

        seg1_words = selected_words[:pre_items]
        seg2_words = selected_words[pre_items:pre_items + mid_items]
        seg3_words = selected_words[pre_items + mid_items:]

        seg1_answers = [(w[0], rule1["get_answer"](w)) for w in seg1_words]
        seg2_answers = [(w[0], rule2["get_answer"](w)) for w in seg2_words]
        seg3_answers = [(w[0], rule3["get_answer"](w)) for w in seg3_words]
        perseveration_answers = (
            [(w[0], rule1["get_answer"](w)) for w in seg2_words] +
            [(w[0], rule2["get_answer"](w)) for w in seg3_words]
        )

        warning1 = _format_warning(config["warning"], rule2["instruction"])
        warning2 = _format_warning(config["warning"], rule3["instruction"])

        lines = [f"Classify each word below according to the CURRENT rule.",
                 f"The rule may change partway through — pay close attention.\n",
                 f"RULE: {rule1['instruction']}\n"]

        item_num = 1
        for word, _ in seg1_answers:
            lines.append(f"{item_num}. {word} → ?")
            item_num += 1

        lines.append(warning1)

        for word, _ in seg2_answers:
            lines.append(f"{item_num}. {word} → ?")
            item_num += 1

        lines.append(warning2)

        for word, _ in seg3_answers:
            lines.append(f"{item_num}. {word} → ?")
            item_num += 1

        lines.append(f"\nFormat your answer as:")
        lines.append(f"ANSWER:")
        for i in range(1, total_items + 1):
            lines.append(f"{i}. [classification]")

        prompt = "\n".join(lines)

        gold_all = seg1_answers + seg2_answers + seg3_answers
        gold_answer = {str(i+1): ans for i, (_, ans) in enumerate(gold_all)}

        return TaskInstance(
            task_id=f"shifting_{difficulty.lower()}_{instance_idx:03d}",
            task_type="shifting",
            difficulty=difficulty,
            prompt=prompt,
            gold_answer=gold_answer,
            metadata={
                "rule1": rule1_name,
                "rule2": rule2_name,
                "rule3": rule3_name,
                "triple_rule": True,
                "rule_similarity": config["similarity"],
                "warning_type": config["warning"],
                "pre_items": pre_items,
                "post_items": post_items,
                "switch_points": [pre_items, pre_items + mid_items],
                "pre_answers": seg1_answers,
                "post_answers": seg2_answers + seg3_answers,
                "perseveration_answers": perseveration_answers,
            },
        )

    # Original 2-rule path
    pair = rng.choice(RULE_PAIRS[config["similarity"]])
    rule1_name, rule2_name = pair
    rule1, rule2 = RULES[rule1_name], RULES[rule2_name]

    total_items = config["pre_items"] + config["post_items"]
    selected_words = rng.sample(WORD_BANK, min(total_items, len(WORD_BANK)))

    pre_words = selected_words[:config["pre_items"]]
    post_words = selected_words[config["pre_items"]:]

    # Compute ground truth
    pre_answers = [(w[0], rule1["get_answer"](w)) for w in pre_words]
    post_answers = [(w[0], rule2["get_answer"](w)) for w in post_words]
    # Also compute what OLD rule would give for post items (to detect perseveration)
    perseveration_answers = [(w[0], rule1["get_answer"](w)) for w in post_words]

    # Build prompt
    warning = _format_warning(config["warning"], rule2["instruction"])

    lines = [f"Classify each word below according to the CURRENT rule.",
             f"The rule may change partway through — pay close attention.\n",
             f"RULE: {rule1['instruction']}\n"]

    item_num = 1
    for word, _ in pre_answers:
        lines.append(f"{item_num}. {word} → ?")
        item_num += 1

    lines.append(warning)

    for word, _ in post_answers:
        lines.append(f"{item_num}. {word} → ?")
        item_num += 1

    lines.append(f"\nFormat your answer as:")
    lines.append(f"ANSWER:")
    for i in range(1, total_items + 1):
        lines.append(f"{i}. [classification]")

    prompt = "\n".join(lines)

    gold_all = pre_answers + post_answers
    gold_answer = {str(i+1): ans for i, (_, ans) in enumerate(gold_all)}

    return TaskInstance(
        task_id=f"shifting_{difficulty.lower()}_{instance_idx:03d}",
        task_type="shifting",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answer,
        metadata={
            "rule1": rule1_name,
            "rule2": rule2_name,
            "rule_similarity": config["similarity"],
            "warning_type": config["warning"],
            "pre_items": config["pre_items"],
            "post_items": config["post_items"],
            "switch_point": config["pre_items"],
            "pre_answers": pre_answers,
            "post_answers": post_answers,
            "perseveration_answers": perseveration_answers,
        },
    )


def generate_shifting_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_shifting_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_shifting_dataset()
    print(f"Generated {len(dataset)} Rule Shift instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['rule1']}→{s.metadata['rule2']}, "
              f"similarity={s.metadata['rule_similarity']}, "
              f"warning={s.metadata['warning_type']}")
        print(f"    Pre: {s.metadata['pre_answers']}")
        print(f"    Post: {s.metadata['post_answers']}")
