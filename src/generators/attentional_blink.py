"""
Task A-Novel-2: Attentional Blink (Attention Capacity)

NOVEL CONTRIBUTION: No existing LLM benchmark tests this.
Adapts Rapid Serial Visual Presentation (RSVP) paradigm for text.

A stream of words is presented. Two targets (T1, T2) are embedded at a
controlled lag. The model must identify both. T2 accuracy drops when
the lag between T1 and T2 is short (attentional blink, ~2-8 items).

Ground truth is 100% programmatically guaranteed.
"""

import random
from typing import List
from .base import TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES, COLORS, OBJECTS

ITEMS_PER_DIFFICULTY = 8

# Filler word categories (common, unremarkable words that blend together)
FILLER_WORDS = [
    "table", "window", "garden", "river", "bridge", "market", "street",
    "building", "morning", "evening", "kitchen", "corner", "station",
    "passage", "chapter", "signal", "pattern", "column", "surface",
    "shelter", "chamber", "village", "harbor", "factory", "highway",
    "gallery", "terrace", "balcony", "ceiling", "doorway", "fountain",
    "library", "cabinet", "curtain", "blanket", "lantern", "chimney",
    "platform", "corridor", "stairway", "pavilion", "monument", "district",
]

# Target categories — each T1/T2 is from a distinct, clearly identifiable category
TARGET_CATEGORIES = {
    "uppercase": lambda rng: rng.choice(["DIAMOND", "EMERALD", "SAPPHIRE", "RUBY",
                                          "TOPAZ", "GARNET", "OPAL", "AMETHYST"]),
    "number_word": lambda rng: rng.choice(["seven-hundred", "four-thousand", "nine-million",
                                            "sixty-three", "twenty-eight", "forty-five",
                                            "eighty-one", "thirteen"]),
    "animal_caps": lambda rng: rng.choice(["ELEPHANT", "GIRAFFE", "PENGUIN", "DOLPHIN",
                                            "LEOPARD", "PEACOCK", "OCTOPUS", "BUFFALO"]),
}

DIFFICULTY_CONFIG = {
    "Easy":     {"stream_length": 20, "lag": 8,  "n_fillers_between": 0, "t1_category": "uppercase", "t2_category": "number_word"},
    "Medium":   {"stream_length": 30, "lag": 5,  "n_fillers_between": 0, "t1_category": "uppercase", "t2_category": "number_word"},
    "Hard":     {"stream_length": 40, "lag": 3,  "n_fillers_between": 0, "t1_category": "uppercase", "t2_category": "animal_caps"},
    "Expert":   {"stream_length": 50, "lag": 2,  "n_fillers_between": 0, "t1_category": "uppercase", "t2_category": "number_word"},
    "Frontier": {"stream_length": 80, "lag": 1,  "n_fillers_between": 0, "t1_category": "number_word", "t2_category": "uppercase"},
}


def generate_blink_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1061 + DIFFICULTY_LEVELS.index(difficulty) * 3701)
    config = DIFFICULTY_CONFIG[difficulty]

    stream_length = config["stream_length"]
    lag = config["lag"]

    # Generate T1 and T2
    t1 = TARGET_CATEGORIES[config["t1_category"]](rng)
    t2 = TARGET_CATEGORIES[config["t2_category"]](rng)

    # Place T1 in first third of stream, T2 exactly `lag` positions after
    earliest_t1 = max(3, stream_length // 5)
    latest_t1 = stream_length // 2 - lag
    t1_pos = rng.randint(earliest_t1, max(earliest_t1, latest_t1))
    t2_pos = t1_pos + lag

    # Build stream
    stream = []
    used_fillers = set()
    for i in range(stream_length):
        if i == t1_pos:
            stream.append(t1)
        elif i == t2_pos:
            stream.append(t2)
        else:
            # Pick a filler word not yet used (if possible)
            available = [w for w in FILLER_WORDS if w not in used_fillers]
            if not available:
                available = FILLER_WORDS
            word = rng.choice(available)
            used_fillers.add(word)
            stream.append(word)

    # Format stream as numbered list (RSVP style)
    stream_text = "\n".join(f"{i+1}. {word}" for i, word in enumerate(stream))

    # Build prompt
    prompt = (
        f"Below is a rapid word stream of {stream_length} items. "
        f"Most words are common nouns. Two special target items are hidden in the stream:\n"
        f"  - Target 1 (T1): an ALL-CAPS word\n"
        f"  - Target 2 (T2): appears shortly after T1\n\n"
        f"Read the entire stream carefully, then report both targets.\n\n"
        f"Word stream:\n{stream_text}\n\n"
        f"ANSWER:\n"
        f"T1: [the first target word]\n"
        f"T2: [the second target word]"
    )

    gold_answer = {"T1": t1, "T2": t2}

    return TaskInstance(
        task_id=f"blink_{difficulty.lower()}_{instance_idx:03d}",
        task_type="blink",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answer,
        metadata={
            "stream_length": stream_length,
            "lag": lag,
            "t1": t1,
            "t2": t2,
            "t1_position": t1_pos,
            "t2_position": t2_pos,
            "t1_category": config["t1_category"],
            "t2_category": config["t2_category"],
        },
    )


def generate_blink_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_blink_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_blink_dataset()
    print(f"Generated {len(dataset)} Attentional Blink instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: stream={s.metadata['stream_length']}, lag={s.metadata['lag']}, "
              f"T1={s.metadata['t1']}@{s.metadata['t1_position']}, "
              f"T2={s.metadata['t2']}@{s.metadata['t2_position']}")
