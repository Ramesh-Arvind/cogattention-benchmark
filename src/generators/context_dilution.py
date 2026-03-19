"""
Task: Context Dilution (Attention Tax Measurement)

NOVEL CONTRIBUTION: Isolates the pure "attention tax" of increasing context
length while holding task difficulty CONSTANT.

A trivially easy factual QA task (e.g., "What shade was the automobile?")
is embedded in varying amounts of filler text. Difficulty comes ONLY from
context length scaling: 1K, 4K, 8K, 16K, 32K words.

Key design insight from Context Rot research: includes both COHERENT narrative
filler and SHUFFLED sentence filler variants. Models often perform WORSE on
coherent text because narrative flow captures attention away from the needle.

Anti-cheat: needle and question share ZERO lexical overlap — different
phrasing templates ensure keyword matching cannot shortcut the task.
"""

import random
from typing import List
from .base import (
    TaskInstance,
    DIFFICULTY_LEVELS,
    FIRST_NAMES,
    CITIES,
    generate_filler_paragraph,
)

ITEMS_PER_DIFFICULTY = 8

# Word count targets per difficulty level
WORD_COUNT_TARGETS = {
    "Easy": 1_000,
    "Medium": 4_000,
    "Hard": 8_000,
    "Expert": 16_000,
    "Frontier": 32_000,
}

# Vehicle types for needle construction
VEHICLE_TYPES = [
    "sedan", "truck", "coupe", "hatchback", "wagon", "van",
    "convertible", "SUV", "minivan", "roadster", "crossover",
    "pickup", "compact", "limousine", "sports car",
]

# Colors for the needle — distinct from base.COLORS to avoid pool contamination
VEHICLE_COLORS = [
    "maroon", "turquoise", "charcoal", "lavender", "magenta",
    "olive", "burgundy", "cerulean", "vermillion", "pewter",
    "sapphire", "tangerine", "platinum", "mahogany", "periwinkle",
]

# Needle templates — describe the vehicle WITHOUT using words from question templates
NEEDLE_TEMPLATES = [
    "The vehicle registered to {person} in the municipal database was noted as {color} in the latest inspection report.",
    "According to the county motor registry, the {vehicle_type} filed under {person}'s name bears the designation {color}.",
    "Municipal transit records confirm that {person} holds registration for a {color} {vehicle_type} as of the last filing period.",
    "The department of motor vehicles listed a {color} {vehicle_type} under the ownership of {person} in their certified ledger.",
    "Inspection documentation filed at the transport bureau identifies {person}'s registered {vehicle_type} as {color}.",
    "Per the notarized title transfer, {person} acquired a {color} {vehicle_type} that remains on file with the licensing authority.",
]

# Question templates — use DIFFERENT vocabulary from needle templates
# No shared content words with needles: no "vehicle", "registered", "municipal", "inspection", etc.
QUESTION_TEMPLATES = [
    "According to the records, what shade was the automobile belonging to {person}?",
    "Based on the documentation, what hue is {person}'s car?",
    "From the information provided, identify the tint of {person}'s motor conveyance.",
    "What chromatic designation does {person}'s personal transport carry in the files?",
    "Referring to the text, state the pigmentation of the auto owned by {person}.",
    "Per the passage, what coloration is attributed to {person}'s wheeled transport?",
]


def _estimate_word_count(text: str) -> int:
    """Rough word count estimation."""
    return len(text.split())


def _generate_filler_block(rng: random.Random, target_words: int) -> List[str]:
    """Generate a list of filler paragraphs totaling approximately target_words."""
    paragraphs = []
    current_words = 0
    while current_words < target_words:
        para = generate_filler_paragraph(rng)
        paragraphs.append(para)
        current_words += _estimate_word_count(para)
    return paragraphs


def generate_dilution_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1049 + DIFFICULTY_LEVELS.index(difficulty) * 7919)

    target_words = WORD_COUNT_TARGETS[difficulty]

    # Pick entities
    person = rng.choice(FIRST_NAMES)
    color = rng.choice(VEHICLE_COLORS)
    vehicle_type = rng.choice(VEHICLE_TYPES)

    # Build needle sentence (no shared keywords with question)
    needle_template = rng.choice(NEEDLE_TEMPLATES)
    needle_sentence = needle_template.format(
        person=person, color=color, vehicle_type=vehicle_type,
    )

    # Build question (different vocabulary)
    question_template = rng.choice(QUESTION_TEMPLATES)
    question = question_template.format(person=person)

    # Decide if this is a shuffled-filler variant (half and half)
    is_shuffled = (instance_idx % 2 == 1)

    # Generate filler paragraphs
    filler_paragraphs = _generate_filler_block(rng, target_words)

    if is_shuffled:
        # Break all paragraphs into sentences, shuffle them, regroup
        all_sentences = []
        for para in filler_paragraphs:
            # Split on period-space or period-end
            sentences = [s.strip() for s in para.split(". ") if s.strip()]
            # Re-add period if missing
            sentences = [s if s.endswith(".") else s + "." for s in sentences]
            all_sentences.extend(sentences)
        rng.shuffle(all_sentences)
        # Regroup into pseudo-paragraphs of 3-5 sentences
        filler_paragraphs = []
        i = 0
        while i < len(all_sentences):
            group_size = rng.randint(3, 5)
            group = all_sentences[i:i + group_size]
            filler_paragraphs.append(" ".join(group))
            i += group_size

    # Determine needle depth: 25%, 50%, or 75%
    depth_options = [0.25, 0.50, 0.75]
    needle_depth = rng.choice(depth_options)

    # Insert needle at the chosen depth
    insert_pos = max(1, int(len(filler_paragraphs) * needle_depth))
    filler_paragraphs.insert(insert_pos, needle_sentence)

    # Build full document
    document = "\n\n".join(filler_paragraphs)
    actual_word_count = _estimate_word_count(document)

    # Construct the prompt
    prompt = (
        "Read the following document carefully and answer the question at the end.\n\n"
        "---\n"
        f"{document}\n"
        "---\n\n"
        f"Question: {question}\n\n"
        "Answer with ONLY the single-word answer. Do not explain."
    )

    return TaskInstance(
        task_id=f"dilution_{difficulty.lower()}_{instance_idx:03d}",
        task_type="context_dilution",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=color,
        metadata={
            "word_count": actual_word_count,
            "target_word_count": target_words,
            "needle_depth": needle_depth,
            "is_shuffled": is_shuffled,
            "needle_sentence": needle_sentence,
            "question": question,
            "person": person,
            "color": color,
            "vehicle_type": vehicle_type,
        },
    )


def generate_dilution_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_dilution_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_dilution_dataset()
    print(f"Generated {len(dataset)} Context Dilution instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        word_counts = [s.metadata["word_count"] for s in subset]
        shuffled_count = sum(1 for s in subset if s.metadata["is_shuffled"])
        depths = [s.metadata["needle_depth"] for s in subset]
        s = subset[0]
        print(
            f"  {diff}: target={s.metadata['target_word_count']} words, "
            f"actual={min(word_counts)}-{max(word_counts)} words, "
            f"shuffled={shuffled_count}/{len(subset)}, "
            f"depths={sorted(set(depths))}"
        )
        print(f"    Needle: \"{s.metadata['needle_sentence'][:80]}...\"")
        print(f"    Question: \"{s.metadata['question']}\"")
        print(f"    Answer: {s.gold_answer}")
