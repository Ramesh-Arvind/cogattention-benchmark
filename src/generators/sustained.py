"""
Task B: Vigilance Probe (Sustained Attention)

Tests whether attention degrades over long contexts.
Cognitive construct: §7.3.2 Sustained Attention
Paradigm source: CPT + NoLiMa + Context Rot

Targets from a specific category are scattered at known positions in a long
procedurally-generated document. Near-miss distractors test discrimination.
"""

import random
from typing import List, Dict
from .base import (
    TaskInstance, CATEGORY_POOLS, DIFFICULTY_LEVELS,
    generate_filler_paragraph, FIRST_NAMES, CITIES,
)

DIFFICULTY_CONFIG = {
    "Easy":   {"n_targets": 5,  "n_nearmiss": 2,  "filler_paragraphs": 12},
    "Medium": {"n_targets": 8,  "n_nearmiss": 5,  "filler_paragraphs": 30},
    "Hard":   {"n_targets": 10, "n_nearmiss": 8,  "filler_paragraphs": 55},
    "Expert":   {"n_targets": 12, "n_nearmiss": 12, "filler_paragraphs": 80},
    "Frontier": {"n_targets": 10, "n_nearmiss": 15, "filler_paragraphs": 150},
}

ITEMS_PER_DIFFICULTY = 8

# Templates to embed a target/nearmiss naturally in prose
TARGET_TEMPLATES = [
    "A {item} was spotted near the old bridge that morning.",
    "{name} mentioned seeing a {item} while crossing the square.",
    "The logbook recorded a {item} at the northern edge of the district.",
    "Among the items catalogued was a {item}, noted without further comment.",
    "Reports from the harbour mentioned a {item} had been observed twice that week.",
    "The survey team documented a {item} in the area surrounding {city}.",
    "{name} recalled that a {item} had appeared briefly near the market.",
    "A {item} was noted in the margin of the inspector's report.",
]


def _select_category(rng: random.Random):
    """Pick a target category and its near-miss pool."""
    categories = ["birds", "metals", "instruments", "rivers"]
    cat = rng.choice(categories)
    targets_pool = CATEGORY_POOLS[cat]
    nearmiss_pool = CATEGORY_POOLS[f"near_{cat}"]
    # Human-readable category description (avoid exact pool name)
    cat_descriptions = {
        "birds": "a type of bird",
        "metals": "a metallic element",
        "instruments": "a musical instrument",
        "rivers": "a river",
    }
    return cat, cat_descriptions[cat], targets_pool, nearmiss_pool


def generate_sustained_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 997 + DIFFICULTY_LEVELS.index(difficulty) * 7919)
    config = DIFFICULTY_CONFIG[difficulty]

    cat_name, cat_desc, targets_pool, nearmiss_pool = _select_category(rng)

    # Sample targets and near-misses
    targets = rng.sample(targets_pool, min(config["n_targets"], len(targets_pool)))
    nearmisses = rng.sample(nearmiss_pool, min(config["n_nearmiss"], len(nearmiss_pool)))

    n_fillers = config["filler_paragraphs"]

    # Build document: interleave filler paragraphs with target/nearmiss sentences
    paragraphs = []
    for _ in range(n_fillers):
        paragraphs.append(generate_filler_paragraph(rng))

    # Insert targets at uniformly distributed positions
    total_slots = len(paragraphs)
    target_positions = sorted(rng.sample(range(total_slots), min(len(targets), total_slots)))
    nearmiss_positions = sorted(rng.sample(
        [i for i in range(total_slots) if i not in target_positions],
        min(len(nearmisses), total_slots - len(targets))
    ))

    # Create target and nearmiss sentences
    target_sentences = {}
    for pos, tgt in zip(target_positions, targets):
        tmpl = rng.choice(TARGET_TEMPLATES)
        sentence = tmpl.format(
            item=tgt,
            name=rng.choice(FIRST_NAMES),
            city=rng.choice(CITIES),
        )
        target_sentences[pos] = (sentence, tgt)

    nearmiss_sentences = {}
    for pos, nm in zip(nearmiss_positions, nearmisses):
        tmpl = rng.choice(TARGET_TEMPLATES)
        sentence = tmpl.format(
            item=nm,
            name=rng.choice(FIRST_NAMES),
            city=rng.choice(CITIES),
        )
        nearmiss_sentences[pos] = (sentence, nm)

    # Build final document
    doc_parts = []
    target_order = []  # track which targets appear in which quintile
    for i, para in enumerate(paragraphs):
        if i in target_sentences:
            sent, tgt = target_sentences[i]
            doc_parts.append(para + " " + sent)
            target_order.append((i, tgt, i / max(total_slots - 1, 1)))
        elif i in nearmiss_sentences:
            sent, _ = nearmiss_sentences[i]
            doc_parts.append(para + " " + sent)
        else:
            doc_parts.append(para)

    document = "\n\n".join(doc_parts)

    prompt = (
        f"Read the entire document below carefully. Find ALL mentions of "
        f"{cat_desc}. List every {cat_name.rstrip('s')} you find.\n\n"
        f"Important: There may be similar-sounding items that are NOT "
        f"{cat_desc} — do not include those.\n\n"
        f"---\n{document}\n---\n\n"
        f"List ALL {cat_name} mentioned in the document, in the order they appear.\n"
        f"Format: ANSWER: [item1], [item2], [item3], ..."
    )

    gold_answer = [tgt for _, tgt, _ in target_order]

    # Compute quintile positions for vigilance analysis
    quintile_targets = {q: [] for q in range(5)}
    for pos, tgt, frac in target_order:
        q = min(int(frac * 5), 4)
        quintile_targets[q].append(tgt)

    return TaskInstance(
        task_id=f"sustained_{difficulty.lower()}_{instance_idx:03d}",
        task_type="sustained",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answer,
        metadata={
            "category": cat_name,
            "category_description": cat_desc,
            "targets": targets,
            "nearmisses": nearmisses,
            "target_positions": [(pos, tgt) for pos, tgt, _ in target_order],
            "quintile_targets": {str(k): v for k, v in quintile_targets.items()},
            "n_paragraphs": total_slots,
            "doc_word_count": len(document.split()),
        },
    )


def generate_sustained_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_sustained_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_sustained_dataset()
    print(f"Generated {len(dataset)} Vigilance Probe instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_paragraphs']} paragraphs, "
              f"{len(s.gold_answer)} targets, "
              f"{len(s.metadata['nearmisses'])} near-misses, "
              f"~{s.metadata['doc_word_count']} words")
        print(f"    Category: {s.metadata['category_description']}")
        print(f"    Targets: {s.gold_answer}")
