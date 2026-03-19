"""
Semantic Needle-in-a-Haystack (Semantic NIAH)

Adapted from the NoLiMa (ICML 2025) insight: the needle and question share
ZERO lexical overlap, forcing semantic inference rather than keyword matching.
11/12 frontier models drop below 50% accuracy at 32K tokens.

The needle fact is embedded in a long document of filler paragraphs.
The question asks about the needle using completely different vocabulary.
No keyword matching is possible — pure semantic understanding is required.

Example:
  Needle: "The orchestra's first performance of the season will take place
           on the fourteenth of November"
  Question: "When is the musical ensemble's debut concert scheduled?"
  Gold: "the fourteenth of November"
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

# Difficulty -> number of filler paragraphs
DIFFICULTY_CONFIG = {
    "Easy": {"n_paragraphs": 15},
    "Medium": {"n_paragraphs": 40},
    "Hard": {"n_paragraphs": 80},
    "Expert": {"n_paragraphs": 120},
    "Frontier": {"n_paragraphs": 200},
}

# ─── Needle-Question Templates ───────────────────────────────────────
# Each template produces a needle sentence and a semantically equivalent
# question that shares ZERO content words with the needle.
# Fill-in values ensure variety across instances.

PROJECTS = [
    "Helios", "Meridian", "Tidewater", "Lodestar", "Canopy",
    "Bastion", "Foxglove", "Granite", "Nimbus", "Quartz",
]
PROJECT_SYNONYMS = [
    "Helios initiative", "Meridian endeavor", "Tidewater venture",
    "Lodestar undertaking", "Canopy effort", "Bastion program",
    "Foxglove assignment", "Granite operation", "Nimbus scheme",
    "Quartz engagement",
]

LOCATIONS = [
    "Ridgemont", "Ashvale", "Clearwater", "Thornfield", "Windmere",
    "Blackstone", "Oakbridge", "Fernhollow", "Copperhill", "Stonereach",
]
LOCATION_SYNONYMS = [
    "Ridgemont site", "Ashvale facility", "Clearwater station",
    "Thornfield outpost", "Windmere checkpoint", "Blackstone hub",
    "Oakbridge center", "Fernhollow depot", "Copperhill terminal",
    "Stonereach base",
]

DEPARTMENTS = [
    "logistics", "procurement", "compliance", "operations", "analytics",
    "maintenance", "distribution", "acquisitions", "oversight", "coordination",
]
DEPARTMENT_SYNONYMS = [
    "supply chain", "purchasing", "regulatory affairs", "field management",
    "data science", "upkeep", "shipping", "sourcing", "supervision",
    "planning",
]

EVENTS = [
    "annual review", "quarterly summit", "board meeting",
    "stakeholder briefing", "strategy session",
]
EVENT_SYNONYMS = [
    "yearly evaluation", "seasonal conference", "directors' assembly",
    "investor presentation", "planning workshop",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth",
    "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
    "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth", "twentieth", "twenty-first",
    "twenty-second", "twenty-third", "twenty-fourth", "twenty-fifth",
    "twenty-sixth", "twenty-seventh", "twenty-eighth",
]


def _build_templates(rng: random.Random):
    """Return a list of (needle_sentence, question, gold_answer) tuples.

    Each call generates fresh randomized fills so that templates are not
    reused verbatim across instances.
    """
    name1 = rng.choice(FIRST_NAMES)
    name2 = rng.choice([n for n in FIRST_NAMES if n != name1])
    name3 = rng.choice([n for n in FIRST_NAMES if n not in (name1, name2)])

    proj_idx = rng.randint(0, len(PROJECTS) - 1)
    project = PROJECTS[proj_idx]
    project_syn = PROJECT_SYNONYMS[proj_idx]

    loc_idx = rng.randint(0, len(LOCATIONS) - 1)
    location = LOCATIONS[loc_idx]
    location_syn = LOCATION_SYNONYMS[loc_idx]

    dept_idx = rng.randint(0, len(DEPARTMENTS) - 1)
    department = DEPARTMENTS[dept_idx]
    department_syn = DEPARTMENT_SYNONYMS[dept_idx]

    evt_idx = rng.randint(0, len(EVENTS) - 1)
    event = EVENTS[evt_idx]
    event_syn = EVENT_SYNONYMS[evt_idx]

    city1 = rng.choice(CITIES)
    city2 = rng.choice([c for c in CITIES if c != city1])

    number = rng.randint(12, 9800)
    measurement = round(rng.uniform(2.0, 99.0), 1)
    day_idx = rng.randint(0, 27)
    day_ord = ORDINALS[day_idx]
    day_num = day_idx + 1
    month = rng.choice(MONTHS)
    percentage = rng.randint(3, 97)
    duration = rng.randint(2, 48)

    templates = [
        # 1. Person assigned to project
        (
            f"The specialist assigned to the {project} case is {name1}.",
            f"Who is the individual designated to handle the {project_syn} matter?",
            name1,
        ),
        # 2. Shipment count from city
        (
            f"The shipment from {city1} contained exactly {number} units.",
            f"How many items arrived in the delivery originating from {city1}?",
            str(number),
        ),
        # 3. Deadline for event
        (
            f"The deadline for the {event} has been set to the {day_ord} of {month}.",
            f"By when must the {event_syn} be completed?",
            f"the {day_ord} of {month}",
        ),
        # 4. Measurement at location
        (
            f"The final measurement recorded at the {location} station was {measurement} degrees.",
            f"What reading was obtained at the {location_syn}?",
            f"{measurement} degrees",
        ),
        # 5. Person heading department
        (
            f"{name2} was appointed to lead the {department} division last quarter.",
            f"Who took charge of the {department_syn} unit in the previous period?",
            name2,
        ),
        # 6. Budget allocation
        (
            f"The total budget allocated for the {project} initiative was {number} thousand dollars.",
            f"What is the financial outlay earmarked for the {project_syn}?",
            f"{number} thousand dollars",
        ),
        # 7. Duration of inspection
        (
            f"The inspection of the {location} facility lasted {duration} hours in total.",
            f"How long did the examination of the {location_syn} take to finish?",
            f"{duration} hours",
        ),
        # 8. Person responsible for city office
        (
            f"The {city2} regional office is managed by {name3}.",
            f"Who is the administrator overseeing the {city2} branch?",
            name3,
        ),
        # 9. Percentage change
        (
            f"Output from the {department} team increased by {percentage} percent this cycle.",
            f"By what proportion did the {department_syn} group's production grow in the current period?",
            f"{percentage} percent",
        ),
        # 10. Scheduled date for concert/performance
        (
            f"The orchestra's first performance of the season will take place on the {day_ord} of {month}.",
            f"When is the musical ensemble's debut concert scheduled?",
            f"the {day_ord} of {month}",
        ),
    ]

    return templates


def generate_sniah_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1051 + DIFFICULTY_LEVELS.index(difficulty) * 7919)
    config = DIFFICULTY_CONFIG[difficulty]
    n_paragraphs = config["n_paragraphs"]

    # Pick one template
    templates = _build_templates(rng)
    needle_sentence, question, gold_answer = rng.choice(templates)

    # Decide needle placement depth (0.0 = start, 1.0 = end)
    needle_depth_ratio = round(rng.uniform(0.1, 0.9), 2)
    insert_pos = max(1, int(needle_depth_ratio * n_paragraphs))

    # Generate filler paragraphs
    fillers = [generate_filler_paragraph(rng) for _ in range(n_paragraphs)]

    # Insert needle paragraph
    fillers.insert(insert_pos, needle_sentence)

    passage = "\n\n".join(fillers)
    word_count = len(passage.split())

    # Build prompt
    prompt = (
        "Read the following document carefully and answer the question at the end.\n\n"
        "---BEGIN DOCUMENT---\n"
        f"{passage}\n"
        "---END DOCUMENT---\n\n"
        f"Question: {question}\n\n"
        "Answer with ONLY the relevant fact from the document. "
        "Be concise — do not repeat the question or add explanation."
    )

    return TaskInstance(
        task_id=f"sniah_{difficulty.lower()}_{instance_idx:03d}",
        task_type="semantic_niah",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold_answer,
        metadata={
            "n_paragraphs": n_paragraphs,
            "needle_depth_ratio": needle_depth_ratio,
            "needle_sentence": needle_sentence,
            "question": question,
            "word_count": word_count,
        },
    )


def generate_sniah_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_sniah_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_sniah_dataset()
    print(f"Generated {len(dataset)} Semantic NIAH instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"\n  [{diff}] n_paragraphs={s.metadata['n_paragraphs']}, "
              f"word_count={s.metadata['word_count']}, "
              f"needle_depth={s.metadata['needle_depth_ratio']}")
        print(f"    Needle:   {s.metadata['needle_sentence']}")
        print(f"    Question: {s.metadata['question']}")
        print(f"    Gold:     {s.gold_answer}")
