"""
Multi-hop Scattered Reasoning (Attention + Reasoning)

Tests whether a model can attend to and integrate multiple distant pieces of
information scattered across a long document.  Inspired by BABILong (NeurIPS 2024).

Cognitive construct: joint selective attention + working-memory integration.
Facts needed for the answer are placed at controlled positions in a long
filler-paragraph document.  The model must chain 2-3 facts to derive the answer.
"""

import random
from typing import List, Dict, Tuple
from .base import (
    TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES,
    OBJECTS, COLORS, generate_filler_paragraph,
)

DIFFICULTY_CONFIG = {
    "Easy":     {"n_hops": 2, "n_fillers": 15,  "fact_spread": [0.30, 0.60],                 "has_distractors": False},
    "Medium":   {"n_hops": 2, "n_fillers": 35,  "fact_spread": [0.20, 0.70],                 "has_distractors": False},
    "Hard":     {"n_hops": 3, "n_fillers": 60,  "fact_spread": [0.15, 0.50, 0.85],           "has_distractors": False},
    "Expert":   {"n_hops": 3, "n_fillers": 100, "fact_spread": [0.10, 0.50, 0.90],           "has_distractors": False},
    "Frontier": {"n_hops": 3, "n_fillers": 180, "fact_spread": [0.10, 0.50, 0.90],           "has_distractors": True},
}

ITEMS_PER_DIFFICULTY = 8

# ─── Location / entity pools ─────────────────────────────────────────

LOCATIONS = [
    "the third floor archive", "the east wing storage", "the basement vault",
    "the rooftop greenhouse", "the south tower office", "the courtyard shed",
    "the harbor warehouse", "the library annex", "the clock tower room",
    "the garden pavilion", "the marble atrium", "the old chapel",
    "the north corridor cabinet", "the loading dock", "the observatory loft",
    "the records room", "the west gallery", "the cellar workshop",
    "the conservatory", "the guard station",
]

CONTAINERS = [
    "mahogany chest", "iron strongbox", "leather satchel", "oak cabinet",
    "glass display case", "copper lockbox", "canvas duffel bag",
    "wooden crate", "velvet pouch", "steel filing cabinet",
    "wicker basket", "ceramic urn", "tin canister", "bamboo case",
    "linen sack", "brass trunk", "cardboard box", "rosewood drawer",
    "plastic bin", "silver coffer",
]

DEPARTMENTS = [
    "the linguistics division", "the cartography department",
    "the metallurgy lab", "the antiquities office", "the signals bureau",
    "the archival unit", "the logistics branch", "the compliance section",
    "the research annex", "the procurement desk",
]

BUILDINGS = [
    "the Kessler Building", "the Aldrin Annex", "the Meridian Tower",
    "the Voss Institute", "the Patel Center", "the Bergmann Wing",
    "the Novak Pavilion", "the Okafor Complex", "the Lindgren Hall",
    "the Tanaka Block",
]

ROOMS = [
    "room 14", "room 27", "room 33", "room 6", "room 41",
    "room 58", "room 72", "room 19", "room 85", "room 3",
]

DOORS = [
    "the iron gate", "the oak door", "the service entrance",
    "the main portal", "the side passage", "the loading bay door",
    "the archive entrance", "the vault door", "the glass doorway",
    "the fire exit",
]

OFFICES = [
    "the registrar's office", "the director's suite",
    "the compliance office", "the permits desk",
    "the audit chamber", "the intake counter",
    "the dispatch window", "the clearance booth",
    "the filing bureau", "the review panel room",
]

# ─── Chain templates ─────────────────────────────────────────────────
# Each template is a list of fact-format-strings and a question + gold-answer
# format.  Placeholders: {person1}, {person2}, {object}, {color}, {container},
# {location1}, {location2}, {department}, {building}, {room}, {door}, {office}


def _chain_object_transfer(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 1: person→person→location  (2-hop or 3-hop)"""
    people = rng.sample(FIRST_NAMES, 3)
    obj = f"the {rng.choice(COLORS)} {rng.choice(OBJECTS)}"
    loc = rng.choice(LOCATIONS)
    facts = [
        f"{people[0]} handed {obj} to {people[1]}.",
        f"{people[1]} later placed everything received that day in {loc}.",
    ]
    question = f"Where is {obj} now?"
    gold = loc
    pattern = "object_transfer"
    return facts, question, gold, pattern


def _chain_container_nesting(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 2: item→container→location"""
    obj = f"the {rng.choice(COLORS)} {rng.choice(OBJECTS)}"
    container = rng.choice(CONTAINERS)
    loc = rng.choice(LOCATIONS)
    person = rng.choice(FIRST_NAMES)
    facts = [
        f"{person} placed {obj} inside the {container}.",
        f"The {container} was moved to {loc}.",
    ]
    question = f"Where is {obj} now?"
    gold = loc
    pattern = "container_nesting"
    return facts, question, gold, pattern


def _chain_assignment(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 3: task→person→department"""
    task_name = rng.choice([
        "the quarterly audit", "the inventory review", "the compliance report",
        "the safety inspection", "the vendor assessment", "the budget reconciliation",
        "the permit renewal", "the equipment certification",
    ])
    person = rng.choice(FIRST_NAMES)
    dept = rng.choice(DEPARTMENTS)
    facts = [
        f"{task_name.capitalize()} was assigned to {person}.",
        f"{person} was transferred to {dept} the following week.",
    ]
    question = f"Which department is now responsible for {task_name}?"
    gold = dept
    pattern = "assignment_chain"
    return facts, question, gold, pattern


def _chain_document_routing(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 4: document→office→building"""
    doc_name = rng.choice([
        "the sealed envelope", "the classified dossier", "the signed contract",
        "the patent application", "the shipping manifest", "the notarized deed",
        "the insurance claim", "the inspection certificate",
    ])
    office = rng.choice(OFFICES)
    building = rng.choice(BUILDINGS)
    facts = [
        f"{doc_name.capitalize()} was forwarded to {office}.",
        f"All materials at {office} were relocated to {building}.",
    ]
    question = f"Where is {doc_name} now?"
    gold = building
    pattern = "document_routing"
    return facts, question, gold, pattern


def _chain_key_lock(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 5: key→door→room"""
    key_desc = f"the {rng.choice(COLORS)} key"
    door = rng.choice(DOORS)
    room = rng.choice(ROOMS)
    person = rng.choice(FIRST_NAMES)
    facts = [
        f"{person} used {key_desc} to unlock {door}.",
        f"Behind {door} was {room}, which had been sealed for years.",
    ]
    question = f"Which room did {key_desc} give access to?"
    gold = room
    pattern = "key_lock"
    return facts, question, gold, pattern


def _chain_custody(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 6: item→person→person→location (3-hop)"""
    people = rng.sample(FIRST_NAMES, 3)
    obj = f"the {rng.choice(COLORS)} {rng.choice(OBJECTS)}"
    loc = rng.choice(LOCATIONS)
    facts = [
        f"{people[0]} entrusted {obj} to {people[1]} before leaving the city.",
        f"{people[1]} passed {obj} to {people[2]} at the railway station.",
        f"{people[2]} stored all received items in {loc}.",
    ]
    question = f"Where is {obj} that {people[0]} originally had?"
    gold = loc
    pattern = "custody_chain"
    return facts, question, gold, pattern


def _chain_relocation(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 7: item→container→location→final_location (3-hop)"""
    obj = f"the {rng.choice(COLORS)} {rng.choice(OBJECTS)}"
    container = rng.choice(CONTAINERS)
    loc1 = rng.choice(LOCATIONS)
    loc2_choices = [l for l in LOCATIONS if l != loc1]
    loc2 = rng.choice(loc2_choices)
    person = rng.choice(FIRST_NAMES)
    facts = [
        f"{person} sealed {obj} inside the {container}.",
        f"The {container} was shipped to {loc1}.",
        f"Everything stored at {loc1} was subsequently transferred to {loc2}.",
    ]
    question = f"Where is {obj} now?"
    gold = loc2
    pattern = "relocation_chain"
    return facts, question, gold, pattern


def _chain_bureaucratic(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 8: document→person→office→building (3-hop)"""
    doc_name = rng.choice([
        "the funding proposal", "the research grant", "the incident report",
        "the transfer request", "the authorization form", "the expense voucher",
    ])
    person = rng.choice(FIRST_NAMES)
    office = rng.choice(OFFICES)
    building = rng.choice(BUILDINGS)
    facts = [
        f"{doc_name.capitalize()} was prepared by {person}.",
        f"{person} submitted it to {office} for processing.",
        f"{office} forwarded all pending documents to {building}.",
    ]
    question = f"Where is {doc_name} now?"
    gold = building
    pattern = "bureaucratic_chain"
    return facts, question, gold, pattern


def _chain_delegation(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 9: task→person→person→department (3-hop)"""
    task_name = rng.choice([
        "the translation project", "the calibration task",
        "the data migration job", "the network upgrade",
        "the policy review", "the outreach initiative",
    ])
    people = rng.sample(FIRST_NAMES, 2)
    dept = rng.choice(DEPARTMENTS)
    facts = [
        f"{task_name.capitalize()} was initially given to {people[0]}.",
        f"{people[0]} delegated the work to {people[1]}.",
        f"{people[1]} completed it under {dept}.",
    ]
    question = f"Which department handled {task_name}?"
    gold = dept
    pattern = "delegation_chain"
    return facts, question, gold, pattern


def _chain_artifact(rng: random.Random) -> Tuple[List[str], str, str, str]:
    """Pattern 10: artifact→container→room→building (3-hop)"""
    obj = f"the {rng.choice(COLORS)} {rng.choice(OBJECTS)}"
    container = rng.choice(CONTAINERS)
    room = rng.choice(ROOMS)
    building = rng.choice(BUILDINGS)
    facts = [
        f"{obj.capitalize()} was wrapped and placed in the {container}.",
        f"The {container} was stored in {room}.",
        f"The contents of {room} were moved to {building} during renovation.",
    ]
    question = f"Where is {obj} now?"
    gold = building
    pattern = "artifact_chain"
    return facts, question, gold, pattern


# Registry split by hop count
_2HOP_CHAINS = [
    _chain_object_transfer,
    _chain_container_nesting,
    _chain_assignment,
    _chain_document_routing,
    _chain_key_lock,
]

_3HOP_CHAINS = [
    _chain_custody,
    _chain_relocation,
    _chain_bureaucratic,
    _chain_delegation,
    _chain_artifact,
]


# ─── Distractor chain generation (Frontier only) ─────────────────────

def _generate_distractor_chain(
    rng: random.Random,
    real_facts: List[str],
    n_hops: int,
) -> List[str]:
    """Create a plausible but WRONG chain with similar entity types.

    Uses different entities so the chain is wrong but structurally similar,
    making it harder for the model to pick the correct thread.
    """
    # Generate a completely independent chain using a random template
    chain_fns = _3HOP_CHAINS if n_hops == 3 else _2HOP_CHAINS
    fn = rng.choice(chain_fns)
    distractor_facts, _, _, _ = fn(rng)
    return distractor_facts


# ─── Instance generation ─────────────────────────────────────────────

def generate_multihop_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1009 + DIFFICULTY_LEVELS.index(difficulty) * 8111)
    config = DIFFICULTY_CONFIG[difficulty]
    n_hops = config["n_hops"]

    # Pick a chain template
    chain_fns = _3HOP_CHAINS if n_hops == 3 else _2HOP_CHAINS
    chain_fn = rng.choice(chain_fns)
    facts, question, gold, pattern = chain_fn(rng)
    # Ensure we have exactly n_hops facts (2-hop chains return 2, 3-hop return 3)
    assert len(facts) == n_hops, f"Chain {pattern} returned {len(facts)} facts for {n_hops}-hop"

    # Generate filler paragraphs
    n_fillers = config["n_fillers"]
    fillers = [generate_filler_paragraph(rng) for _ in range(n_fillers)]

    # Compute fact insertion positions from spread ratios
    spread = config["fact_spread"]
    fact_positions = [max(0, min(n_fillers - 1, int(s * n_fillers))) for s in spread[:n_hops]]
    # Ensure positions are unique and sorted
    fact_positions = sorted(set(fact_positions))
    while len(fact_positions) < n_hops:
        # Add nearby positions if duplicates arose
        for candidate in range(n_fillers):
            if candidate not in fact_positions:
                fact_positions.append(candidate)
                fact_positions = sorted(fact_positions)
                break
            if len(fact_positions) == n_hops:
                break

    # Optionally generate distractor chains (Frontier)
    distractor_facts_flat: List[str] = []
    distractor_positions: List[int] = []
    if config["has_distractors"]:
        # 2 distractor chains
        for _ in range(2):
            d_facts = _generate_distractor_chain(rng, facts, n_hops)
            distractor_facts_flat.extend(d_facts)
        # Place distractor facts at random positions not overlapping real facts
        available = [i for i in range(n_fillers) if i not in fact_positions]
        n_dist = min(len(distractor_facts_flat), len(available))
        distractor_positions = sorted(rng.sample(available, n_dist))

    # Build the document
    # Map position -> sentence to insert after the filler paragraph at that position
    insertions: Dict[int, str] = {}
    for pos, fact in zip(fact_positions, facts):
        insertions[pos] = fact
    for pos, d_fact in zip(distractor_positions, distractor_facts_flat):
        insertions[pos] = d_fact

    doc_parts = []
    for i, filler in enumerate(fillers):
        if i in insertions:
            doc_parts.append(filler + " " + insertions[i])
        else:
            doc_parts.append(filler)

    document = "\n\n".join(doc_parts)
    word_count = len(document.split())

    # Build prompt
    prompt = (
        "Read the following document carefully and answer the question at the end. "
        "The answer requires combining multiple pieces of information scattered "
        "throughout the text.\n\n"
        f"---\n{document}\n---\n\n"
        f"Question: {question}\n\n"
        "Think step by step, then give your final answer.\n"
        "ANSWER: [your answer]"
    )

    return TaskInstance(
        task_id=f"multihop_{difficulty.lower()}_{instance_idx:03d}",
        task_type="multihop",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=gold,
        metadata={
            "n_hops": n_hops,
            "chain_pattern": pattern,
            "facts": facts,
            "fact_positions": fact_positions,
            "n_paragraphs": n_fillers,
            "has_distractors": config["has_distractors"],
            "n_distractor_facts": len(distractor_facts_flat),
            "word_count": word_count,
            "question": question,
        },
    )


def generate_multihop_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_multihop_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_multihop_dataset()
    print(f"Generated {len(dataset)} Multi-hop Scattered Reasoning instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"\n  {diff}: {s.metadata['n_hops']} hops, "
              f"{s.metadata['n_paragraphs']} filler paragraphs, "
              f"~{s.metadata['word_count']} words, "
              f"distractors={s.metadata['has_distractors']}")
        print(f"    Pattern: {s.metadata['chain_pattern']}")
        print(f"    Facts: {s.metadata['facts']}")
        print(f"    Fact positions: {s.metadata['fact_positions']}")
        print(f"    Question: {s.metadata['question']}")
        print(f"    Gold answer: {s.gold_answer}")
