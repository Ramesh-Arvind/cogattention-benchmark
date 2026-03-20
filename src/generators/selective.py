"""
Task C: Distractor Filtering (Selective Attention)

Tests ability to extract relevant info while ignoring semantically similar distractors.
Cognitive construct: §7.3.2 Perceptual Inhibition
Paradigm source: GSM-DC power-law + Flanker + SiN

Source A facts (signal) are interleaved with Source B facts (distractors).
Model must report ONLY Source A values.
"""

import random
from typing import List, Dict
from .base import TaskInstance, DIFFICULTY_LEVELS, FIRST_NAMES, CITIES

DIFFICULTY_CONFIG = {
    "Easy":   {"n_signal": 3,  "n_distractor": 3,  "labeling": "explicit"},
    "Medium": {"n_signal": 4,  "n_distractor": 6,  "labeling": "inline"},
    "Hard":   {"n_signal": 5,  "n_distractor": 10, "labeling": "subtle"},
    "Expert":   {"n_signal": 6,  "n_distractor": 15, "labeling": "minimal"},
    "Frontier": {"n_signal": 6,  "n_distractor": 20, "labeling": "none"},
}

ITEMS_PER_DIFFICULTY = 8

# Domains for procedural fact generation
DOMAINS = {
    "financial": {
        "unit": "$",
        "suffix": "",
        "template_signal": [
            "[ATTR] The verified audit found revenue of {unit}{value}{suffix}.",
            "[ATTR] According to the certified report, expenses totaled {unit}{value}{suffix}.",
            "[ATTR] The validated balance sheet shows assets of {unit}{value}{suffix}.",
            "[ATTR] Confirmed quarterly earnings reached {unit}{value}{suffix}.",
        ],
        "template_distractor": [
            "[ATTR] Preliminary estimates suggest costs near {unit}{value}{suffix}.",
            "[ATTR] Unconfirmed sources report gains of {unit}{value}{suffix}.",
            "[ATTR] An unaudited draft mentions liabilities of {unit}{value}{suffix}.",
            "[ATTR] Speculation puts the figure at {unit}{value}{suffix}.",
        ],
        "value_range": (10, 9999),
        "value_fmt": lambda v: f"{v:,.2f}",
    },
    "weather": {
        "unit": "",
        "suffix": "°C",
        "template_signal": [
            "[ATTR] The calibrated station recorded a temperature of {value}{suffix}.",
            "[ATTR] Official meteorological data shows {value}{suffix} at noon.",
            "[ATTR] The verified sensor reading was {value}{suffix}.",
            "[ATTR] According to the certified gauge, pressure-adjusted temperature was {value}{suffix}.",
        ],
        "template_distractor": [
            "[ATTR] A passerby estimated the temperature at {value}{suffix}.",
            "[ATTR] An uncalibrated thermometer showed {value}{suffix}.",
            "[ATTR] Social media posts claimed it felt like {value}{suffix}.",
            "[ATTR] An unverified weather app displayed {value}{suffix}.",
        ],
        "value_range": (-15, 42),
        "value_fmt": lambda v: f"{v:.1f}",
    },
    "lab_results": {
        "unit": "",
        "suffix": " mg/L",
        "template_signal": [
            "[ATTR] The accredited lab measured concentration at {value}{suffix}.",
            "[ATTR] Certified analysis confirmed {value}{suffix} of the compound.",
            "[ATTR] The peer-reviewed measurement yielded {value}{suffix}.",
            "[ATTR] Quality-controlled testing showed {value}{suffix}.",
        ],
        "template_distractor": [
            "[ATTR] A field test kit indicated approximately {value}{suffix}.",
            "[ATTR] Preliminary screening suggested {value}{suffix}.",
            "[ATTR] An uncertified instrument read {value}{suffix}.",
            "[ATTR] The expired reagent test showed {value}{suffix}.",
        ],
        "value_range": (1, 500),
        "value_fmt": lambda v: f"{v:.1f}",
    },
}

# Attribution styles by difficulty
def _make_attribution(labeling: str, is_signal: bool, rng: random.Random) -> str:
    source = "Source A (verified)" if is_signal else "Source B (unverified)"
    if labeling == "explicit":
        return f"[{source}]"
    elif labeling == "inline":
        if is_signal:
            return rng.choice([
                "According to Source A,",
                "Source A reports that",
                "The verified source states",
            ])
        else:
            return rng.choice([
                "According to Source B,",
                "Source B suggests that",
                "The unverified source claims",
            ])
    elif labeling == "subtle":
        if is_signal:
            return rng.choice([
                "The audited record indicates",
                "Per the verified data,",
                "Cross-referenced documentation shows",
            ])
        else:
            return rng.choice([
                "Anecdotal evidence suggests",
                "Unofficial channels report",
                "Unconfirmed data indicates",
            ])
    elif labeling == "minimal":
        if is_signal:
            return rng.choice([
                "Records show",
                "The data indicates",
                "Documentation confirms",
            ])
        else:
            return rng.choice([
                "It has been suggested that",
                "Some estimates place",
                "Reportedly,",
            ])
    else:  # none — zero source cues, identical phrasing
        return rng.choice([
            "The report states",
            "According to the data,",
            "The findings indicate",
            "Analysis shows",
        ])


def generate_selective_instance(
    instance_idx: int,
    difficulty: str,
    seed: int = 2026,
) -> TaskInstance:
    rng = random.Random(seed + instance_idx * 1013 + DIFFICULTY_LEVELS.index(difficulty) * 8191)
    config = DIFFICULTY_CONFIG[difficulty]

    # Pick domain
    domain_name = rng.choice(list(DOMAINS.keys()))
    domain = DOMAINS[domain_name]

    # Generate signal values
    signal_values = []
    for _ in range(config["n_signal"]):
        raw = rng.uniform(*domain["value_range"])
        signal_values.append(domain["value_fmt"](raw))

    # Generate distractor values (guaranteed different from signals)
    signal_set = set(signal_values)
    distractor_values = []
    for _ in range(config["n_distractor"]):
        while True:
            raw = rng.uniform(*domain["value_range"])
            val = domain["value_fmt"](raw)
            if val not in signal_set and val not in distractor_values:
                break
        distractor_values.append(val)

    # Build sentences
    signal_sentences = []
    for val in signal_values:
        tmpl = rng.choice(domain["template_signal"])
        attr = _make_attribution(config["labeling"], True, rng)
        sent = tmpl.format(
            value=val, unit=domain["unit"], suffix=domain["suffix"]
        ).replace("[ATTR]", attr)
        signal_sentences.append(sent)

    distractor_sentences = []
    for val in distractor_values:
        tmpl = rng.choice(domain["template_distractor"])
        attr = _make_attribution(config["labeling"], False, rng)
        sent = tmpl.format(
            value=val, unit=domain["unit"], suffix=domain["suffix"]
        ).replace("[ATTR]", attr)
        distractor_sentences.append(sent)

    # Interleave randomly
    all_sentences = [(s, "signal", v) for s, v in zip(signal_sentences, signal_values)]
    all_sentences += [(s, "distractor", v) for s, v in zip(distractor_sentences, distractor_values)]
    rng.shuffle(all_sentences)

    passage = "\n".join(s for s, _, _ in all_sentences)

    # Build prompt
    source_desc = {
        "explicit": "Statements are tagged with [Source A (verified)] or [Source B (unverified)].",
        "inline": "Statements mention their source inline (Source A is verified, Source B is unverified).",
        "subtle": "Verified data comes from audited/cross-referenced sources. Unverified data comes from unofficial/anecdotal sources.",
        "minimal": "You must determine which statements come from reliable documentation vs. unconfirmed reports.",
        "none": "All statements use identical phrasing. Use the context and precision of the values to determine which are from the verified source.",
    }

    prompt = (
        f"The following report contains statements from two sources:\n"
        f"- Source A (verified): these are the facts you need.\n"
        f"- Source B (unverified): ignore these completely.\n\n"
        f"{source_desc[config['labeling']]}\n\n"
        f"Extract ONLY the numerical values reported by the verified source (Source A).\n\n"
        f"---\n{passage}\n---\n\n"
        f"ANSWER: [value1], [value2], [value3], ..."
    )

    return TaskInstance(
        task_id=f"selective_{difficulty.lower()}_{instance_idx:03d}",
        task_type="selective",
        difficulty=difficulty,
        prompt=prompt,
        gold_answer=signal_values,
        metadata={
            "domain": domain_name,
            "labeling": config["labeling"],
            "signal_values": signal_values,
            "distractor_values": distractor_values,
            "n_signal": config["n_signal"],
            "n_distractor": config["n_distractor"],
            "noise_ratio": round(config["n_distractor"] / (config["n_signal"] + config["n_distractor"]), 3),
        },
    )


def generate_selective_dataset(seed: int = 2026) -> List[TaskInstance]:
    instances = []
    for diff in DIFFICULTY_LEVELS:
        for i in range(ITEMS_PER_DIFFICULTY):
            inst = generate_selective_instance(len(instances), diff, seed)
            instances.append(inst)
    return instances


if __name__ == "__main__":
    dataset = generate_selective_dataset()
    print(f"Generated {len(dataset)} Distractor Filtering instances")
    for diff in DIFFICULTY_LEVELS:
        subset = [d for d in dataset if d.difficulty == diff]
        s = subset[0]
        print(f"  {diff}: {s.metadata['n_signal']} signals, "
              f"{s.metadata['n_distractor']} distractors, "
              f"noise_ratio={s.metadata['noise_ratio']}, "
              f"labeling={s.metadata['labeling']}")
        print(f"    Signal values: {s.gold_answer}")
