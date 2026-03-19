"""
Notebook generator for Kaggle Community Benchmarks SDK.

Generates 5 self-contained .ipynb files. Each notebook embeds its dataset (JSON),
inline scoring helpers, task definitions, and execution loop.
No external src/ imports needed on Kaggle.

CLI: python src/build_kbench_notebooks.py --seed 2026 --output-dir notebooks/

Notebook grouping (5 notebooks matching 5 cognitive abilities):
  task_a_capacity.ipynb   → capacity (40) + interference (40) + blink (40) = 120 items
  task_b_sustained.ipynb  → sustained (40) + stream_segregation (40) + context_dilution (40) + semantic_niah (40) + multihop (40) = 200 items
  task_c_selective.ipynb  → selective (40) + stroop (40) + flanker (40) = 120 items
  task_d_shifting.ipynb   → shifting (40) + inhibition_return (40) = 80 items
  task_e_anomaly.ipynb    → anomaly (40) = 40 items
"""

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_config import get_task_registry
from src.kbench_assertions import (
    build_gold_json,
    INLINE_HELPERS_SOURCE,
    _get_assertion_runner_source,
)


# ── Notebook grouping ─────────────────────────────────────────────────

NOTEBOOK_SPECS = [
    {
        "filename": "task_a_capacity.ipynb",
        "title": "CogAttention — Capacity & Interference",
        "cognitive_ability": "Attention Capacity",
        "task_types": ["capacity", "interference", "blink"],
        "description": (
            "Tests attention capacity through multi-object tracking (Thread Tracking), "
            "proactive interference resistance (Interference Chain), and attentional blink "
            "(rapid serial target detection). "
            "Based on Pylyshyn's MOT paradigm, PI-LLM (Wang & Sun, 2025), and RSVP."
        ),
    },
    {
        "filename": "task_b1_sustained.ipynb",
        "title": "CogAttention — Vigilance & Stream Segregation",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["sustained", "stream_segregation"],
        "description": (
            "Tests sustained attention through vigilance probes (detecting targets scattered "
            "across long documents) and stream segregation (tracking one conversation while "
            "ignoring an interleaved distractor stream). "
            "Based on CPT (Mackworth, 1948) and Dichotic Listening (Cherry, 1953)."
        ),
    },
    {
        "filename": "task_b2_dilution_emh.ipynb",
        "title": "CogAttention — Context Dilution (Easy-Medium-Hard)",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["context_dilution"],
        "difficulty_filter": ["Easy", "Medium", "Hard"],
        "description": (
            "Tests sustained attention under context scaling (short-to-medium contexts). "
            "Based on Context Rot (Chroma 2025)."
        ),
    },
    {
        "filename": "task_b2_dilution_expert.ipynb",
        "title": "CogAttention — Context Dilution (Expert)",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["context_dilution"],
        "difficulty_filter": ["Expert"],
        "description": (
            "Tests sustained attention under long context scaling (Expert difficulty). "
            "Based on Context Rot (Chroma 2025)."
        ),
    },
    {
        "filename": "task_b2_dilution_frontier_a.ipynb",
        "title": "CogAttention — Context Dilution (Frontier Part A)",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["context_dilution"],
        "difficulty_filter": ["Frontier"],
        "max_items": 4,
        "description": (
            "Tests sustained attention under extreme context scaling (Frontier, Part A). "
            "Based on Context Rot (Chroma 2025)."
        ),
    },
    {
        "filename": "task_b2_dilution_frontier_b.ipynb",
        "title": "CogAttention — Context Dilution (Frontier Part B)",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["context_dilution"],
        "difficulty_filter": ["Frontier"],
        "item_offset": 4,
        "max_items": 4,
        "description": (
            "Tests sustained attention under extreme context scaling (Frontier, Part B). "
            "Based on Context Rot (Chroma 2025)."
        ),
    },
    {
        "filename": "task_b3_niah.ipynb",
        "title": "CogAttention — Semantic Needle-in-a-Haystack",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["semantic_niah"],
        "description": (
            "Tests sustained attention through semantic NIAH — the needle has zero lexical "
            "overlap with the question, forcing semantic inference rather than keyword matching. "
            "Based on NoLiMa (ICML 2025)."
        ),
    },
    {
        "filename": "task_b4_multihop.ipynb",
        "title": "CogAttention — Multi-hop Attention",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["multihop"],
        "description": (
            "Tests sustained attention through multi-hop scattered reasoning — the model "
            "must chain facts distributed across a long document to reach the final answer. "
            "Based on BABILong (NeurIPS 2024)."
        ),
    },
    {
        "filename": "task_c_selective.ipynb",
        "title": "CogAttention — Selective & Stroop",
        "cognitive_ability": "Selective Attention",
        "task_types": ["selective", "stroop", "flanker"],
        "description": (
            "Tests selective attention through distractor filtering (signal vs noise), "
            "semantic Stroop interference (suppressing correction reflex), and flanker "
            "interference (extracting target from semantically similar distractors). "
            "Based on Flanker/SiN paradigms and classic Stroop task."
        ),
    },
    {
        "filename": "task_d_shifting.ipynb",
        "title": "CogAttention — Attention Shifting",
        "cognitive_ability": "Attention Shifting",
        "task_types": ["shifting", "inhibition_return"],
        "description": (
            "Tests attention shifting through rule-switch classification (WCST-like) "
            "and inhibition of return (accuracy penalty when returning to previously attended passage). "
            "Measures perseveration errors and return penalties. "
            "Based on WCST, Task Interference (EMNLP 2024), and IOR paradigm."
        ),
    },
    {
        "filename": "task_e_anomaly.ipynb",
        "title": "CogAttention — Stimulus-Driven Attention",
        "cognitive_ability": "Stimulus-Driven Attention",
        "task_types": ["anomaly"],
        "description": (
            "Tests stimulus-driven attention through dual-task anomaly detection. "
            "Model must complete a primary counting task while detecting embedded anomalies. "
            "Based on Inattentional Blindness (Simons & Chabris, 1999)."
        ),
    },
]


# ── Canary generator ──────────────────────────────────────────────────

def make_canary(seed: int, notebook_name: str) -> str:
    """Generate a contamination-detection canary string."""
    h = hashlib.md5(f"{seed}-{notebook_name}".encode()).hexdigest()[:12]
    return f"COGATTENTION-BENCH-CANARY-{h.upper()}"


# ── Dataset generation ────────────────────────────────────────────────

def generate_datasets(seed: int) -> dict:
    """Generate all task instances, grouped by task type."""
    registry = get_task_registry()
    datasets = {}
    for task_type, (gen_fn, _scorer_fn) in registry.items():
        instances = gen_fn(seed=seed)
        datasets[task_type] = instances
    return datasets


def serialize_dataset_for_notebook(instances: list) -> list:
    """Convert TaskInstances into JSON-serializable dicts for embedding."""
    items = []
    for inst in instances:
        items.append({
            "task_id": inst.task_id,
            "task_type": inst.task_type,
            "difficulty": inst.difficulty,
            "prompt": inst.prompt,
            "gold_json": build_gold_json(inst),
        })
    return items


# ── Notebook cell builders ────────────────────────────────────────────

def _make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    }


def _make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": [],
        "execution_count": None,
    }


def build_cell_1_markdown(spec: dict, canary: str) -> dict:
    """Cell 1: Markdown header with benchmark name, cognitive ability, methodology."""
    md = f"""# {spec['title']}

**Track:** Attention — {spec['cognitive_ability']}
**Benchmark:** CogAttention v1.0
**Tasks:** {', '.join(spec['task_types'])}

---

## Methodology

{spec['description']}

### Cognitive Science Grounding

This benchmark is grounded in established cognitive science paradigms:
- **Selective attention** (Cherry, 1953; Broadbent, 1958): filtering relevant from irrelevant stimuli
- **Sustained attention** (Mackworth, 1948): maintaining focus over extended periods
- **Alternating/shifting attention** (Monsell, 2003): switching between task rules
- **Divided attention** (Kahneman, 1973): performing concurrent tasks
- **Attention capacity** (Pylyshyn & Storm, 2001): tracking multiple objects simultaneously

### Difficulty Scaling

Each task uses 5 difficulty levels (Easy, Medium, Hard, Expert, Frontier) with parametric
scaling of cognitive load. Difficulty affects number of items, context length,
distractor density, and cueing clarity. Frontier tier is designed to break frontier models.

### Scoring

SDK assertion pass rate = per-element accuracy. Fine-grained assertions
(one per checkable element) provide continuous scoring rather than binary.

---

`<!-- {canary} -->`
"""
    return _make_markdown_cell(md)


def build_cell_2_imports(spec: dict) -> dict:
    """Cell 2: Imports + inline helpers."""
    task_types = spec["task_types"]
    assertion_runners = _get_assertion_runner_source(task_types)

    source = f"""# ══════════════════════════════════════════════════════════════════════
# Cell 2: Imports + Inline Helpers
# CogAttention — {spec['cognitive_ability']}
# ══════════════════════════════════════════════════════════════════════

import kaggle_benchmarks as kbench
{INLINE_HELPERS_SOURCE}
{assertion_runners}

print("CogAttention helpers loaded")
print(f"Task types: {task_types}")
"""
    return _make_code_cell(source)


def build_cell_3_tasks(spec: dict, all_items: list) -> dict:
    """Cell 3: @kbench.task definitions + embedded DATASET."""
    task_types = spec["task_types"]

    # Build task function definitions
    task_defs = []
    for tt in task_types:
        func_name = f"cogattention_{tt}"
        runner_name = f"run_assertions_{tt}"
        task_defs.append(f'''
@kbench.task(name="{func_name}")
def {func_name}(llm, prompt: str, gold_json: str, task_id: str, difficulty: str):
    """CogAttention {tt} task."""
    response = llm.prompt(prompt)
    gold = json.loads(gold_json)
    {runner_name}(response, gold, kbench)
''')

    task_defs_str = "\n".join(task_defs)

    # Serialize dataset
    dataset_json = json.dumps(all_items, ensure_ascii=False, indent=1)

    source = f"""# ══════════════════════════════════════════════════════════════════════
# Cell 3: Task Definitions + Embedded Dataset
# ══════════════════════════════════════════════════════════════════════

{task_defs_str}

# ── Embedded dataset ──────────────────────────────────────────────────
DATASET = json.loads(r'''
{dataset_json}
''')

print(f"Loaded {{len(DATASET)}} items")
for tt in {task_types!r}:
    count = sum(1 for d in DATASET if d["task_type"] == tt)
    print(f"  {{tt}}: {{count}} items")
"""
    return _make_code_cell(source)


def build_cell_4_execution(spec: dict) -> dict:
    """Cell 4: Execution loop calling task.run() per item."""
    task_types = spec["task_types"]

    # Build dispatcher
    dispatch_lines = []
    for tt in task_types:
        func_name = f"cogattention_{tt}"
        dispatch_lines.append(f'    "{tt}": {func_name},')
    dispatch_str = "\n".join(dispatch_lines)

    source = f"""# ══════════════════════════════════════════════════════════════════════
# Cell 4: Execution Loop
# ══════════════════════════════════════════════════════════════════════

TASK_DISPATCH = {{
{dispatch_str}
}}

n_total = len(DATASET)
for i, item in enumerate(DATASET):
    task_fn = TASK_DISPATCH[item["task_type"]]
    print(f"[{{i+1}}/{{n_total}}] {{item['task_id']}} ({{item['difficulty']}})")
    task_fn.run(
        llm=kbench.llm,
        prompt=item["prompt"],
        gold_json=item["gold_json"],
        task_id=item["task_id"],
        difficulty=item["difficulty"],
    )

print(f"\\nCompleted {{n_total}} items for {spec['cognitive_ability']}")
"""
    return _make_code_cell(source)


# ── Notebook assembly ─────────────────────────────────────────────────

def build_notebook(spec: dict, datasets: dict, seed: int) -> dict:
    """Build a complete .ipynb notebook for a spec."""
    canary = make_canary(seed, spec["filename"])

    # Gather all items for this notebook
    difficulty_filter = spec.get("difficulty_filter", None)
    max_items = spec.get("max_items", None)
    item_offset = spec.get("item_offset", 0)
    all_items = []
    for tt in spec["task_types"]:
        instances = datasets[tt]
        if difficulty_filter:
            instances = [inst for inst in instances if inst.difficulty in difficulty_filter]
        if item_offset:
            instances = instances[item_offset:]
        if max_items:
            instances = instances[:max_items]
        items = serialize_dataset_for_notebook(instances)
        all_items.extend(items)

    # Build cells
    cells = [
        build_cell_1_markdown(spec, canary),
        build_cell_2_imports(spec),
        build_cell_3_tasks(spec, all_items),
        build_cell_4_execution(spec),
    ]

    # Assemble notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }

    return notebook


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate kbench notebooks")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="notebooks/",
        help="Output directory for notebooks",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating datasets with seed={args.seed}...")
    datasets = generate_datasets(args.seed)

    for tt, instances in datasets.items():
        print(f"  {tt}: {len(instances)} instances")

    print(f"\nBuilding {len(NOTEBOOK_SPECS)} notebooks...")

    total_items = 0
    total_assertions_estimate = 0

    for spec in NOTEBOOK_SPECS:
        notebook = build_notebook(spec, datasets, args.seed)
        out_path = output_dir / spec["filename"]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=1)

        # Count items and estimate assertions
        n_items = sum(len(datasets[tt]) for tt in spec["task_types"])
        total_items += n_items

        # Estimate assertion count by running build_gold_json
        n_assertions = 0
        for tt in spec["task_types"]:
            for inst in datasets[tt]:
                gold_json = build_gold_json(inst)
                gold = json.loads(gold_json)
                if tt == "capacity":
                    n_assertions += len(gold["people"])
                elif tt == "interference":
                    n_assertions += len(gold["key_names"])
                elif tt == "blink":
                    n_assertions += 2  # T1 and T2
                elif tt == "sustained":
                    n_assertions += len(gold["targets"])
                elif tt == "stream_segregation":
                    n_assertions += 1
                    if gold["has_breakthrough"]:
                        n_assertions += 1
                elif tt == "selective":
                    n_assertions += len(gold["signals"])
                elif tt == "stroop":
                    n_assertions += len(gold["answers"])
                elif tt == "flanker":
                    n_assertions += 1  # single value check
                elif tt == "context_dilution":
                    n_assertions += 1
                elif tt == "semantic_niah":
                    n_assertions += 1
                elif tt == "multihop":
                    n_assertions += 1
                elif tt == "shifting":
                    n_assertions += len(gold["answers"])
                elif tt == "inhibition_return":
                    n_assertions += len(gold["answers"])
                elif tt == "anomaly":
                    n_assertions += 2

        total_assertions_estimate += n_assertions
        print(f"  {spec['filename']}: {n_items} items, ~{n_assertions} assertions")
        print(f"    → {out_path}")

    print(f"\nTotal: {total_items} items, ~{total_assertions_estimate} assertions")
    print("Done!")


if __name__ == "__main__":
    main()
