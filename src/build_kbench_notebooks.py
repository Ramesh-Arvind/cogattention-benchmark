"""
Notebook generator for Kaggle Community Benchmarks SDK.

Generates 5 self-contained .ipynb files. Each notebook embeds its dataset (JSON),
inline scoring helpers, task definitions, and execution loop.
No external src/ imports needed on Kaggle.

CLI: python src/build_kbench_notebooks.py --seed 2026 --output-dir notebooks/

Notebook grouping (one task per notebook for Kaggle Benchmarks compatibility):
  task_a1_capacity.ipynb       → capacity (40 items)
  task_a2_interference.ipynb   → interference (40 items)
  task_a3_blink.ipynb          → blink (40 items)
  task_b1a_vigilance.ipynb     → sustained/vigilance (40 items)
  task_b1b_stream.ipynb        → stream_segregation (40 items)
  task_b2_dilution_emh.ipynb   → context_dilution Easy/Med/Hard (24 items)
  task_b2_dilution_expert.ipynb→ context_dilution Expert (8 items)
  task_b2_dilution_frontier_a  → context_dilution Frontier part A (4 items)
  task_b2_dilution_frontier_b  → context_dilution Frontier part B (4 items)
  task_b3_niah.ipynb           → semantic_niah (40 items)
  task_b4_multihop.ipynb       → multihop (40 items)
  task_c1_selective.ipynb      → selective (40 items)
  task_c2_stroop.ipynb         → stroop (40 items)
  task_c3_flanker.ipynb        → flanker (40 items)
  task_d1_shifting.ipynb       → shifting (40 items)
  task_d2_inhibition_return.ipynb → inhibition_return (40 items)
  task_e_anomaly.ipynb         → anomaly (40 items)
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
    # ── Attention Capacity (split: one task per notebook for Kaggle) ──
    {
        "filename": "task_a1_capacity.ipynb",
        "title": "CogAttention — Thread Tracking (Capacity)",
        "cognitive_ability": "Attention Capacity",
        "task_types": ["capacity"],
        "description": (
            "Tests attention capacity through multi-object tracking (Thread Tracking). "
            "N people each hold a unique item and swap in pairwise trades; the model must "
            "report who holds what after all swaps. "
            "Based on Pylyshyn's MOT paradigm (Pylyshyn & Storm, 1988)."
        ),
    },
    {
        "filename": "task_a2_interference.ipynb",
        "title": "CogAttention — Proactive Interference",
        "cognitive_ability": "Attention Capacity",
        "task_types": ["interference"],
        "description": (
            "Tests proactive interference resistance. A series of updates assigns new values "
            "to the same keys repeatedly; the model must report only the final value. "
            "Based on PI-LLM (Wang & Sun, 2025; arXiv:2506.08184)."
        ),
    },
    {
        "filename": "task_a3_blink.ipynb",
        "title": "CogAttention — Attentional Blink",
        "cognitive_ability": "Attention Capacity",
        "task_types": ["blink"],
        "description": (
            "Tests attentional blink — identifying two targets in a rapid serial word stream. "
            "When the second target appears shortly after the first, humans exhibit a temporary "
            "inability to process it. We test whether LLMs show analogous temporal bottlenecks. "
            "Based on RSVP (Raymond et al., 1992)."
        ),
    },
    # ── Sustained Attention (split) ──
    {
        "filename": "task_b1a_vigilance.ipynb",
        "title": "CogAttention — Vigilance Probe",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["sustained"],
        "description": (
            "Tests sustained attention through vigilance probes — detecting targets scattered "
            "across long documents with near-miss distractors. Measures whether detection "
            "drops off later in the document (vigilance decrement). "
            "Based on CPT (Mackworth, 1948)."
        ),
    },
    {
        "filename": "task_b1b_stream.ipynb",
        "title": "CogAttention — Stream Segregation (Cocktail Party)",
        "cognitive_ability": "Sustained Attention",
        "task_types": ["stream_segregation"],
        "description": (
            "Tests stream segregation — tracking one interleaved conversation while ignoring "
            "a distractor stream, with optional breakthrough keyword detection. "
            "Based on Dichotic Listening / Cocktail Party effect (Cherry, 1953)."
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
    # ── Selective Attention (split) ──
    {
        "filename": "task_c1_selective.ipynb",
        "title": "CogAttention — Distractor Filtering",
        "cognitive_ability": "Selective Attention",
        "task_types": ["selective"],
        "description": (
            "Tests selective attention through distractor filtering — extracting verified "
            "signal facts while ignoring unverified noise at varying signal-to-noise ratios. "
            "Based on Signal-in-Noise (SiN) paradigms."
        ),
    },
    {
        "filename": "task_c2_stroop.ipynb",
        "title": "CogAttention — Semantic Stroop",
        "cognitive_ability": "Selective Attention",
        "task_types": ["stroop"],
        "description": (
            "Tests semantic Stroop interference — the model must report what sentences "
            "literally say despite containing factual errors, suppressing the correction reflex. "
            "Based on the classic Stroop task (Stroop, 1935)."
        ),
    },
    {
        "filename": "task_c3_flanker.ipynb",
        "title": "CogAttention — Flanker Interference",
        "cognitive_ability": "Selective Attention",
        "task_types": ["flanker"],
        "description": (
            "Tests flanker interference — extracting a target value surrounded by "
            "conflicting flanker values. "
            "Based on the Eriksen Flanker Task (Eriksen & Eriksen, 1974)."
        ),
    },
    # ── Attention Shifting (split) ──
    {
        "filename": "task_d1_shifting.ipynb",
        "title": "CogAttention — Rule Shift",
        "cognitive_ability": "Attention Shifting",
        "task_types": ["shifting"],
        "description": (
            "Tests attention shifting through rule-switch classification — the model classifies "
            "words by one rule, then must switch to a new rule mid-task. Measures perseveration "
            "errors and attentional residue. "
            "Based on WCST (Monsell, 2003) and Task Interference (EMNLP 2024)."
        ),
    },
    {
        "filename": "task_d2_inhibition_return.ipynb",
        "title": "CogAttention — Inhibition of Return",
        "cognitive_ability": "Attention Shifting",
        "task_types": ["inhibition_return"],
        "description": (
            "Tests inhibition of return — the model answers questions about a passage, then "
            "must answer updated questions about a modified version, suppressing prior answers. "
            "Based on the IOR paradigm (Posner & Cohen, 1984)."
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


TASK_DETAILS = {
    "capacity": {
        "science": (
            "- **Multiple Object Tracking** (Pylyshyn & Storm, 1988): humans can track ~4 "
            "independent objects; we test whether LLMs hit similar capacity walls\n"
            "- Models must maintain distinct identity-item bindings through pairwise swaps"
        ),
        "scaling": (
            "Easy: 2 people, 2 swaps | Medium: 3 people, 5 swaps | Hard: 4 people, 8 swaps | "
            "Expert: 5 people, 12 swaps | Frontier: 8 people, 25 swaps"
        ),
        "scoring": (
            "One assertion per person — checks whether the model correctly reports each "
            "person's final item after all swaps."
        ),
    },
    "interference": {
        "science": (
            "- **Proactive Interference** (Wang & Sun, 2025; arXiv:2506.08184): old memories "
            "inhibit retrieval of new ones\n"
            "- In Transformers, this maps to KV-cache attention sinks anchoring to initial token states\n"
            "- Qwen-72B scores 1.0; Llama-8B collapses to 0.0 at Expert — the largest cross-model gap in CogAttention"
        ),
        "scaling": (
            "Easy: 3 updates, 1 key | Medium: 8 updates, 2 keys | Hard: 15 updates, 3 keys | "
            "Expert: 25 updates, 4 keys | Frontier: 50 updates, 8 keys"
        ),
        "scoring": (
            "One assertion per key — checks whether the model reports the FINAL value, "
            "not any earlier (interfering) value."
        ),
    },
    "blink": {
        "science": (
            "- **Attentional Blink** (Raymond et al., 1992): after detecting a first target "
            "in a rapid stream, humans temporarily cannot process a second target appearing 200-500ms later\n"
            "- We test whether LLMs show analogous temporal bottlenecks in sequential processing"
        ),
        "scaling": (
            "Easy: 20-item stream, lag 8 | Medium: 30 items, lag 5 | Hard: 40 items, lag 3 | "
            "Expert: 60 items, lag 2 | Frontier: 80 items, lag 1"
        ),
        "scoring": (
            "Two assertions per item — one for T1 (ALL-CAPS target), one for T2 (hyphenated "
            "number-word). Both must be identified correctly."
        ),
    },
    "sustained": {
        "science": (
            "- **Continuous Performance Test / Vigilance** (Mackworth, 1948): detection "
            "accuracy degrades over time even when targets remain constant\n"
            "- In LLMs, this manifests as the Lost-in-the-Middle phenomenon — softmax "
            "attention dilution and RoPE decay reduce mid-document target detection"
        ),
        "scaling": (
            "Easy: 10 paragraphs | Medium: 25 paragraphs | Hard: 50 paragraphs | "
            "Expert: 80 paragraphs | Frontier: 150+ paragraphs with subtler targets"
        ),
        "scoring": (
            "One assertion per target item — checks whether each scattered target "
            "was found. Measures vigilance decrement by position."
        ),
    },
    "stream_segregation": {
        "science": (
            "- **Dichotic Listening / Cocktail Party Effect** (Cherry, 1953): attending to "
            "one conversation while ignoring another interleaved stream\n"
            "- At harder levels, the model must also detect a breakthrough keyword in the "
            "ignored stream — testing divided attention under load"
        ),
        "scaling": (
            "Easy: 2 streams, 8 turns each | Medium: 2 streams, 15 turns | "
            "Hard: 2 streams, 20 turns + breakthrough | Expert: 2 streams, 30 turns + breakthrough | "
            "Frontier: 3 interleaved streams, 40 turns + breakthrough"
        ),
        "scoring": (
            "Primary assertion: correct answer from target stream. "
            "Breakthrough assertion (when applicable): detect ALERT keyword in ignored stream."
        ),
    },
    "context_dilution": {
        "science": (
            "- **Context Dilution / Lost in the Middle** (Liu et al., 2024): extracting a "
            "single target fact from progressively longer thematically similar filler\n"
            "- Directly tests softmax attention smoothing — as context grows, attention "
            "weights spread too thin to locate the needle"
        ),
        "scaling": (
            "Easy: ~500 tokens | Medium: ~2K tokens | Hard: ~8K tokens | "
            "Expert: ~20K tokens | Frontier: 50K+ tokens"
        ),
        "scoring": (
            "Single assertion — checks whether the model extracted the target value "
            "despite massive context dilution."
        ),
    },
    "semantic_niah": {
        "science": (
            "- **Semantic Needle-in-a-Haystack** — unlike standard NIAH benchmarks that use "
            "lexically distinctive needles, our variant uses zero keyword overlap between "
            "query and needle\n"
            "- Based on the NoLiMa finding (ICML 2025) that most NIAH benchmarks are solvable "
            "by keyword search — we force genuine semantic comprehension"
        ),
        "scaling": (
            "Easy: ~1K tokens, clear semantic link | Medium: ~4K tokens | Hard: ~10K tokens | "
            "Expert: ~25K tokens | Frontier: ~50K tokens, subtle semantic link"
        ),
        "scoring": (
            "Single assertion — checks whether the model found the semantic needle "
            "without any lexical shortcuts."
        ),
    },
    "multihop": {
        "science": (
            "- **Multi-hop Reasoning over Distributed Context** — facts are scattered across "
            "a long document; the model must chain 2-3 facts to derive the answer\n"
            "- Tests whether sustained attention can support multi-step reasoning — "
            "no single location contains the complete answer"
        ),
        "scaling": (
            "Easy: 2 hops, ~1K tokens | Medium: 2 hops, ~4K tokens | Hard: 2-3 hops, ~10K tokens | "
            "Expert: 3 hops, ~25K tokens | Frontier: 3 hops, ~50K tokens"
        ),
        "scoring": (
            "Single assertion — checks whether the model derived the correct final "
            "answer by chaining distributed facts."
        ),
    },
    "selective": {
        "science": (
            "- **Signal-in-Noise (SiN) Filtering** — extracting verified facts from a mix "
            "of verified (signal) and unverified (noise) sources\n"
            "- Difficulty scales by making source labels less obvious and increasing "
            "noise density, following power-law degradation (GSM-DC, EMNLP 2025)"
        ),
        "scaling": (
            "Easy: explicit source tags, 30% noise | Medium: subtle labels, 50% noise | "
            "Hard: minimal cues, 60% noise | Expert: implicit cues only, 70% noise | "
            "Frontier: zero explicit cues, pure inference required"
        ),
        "scoring": (
            "One assertion per signal value — checks whether the model extracted each "
            "verified fact without including unverified distractors."
        ),
    },
    "stroop": {
        "science": (
            "- **Semantic Stroop Interference** (Stroop, 1935): sentences contain deliberate "
            "factual errors; the model must report what the text literally says, not correct it\n"
            "- Exposes conflict between pre-training priors and in-context instructions — "
            "parameterized weights trigger MLP activations that overpower attention heads\n"
            "- Humans score 0.852 on Stroop resistance; most LLMs score lower"
        ),
        "scaling": (
            "Easy: 2 items, obvious factual errors | Medium: 4 items | Hard: 6 items, "
            "subtle errors | Expert: 8 items | Frontier: 15 items, multi-hop factual traps"
        ),
        "scoring": (
            "One assertion per Stroop item — checks the model reported the literal "
            "(incorrect) statement, not the factually correct answer."
        ),
    },
    "flanker": {
        "science": (
            "- **Eriksen Flanker Task** (Eriksen & Eriksen, 1974): a target value is "
            "surrounded by conflicting flanker values; the model must extract only the target\n"
            "- Tests spatial/sequential inhibition of adjacent distractors"
        ),
        "scaling": (
            "Easy: 1 flanker, clearly marked target | Medium: 2 flankers | "
            "Hard: 3 flankers, subtler marking | Expert: 4 flankers | "
            "Frontier: 6+ flankers, minimal target marking"
        ),
        "scoring": (
            "Single assertion — checks whether the model extracted the correct target "
            "value while ignoring conflicting flankers."
        ),
    },
    "shifting": {
        "science": (
            "- **Wisconsin Card Sorting Test / Task Switching** (Monsell, 2003): classify items "
            "by one rule, then the rule changes mid-task\n"
            "- 60-70% of post-switch errors are perseveration (old rule) or attentional residue "
            "(pre-switch context bleeding in), not random hallucination\n"
            "- Causal self-attention mechanically anchors to earlier context, capping flexibility"
        ),
        "scaling": (
            "Easy: 4 post-switch items, 1 rule change | Medium: 6 items | "
            "Hard: 10 items | Expert: 15 items | Frontier: 20 items, triple rule change"
        ),
        "scoring": (
            "One assertion per classified item — checks correct rule application. "
            "Post-switch errors are classified as perseveration, residue, or random."
        ),
    },
    "inhibition_return": {
        "science": (
            "- **Inhibition of Return** (Posner & Cohen, 1984): after attending to a location, "
            "returning attention there is slower/less accurate\n"
            "- The model answers questions, then sees a modified passage and must suppress "
            "its memory of original answers to attend to changes"
        ),
        "scaling": (
            "Easy: 2 questions, obvious modifications | Medium: 3 questions | "
            "Hard: 4 questions, subtle modifications | Expert: 5 questions | "
            "Frontier: 6+ questions, near-identical passages"
        ),
        "scoring": (
            "One assertion per question — checks the model answered based on the "
            "modified passage, not the original."
        ),
    },
    "anomaly": {
        "science": (
            "- **Inattentional Blindness** (Simons & Chabris, 1999): under cognitive load "
            "(primary counting task), observers fail to notice unexpected stimuli\n"
            "- Phi-3.5 scores 0.16 (near-blind); humans score 0.676 — the largest "
            "human-vs-LLM gap in CogAttention\n"
            "- Fixed attention head count creates a hard capacity constraint: heads allocated "
            "to the primary task leave none for anomaly detection"
        ),
        "scaling": (
            "Easy: obvious anomaly (foreign language block) | Medium: code snippet | "
            "Hard: factual absurdity | Expert: numerical outlier | "
            "Frontier: ultra-subtle (single character swap, off-by-one)"
        ),
        "scoring": (
            "Two assertions — (1) primary counting answer correct, (2) anomaly detected "
            "and described. Both must pass for full credit."
        ),
    },
}


def build_cell_1_markdown(spec: dict, canary: str) -> dict:
    """Cell 1: Markdown header with task-specific methodology and scaling."""
    task_type = spec["task_types"][0]
    details = TASK_DETAILS.get(task_type, {})

    science = details.get("science", "Grounded in established cognitive science paradigms.")
    scaling = details.get("scaling", "5 difficulty levels: Easy, Medium, Hard, Expert, Frontier.")
    scoring = details.get("scoring", "SDK assertion pass rate = per-element accuracy.")

    md = f"""# {spec['title']}

**Track:** Attention — {spec['cognitive_ability']}
**Benchmark:** CogAttention v1.0
**Task:** {', '.join(spec['task_types'])}

---

## Methodology

{spec['description']}

### Cognitive Science Grounding

{science}

### Difficulty Scaling

{scaling}

### Scoring

{scoring}

All instances are procedurally generated from a seed with programmatic ground truth.
No static datasets. 7 layers of contamination resistance including canary strings,
zero lexical overlap (Semantic NIAH), and seed-based regeneration.

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
