# CogAttention: Benchmark Description

CogAttention measures attention in frontier language models as the DeepMind cognitive framework (Burnell et al., 2026, §7.3) defines it: three separable faculties that decompose into five sub-abilities, tested through 16 procedurally generated task types and 860 items. Unlike retrieval-focused long-context tests (NIAH, RULER), CogAttention operationalizes classical attention paradigms from cognitive psychology as behavioral tasks with programmatic ground truth.

## 16 task types across 5 attention sub-abilities (3-tier hierarchy)

### Attention Capacity (§7.3.1)
- **Thread Tracking** — N agents swap items; report final holdings (Pylyshyn & Storm, 1988)
- **Proactive Interference** — repeated value updates; report only the final value (Wang & Sun, 2025)
- **Attentional Blink** — identify two targets in rapid serial presentation (Raymond et al., 1992)

### Selective Attention / Attentional Control (§7.3.2)

**Sustained focus:**
- **Vigilance Probe** (Mackworth, 1948)
- **Stream Segregation** (Cherry, 1953)
- **Context Dilution** (50K+ tokens)
- **Semantic NIAH** (zero lexical overlap)
- **Multihop** (chained facts across distributed context)

**Perceptual Inhibition:**
- **Distractor Filtering** (verified vs. unverified facts)
- **Semantic Stroop** (Stroop, 1935)
- **Flanker** (Eriksen & Eriksen, 1974)

**Shifting:**
- **Rule Shift** (Monsell, 2003)
- **Inhibition of Return** (Posner & Cohen, 1984)

### Stimulus-Driven Attention (§7.3.3)
- **Anomaly Detection** — embedded foreign language, code, or factual absurdity must be flagged during a distractor task (Simons & Chabris, 1999)

### Multimodal extensions (VLM)
- **Visual Stroop** — name ink color of a color word in conflicting ink
- **Visual Inattentional Blindness** — count shapes while detecting an unexpected stimulus

## Dataset and contamination resistance

All items are procedurally generated from a seed with programmatic ground truth — no static datasets, no memorization shortcuts. Contamination resistance via combinatorial pools, canary strings, seed-based regeneration, zero lexical overlap, and private scoring logic held out of model contexts. Difficulty spans five tiers (Easy → Frontier) by varying tracked-item count, passage length, distractor density, and cue clarity.

## Scoring

Composite Attention Score (CAS) reported in two modes per BetterBench (Reuel et al., 2024) — **Arithmetic CAS** (compensatory) and **Geometric CAS** (non-compensatory, penalizing uneven profiles). Statistical analysis: bootstrapped 95% CIs (10,000 resamples), Cohen's d, and Classical Test Theory point-biserial discrimination (`r_pb`) against total CAS.

## What to use CogAttention for

- **Rank frontier models on attention-specific failure modes** that retrieval-style long-context suites miss (perseveration, distractor intrusion, vigilance decrement, inattentional blindness).
- **Diagnose where a model breaks** — Geometric CAS + per-ability breakdown exposes which of the five sub-abilities collapse, instead of a single averaged number.
- **Evaluate open models locally** — seeded procedural generation + vLLM harness with OOM resilience and checkpoint/resume reproduce the 860-item suite on commodity GPUs.
- **Extend to new cognitive paradigms** — every task is a `generator.py` + `scorer.py` pair with programmatic ground truth; forks inherit the CAS aggregation, CIs, and discrimination analysis for free.
- **Stress-test VLMs on selective and stimulus-driven attention** — Visual Stroop and Visual Inattentional Blindness currently floor all 7 frontier models (0/7 pass), giving a clean signal for the next generation.
- **Track progress toward AGI on the Burnell et al. (2026) cognitive framework** — CogAttention is the attention pillar; combine with Executive Functions (Agrawal) and other cognitive suites for a composite profile.

## Results (7 frontier models, April 2026)

21-point CAS spread (DeepSeek-R1 and Claude Opus 4.6 at 0.89; Gemma-3-27B at 0.68). Four discriminating tasks under `r_pb >= 0.3`:

| Task | r_pb | Failure rate |
|------|------|--------------|
| Anomaly Detection | 0.94 | 2/7 |
| Rule Shift | 0.77 | 3/7 |
| Capacity (Thread Tracking) | 0.75 | 1/7 |
| Attentional Blink | 0.68 | 4/7 |

Ten remaining text-only tasks at ceiling; two visual tasks at floor (all 7 models fail).
