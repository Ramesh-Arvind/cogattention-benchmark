# Attention Benchmark — Task Specifications

**Benchmark Name**: CogAttention (Cognitive Attention Benchmark)
**Track**: Attention (DeepMind Cognitive Framework §7.3)
**Version**: 2.0

---

## Overview

CogAttention tests 5 attention sub-abilities with 16 task types across 5 difficulty levels (Easy, Medium, Hard, Expert, Frontier).
All tasks are procedurally generated, contamination-resistant, and produce unambiguous ground truth.

| Task | Sub-ability | Paradigm Source | Items |
|------|------------|-----------------|-------|
| A: Thread Tracking | Capacity (7.3.1) | MOT (Pylyshyn & Storm, 1988) | 40 |
| A: Proactive Interference | Capacity (7.3.1) | PI-LLM (Wang & Sun, 2025) | 40 |
| A: Attentional Blink | Capacity (7.3.1) | RSVP (Raymond et al., 1992) | 40 |
| B: Vigilance Probe | Sustained (7.3.2) | CPT (Mackworth, 1948) | 40 |
| B: Stream Segregation | Sustained (7.3.2) | Dichotic Listening (Cherry, 1953) | 40 |
| B: Context Dilution | Sustained (7.3.2) | Lost in the Middle (Liu et al., 2024) | 40 |
| B: Semantic NIAH | Sustained (7.3.2) | NoLiMa | 40 |
| B: Multi-hop Attention | Sustained (7.3.2) | BABILong | 40 |
| C: Distractor Filtering | Selective (7.3.2) | SiN enhanced | 40 |
| C: Semantic Stroop | Selective (7.3.2) | Stroop (1935) | 40 |
| C: Flanker Interference | Selective (7.3.2) | Eriksen Flanker (1974) | 40 |
| D: Rule Shift | Shifting (7.3.2) | WCST (Monsell, 2003) | 40 |
| D: Inhibition of Return | Shifting (7.3.2) | IOR (Posner & Cohen, 1984) | 40 |
| E: Anomaly Detection | Stimulus-Driven (7.3.3) | Inattentional Blindness (Simons & Chabris, 1999) | 40 |
| F: Visual Stroop | Selective (VLM) | Visual Stroop (Pillow-generated) | 150 |
| G: Visual Inattentional | Stimulus-Driven (VLM) | Gorilla Experiment (Simons & Chabris, 1999) | 150 |
| **Total** | | | **860** |

Text tasks: 8 items per difficulty × 5 difficulties = 40 items each.
Visual tasks: 30 items per difficulty × 5 difficulties = 150 items each.

**Kaggle Benchmark**: 7 core tasks (blink, inhibition_return, flanker, context_dilution, semantic_niah, multihop, visual_stroop).
**GitHub**: Full suite of 13 notebooks covering all 16 task types.

**Scoring**: Arithmetic CAS + Geometric CAS, bootstrapped 95% CIs (10,000 resamples), Cohen's d effect sizes, power-law degradation coefficient, 2PL IRT analysis, position bias detection. See `docs/metrics.md`.

---

## Difficulty Scaling

All tasks scale across 5 tiers with concrete parameter changes:

| Level | Description |
|-------|-------------|
| Easy | Minimal load, explicit cues, short context |
| Medium | Moderate load, inline cues, medium context |
| Hard | High load, subtle cues, long context |
| Expert | Very high load, minimal cues, very long context |
| Frontier | Maximum load, designed to push frontier models to failure |

---

## Task A: Thread Tracking (Attention Capacity)

### Cognitive Construct
Tests how many concurrent items a system can track through transformations.
Maps to §7.3.1: "The amount of information a system can focus on simultaneously."

### Task Description
N characters each hold a unique item. A series of pairwise swaps occur.
The model must report who holds what after all swaps.

### Difficulty Scaling
| Level | N (people) | Swaps |
|-------|-----------|-------|
| Easy | 2 | 2 |
| Medium | 3 | 4 |
| Hard | 4 | 6 |
| Expert | 5 | 10 |
| Frontier | 5 | 12 |

### Ground Truth
Programmatically computed by replaying the swap sequence.

### Scoring
- Per-person exact match via `assert_contains_regex`
- Aggregate assertion: passes if ≥ (N-1) correct when N ≥ 5

---

## Task A: Proactive Interference (Attention Capacity)

### Cognitive Construct
Tests resistance to retroactive interference from prior values.

### Task Description
A series of updates assigns new values to the same keys repeatedly.
The model must report only the final value for each key.

### Difficulty Scaling
| Level | Keys | Updates |
|-------|------|---------|
| Easy | 2 | 4 |
| Medium | 3 | 8 |
| Hard | 3 | 15 |
| Expert | 4 | 25 |
| Frontier | 5 | 30 |

### Scoring
- Aggregate assertion: passes if ≥ (N-1) correct when N ≥ 4

---

## Task A: Attentional Blink (Attention Capacity)

### Cognitive Construct
Tests temporal bottleneck in sequential processing.
Adapts Raymond et al. (1992) RSVP paradigm.

### Task Description
A rapid serial stream of words with two embedded targets (T1 and T2).
The model must identify both targets.

### Scoring
- Aggregate assertion: passes if ≥ 1/2 targets found

---

## Task B: Vigilance Probe (Sustained Attention)

### Cognitive Construct
Tests whether attention degrades over long contexts (Mackworth, 1948).

### Task Description
A long document contains scattered targets from a specific category (e.g., metals).
Near-miss distractors from related categories are present.

### Difficulty Scaling
| Level | Doc Length | Targets | Distractors |
|-------|-----------|---------|-------------|
| Easy | ~2000 words | 3 | 2 |
| Medium | ~4000 words | 5 | 4 |
| Hard | ~6000 words | 7 | 6 |
| Expert | ~8000 words | 8 | 8 |
| Frontier | ~10000 words | 10 | 10 |

### Scoring
- Aggregate assertion: passes if ≥ (N-1) targets found when N ≥ 8

---

## Task B: Stream Segregation (Sustained Attention)

### Cognitive Construct
Tests ability to follow one information stream while ignoring another (Cherry, 1953).

### Task Description
Two conversations interleaved sentence by sentence, marked [A] and [B].
Model answers about stream A and detects breakthrough keyword in stream B.

### Difficulty Scaling
| Level | Sentences/stream | Breakthrough |
|-------|-----------------|-------------|
| Easy | 4 | No |
| Medium | 6 | No |
| Hard | 6 | Yes |
| Expert | 8 | Yes |
| Frontier | 14 (3 streams) | Yes |

---

## Task B: Context Dilution (Sustained Attention)

### Cognitive Construct
Tests target extraction from progressively longer passages (Lost in the Middle, Liu et al. 2024).

### Difficulty Scaling
Passage length scales from ~2K tokens (Easy) to 50K+ tokens (Frontier).

### Scoring
- Single assertion: must find the target value

---

## Task B: Semantic NIAH (Sustained Attention)

### Cognitive Construct
Needle-in-a-haystack with zero lexical overlap between query and needle (NoLiMa).

### Scoring
- Single assertion: must find the needle through meaning alone

---

## Task B: Multi-hop Attention (Sustained Attention)

### Cognitive Construct
Tests chaining 2-3 distributed facts to arrive at an answer.

### Scoring
- Single assertion: must find the chained answer

---

## Task C: Distractor Filtering (Selective Attention)

### Cognitive Construct
Tests signal extraction while ignoring noise (Perceptual Inhibition, §7.3.2).

### Difficulty Scaling
| Level | Signals | Distractors | Labeling |
|-------|---------|-------------|----------|
| Easy | 3 | 3 | Explicit tags |
| Medium | 4 | 6 | Inline attribution |
| Hard | 5 | 10 | Subtle cues |
| Expert | 6 | 15 | Minimal cues |
| Frontier | 6 | 20 | No explicit cues |

### Scoring
- Aggregate assertion: passes if ≥ (N-1) signals found when N ≥ 5

---

## Task C: Semantic Stroop (Selective Attention)

### Cognitive Construct
Tests suppression of automatic semantic processing (Stroop, 1935).

### Task Description
Sentences contain factual errors. Model must report what the sentence literally says, not what is correct.

### Scoring
- Aggregate assertion: passes if ≥ (N-1) correct when N ≥ 8

---

## Task C: Flanker Interference (Selective Attention)

### Cognitive Construct
Tests target extraction amid conflicting flankers (Eriksen & Eriksen, 1974).

### Scoring
- Single assertion: must extract target value

---

## Task D: Rule Shift (Attention Shifting)

### Cognitive Construct
Tests ability to switch classification rules mid-task (WCST, Monsell 2003).

### Difficulty Scaling
| Level | Pre-switch items | Post-switch items | Rule similarity |
|-------|-----------------|-------------------|----------------|
| Easy | 5 | 5 | Low |
| Medium | 5 | 5 | Medium |
| Hard | 5 | 5 | High |
| Expert | 5 | 5 | Very high |
| Frontier | 5+5+5+5+4 | 2 switches | Mixed |

### Scoring
- Aggregate assertion: passes if ≥ (N-1) correct when N ≥ 15

---

## Task D: Inhibition of Return (Attention Shifting)

### Cognitive Construct
Tests suppression of prior answers when passage is modified (Posner & Cohen, 1984).

### Scoring
- Aggregate assertion: passes if ≥ (N-1) correct when N ≥ 8

---

## Task E: Anomaly Detection (Stimulus-Driven Attention)

### Cognitive Construct
Tests noticing unexpected items during routine tasks (Simons & Chabris, 1999).

### Anomaly Types
| Type | Saliency |
|------|----------|
| Language switch (French/Spanish/German) | High |
| Code block (Python/SQL/C) | High |
| Factual absurdity | Medium |
| Numerical anomaly (coffee invoice for $12M) | Medium |
| Name inconsistency | Low |
| Single character swap (typo) | Ultra-low |
| Off-by-one math error | Ultra-low |
| Date inconsistency | Ultra-low |

### Scoring
- Single assertion: must detect the anomaly using keyword matching
- Primary counting task is NOT scored (cover task only)

---

## Task F: Visual Stroop (Multimodal — VLM)

### Cognitive Construct
Direct visual implementation of Stroop Effect. Procedurally generated images.

### Task Description
Color words printed in conflicting ink colors. Model must name the ink color.

### Difficulty Scaling
| Level | Items | Font size | Noise | Distractors |
|-------|-------|-----------|-------|-------------|
| Easy | 1 | 60px | No | 0 |
| Medium | 2 | 48px | No | 2 |
| Hard | 3 | 36px | Yes | 4 |
| Expert | 4 | 28px | Yes | 6 |
| Frontier | 5 | 24px | Yes | 8 |

### Scoring
- Color-variant-aware matching (navy→blue, crimson→red)
- Aggregate assertion: passes if ≥ (N-1) correct when N ≥ 3

---

## Task G: Visual Inattentional Blindness (Multimodal — VLM)

### Cognitive Construct
Adapts Simons & Chabris (1999) gorilla experiment to static visual scenes.

### Task Description
Busy scene with shapes. Primary task: count target shapes. Unexpected stimulus embedded.

### Scoring
- Count assertion: ±1 tolerance
- Detection assertion: keyword matching with expanded vocabulary

---

## Composite Metric: Cognitive Attention Score (CAS)

```
Weights:
  capacity:           0.07
  interference:       0.07
  blink:              0.03
  sustained:          0.07
  stream_segregation: 0.07
  context_dilution:   0.08
  semantic_niah:      0.08
  multihop:           0.08
  selective:          0.07
  stroop:             0.07
  flanker:            0.04
  shifting:           0.07
  inhibition_return:  0.07
  anomaly:            0.13
```

Two modes:
- **Arithmetic CAS**: Weighted mean (compensatory)
- **Geometric CAS**: Weighted geometric mean (non-compensatory — zero on any task tanks composite)

### Statistical Analysis
- Bootstrapped 95% CIs (10,000 resamples)
- Cohen's d effect sizes between models
- Power-law degradation coefficient (δ) for selective attention
- 2PL IRT item parameters (difficulty + discrimination)
- Position bias U-curve analysis
- Attentional residue error classification for shifting
