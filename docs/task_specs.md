# Attention Benchmark — Task Specifications

**Benchmark Name**: CogAttention (Cognitive Attention Benchmark)
**Track**: Attention (DeepMind Cognitive Framework §7.3)
**Version**: 1.0

---

## Overview

CogAttention tests 5 attention sub-abilities with 15 task types across 5 difficulty levels.
All tasks are procedurally generated, contamination-resistant, and produce unambiguous ground truth.

| Task | Sub-ability | Paper Section | Paradigm Source | Items |
|------|------------|---------------|-----------------|-------|
| A: Thread Tracking | Capacity (7.3.1) | Attention Capacity | MOT (Pylyshyn) | 40 |
| A: Proactive Interference | Capacity (7.3.1) | Attention Capacity | PI-LLM (Wang & Sun) | 40 |
| A: Attentional Blink | Capacity (7.3.1) | Temporal Capacity | RSVP | 40 |
| B: Vigilance Probe | Sustained (7.3.2) | Sustained Attention | CPT (Mackworth) | 40 |
| B: Stream Segregation | Sustained (7.3.2) | Sustained Attention | Dichotic Listening (Cherry) | 40 |
| B: Context Dilution | Sustained (7.3.2) | Sustained Attention | Length scaling | 40 |
| B: Semantic NIAH | Sustained (7.3.2) | Sustained Attention | NoLiMa | 40 |
| B: Multi-hop Attention | Sustained (7.3.2) | Sustained Attention | BABILong | 40 |
| C: Distractor Filtering | Selective (7.3.2) | Perceptual Inhibition | SiN enhanced | 40 |
| C: Semantic Stroop | Selective (7.3.2) | Perceptual Inhibition | Stroop (1935) | 40 |
| C: Flanker Interference | Selective (7.3.2) | Perceptual Inhibition | Eriksen Flanker | 40 |
| D: Rule Shift | Shifting (7.3.2) | Attention Shifting | WCST (Monsell) | 40 |
| D: Inhibition of Return | Shifting (7.3.2) | Attention Shifting | IOR | 40 |
| E: Anomaly Detection | Stimulus-Driven (7.3.3) | Stimulus-Driven | Inattentional Blindness (Simons) | 40 |
| F: Visual Stroop | Selective (VLM) | Multi-sensory | Visual Stroop (Pillow-generated) | 40 |
| **Total** | | | | **600** |

Each task: 8 Easy + 8 Medium + 8 Hard + 8 Expert + 8 Frontier = 40 items.

**Scoring upgrades**: Geometric CAS, attentional residue classification, bootstrapped CIs, power-law degradation, IRT analysis, position bias detection. See `docs/metrics.md`.

---

## Task A: Thread Tracking (Attention Capacity)

### Cognitive Construct
Tests how many concurrent items a system can track through transformations.
Maps to §7.3.1: "The amount of information a system can focus on simultaneously."

### Task Description
N characters each hold a unique item. A series of pairwise swaps occur.
The model must report who holds what after all swaps.

### Prompt Template
```
You are given {N} people, each holding a unique item. After a series of swaps,
report who holds each item.

Starting positions:
- Alice holds a red key
- Bob holds a blue book
- Carol holds a green coin
{... more for higher N}

Swaps:
1. Alice and Bob swap items.
2. Bob and Carol swap items.
3. Alice and Carol swap items.
{... more swaps for higher difficulty}

After all swaps, list each person and their current item.
Format your answer as:
ANSWER:
- [Person]: [item]
- [Person]: [item]
...
```

### Difficulty Scaling
| Level | N (people) | Swaps | Expected Accuracy |
|-------|-----------|-------|-------------------|
| Easy | 2 | 2 | 90%+ |
| Medium | 3 | 4 | 60-80% |
| Hard | 4 | 6 | 30-60% |
| Expert | 5 | 10 | <30% |

### Ground Truth
Programmatically computed by replaying the swap sequence.

### Scoring
- Per-person exact match: does the model assign the correct item to each person?
- Score = (correct assignments) / N
- kbench assertion: `assert_contains_regex` for each person-item pair

### Procedural Generation Parameters
- Person names: pool of 50+ diverse names, sampled without replacement
- Item descriptors: [color] + [object], both from pools of 20+ entries
- Swap pairs: random, no consecutive identical swaps
- Seed: deterministic per instance ID

### Edge Cases
- Model lists people not in the prompt
- Model omits some people
- Model describes items differently ("red key" → "crimson key")
- Model adds reasoning before ANSWER

### Assertion Regex Pattern
```python
# For each (person, item) in ground truth:
pattern = rf"(?i){person}.*?{color}.*?{object}"
# OR
pattern = rf"(?i){person}[:\s]+{color}\s+{object}"
```

---

## Task B: Vigilance Probe (Sustained Attention)

### Cognitive Construct
Tests whether attention degrades over long contexts.
Maps to §7.3.2 Sustained Attention: "maintain focus on goal-relevant information over time."

### Task Description
A long document contains a specific instruction to find all instances of a target pattern.
Targets are placed at known positions (early, middle, late). Distractors are semantically
similar but do NOT match the target rule.

### Prompt Template
```
Read the entire document below carefully. Find ALL sentences that mention
a {target_category} (e.g., "a type of bird"). List every instance you find.

There may be similar-sounding items that are NOT {target_category} — do not include those.

---
{document: 2000-15000 words with targets at controlled positions}
---

List ALL {target_category} items mentioned in the document, in the order they appear.
Format: ANSWER: [item1], [item2], [item3], ...
```

### Difficulty Scaling
| Level | Doc Length | # Targets | # Near-Miss Distractors | Target Frequency |
|-------|-----------|-----------|------------------------|-----------------|
| Easy | ~2000 words | 5 | 2 | 1 per 400 words |
| Medium | ~5000 words | 8 | 5 | 1 per 625 words |
| Hard | ~10000 words | 10 | 10 | 1 per 1000 words |
| Expert | ~15000 words | 12 | 15 | 1 per 1250 words |

### Ground Truth
All target items and their exact positions are known at generation time.

### Scoring
- **Recall**: (targets found) / (total targets)
- **Precision**: (true targets found) / (total items reported)
- **Vigilance Decrement**: Compare recall for targets in first-third vs last-third of document
- **Intrusion Rate**: (near-miss distractors incorrectly included) / (total near-misses)
- kbench assertion: `assert_contains_regex` for each target item

### Key Design Principles (from literature)
- **Zero lexical overlap** between query term and target instances (NoLiMa lesson)
- **Structured narrative** haystack, not random text (Context Rot lesson — harder)
- **Near-miss distractors** in same semantic field (e.g., target="birds" → distractors include "bats", "butterflies")
- Target positions are **uniformly distributed** (avoid U-shape exploitation)

### Procedural Generation
- Document: procedurally generated narrative about a fictional town/market/expedition
- Target category: randomly chosen (birds, metals, musical instruments, rivers, etc.)
- Target instances: real members of category, embedded in natural sentences
- Near-misses: items that share features but are NOT members (e.g., "penguin" for flying birds)
- Filler text: topically diverse paragraphs (weather, commerce, characters, landscape)

### Vigilance Decrement Measurement
```python
# Divide document into 5 quintiles by word position
# Compute recall per quintile
# Report: [Q1_recall, Q2_recall, Q3_recall, Q4_recall, Q5_recall]
# Vigilance decrement = Q1_recall - Q5_recall (positive = decrement exists)
```

---

## Task C: Distractor Filtering (Selective Attention)

### Cognitive Construct
Tests ability to extract relevant information while ignoring semantically similar distractors.
Maps to §7.3.2 Perceptual Inhibition: "ignore distracting or goal-irrelevant perceptual information."

### Task Description
A passage contains a set of factual statements. Some are marked as "from Source A"
(the reliable source) and others from "Source B" (unreliable). The model must report
ONLY facts from Source A, ignoring Source B even when Source B facts are more vivid/salient.

### Prompt Template
```
The following report contains statements from two sources:
- Source A (verified): these are the facts you need.
- Source B (unverified): ignore these completely.

Extract ONLY the numerical values reported by Source A.

---
{passage with interleaved Source A and Source B statements}
---

ANSWER: [value1], [value2], [value3], ...
```

### Difficulty Scaling
| Level | Source A Facts | Source B Distractors | Semantic Similarity | Labeling |
|-------|--------------|---------------------|--------------------|---------|
| Easy | 3 | 3 | Low | Explicit "[Source A]" tags |
| Medium | 4 | 6 | Medium | Inline attribution ("according to Source A") |
| Hard | 5 | 10 | High (same domain) | Subtle attribution (buried in sentence) |
| Expert | 6 | 15 | Very High (near-identical phrasing) | Minimal cues (must infer from context) |

### Ground Truth
All Source A values are known. All Source B values are known (these are the distractors).

### Scoring
- **Signal Recall**: (Source A facts found) / (total Source A facts)
- **Distractor Intrusion Rate**: (Source B facts included) / (total Source B facts)
- **Attention Precision**: (Source A facts found) / (total items reported)
- **Selective Attention Score (SAS)**: Recall × (1 - Intrusion Rate)
- kbench assertion: `assert_contains_regex` for each Source A value; fail if Source B value appears

### Power-Law Measurement (from GSM-DC)
```python
# Run with N_distractors = [0, 3, 6, 10, 15, 20]
# Fit: error_rate = alpha * N^delta
# Report delta per model — higher delta = more susceptible to distractors
```

### Procedural Generation
- Domain: randomly chosen (financial report, weather data, lab results, sports stats)
- Source A facts: numerical values with units (e.g., "$12.50", "23°C", "4.7 mg/L")
- Source B distractors: same format, similar range, different values
- Attribution: varies by difficulty (explicit tags → buried mentions → context-only)
- Distractor salience: Source B facts can be made more vivid (larger numbers, exclamation marks, emotional framing)

---

## Task D: Rule Shift (Attention Shifting)

### Cognitive Construct
Tests ability to switch attentional focus when the task rule changes.
Maps to §7.3.2 Attention Shifting: "actively shift attention from one location or piece of information to another."

### Task Description
The model is given a classification rule, applies it to several items, then the rule
changes mid-stream. The model must apply the NEW rule to remaining items without perseverating on the old rule.

### Prompt Template
```
Classify each word below according to the CURRENT rule.
The rule will change partway through — pay attention to the instruction.

RULE: Classify by CATEGORY (animal, food, or object)

1. tiger → ?
2. hammer → ?
3. banana → ?
4. eagle → ?

NEW RULE: Classify by FIRST LETTER (A-M = "early", N-Z = "late")

5. tiger → ?
6. banana → ?
7. pencil → ?
8. orange → ?

Format: ANSWER:
1. [answer]
2. [answer]
...
```

### Difficulty Scaling
| Level | Items Before Switch | Items After | Rule Similarity | Warning |
|-------|-------------------|-------------|----------------|---------|
| Easy | 3 | 3 | Low (category→letter) | "NEW RULE:" header |
| Medium | 5 | 5 | Medium (category→syllables) | "Rule change:" inline |
| Hard | 7 | 7 | High (size→weight) | Subtle mention |
| Expert | 4+4+4 | 2 switches | Very High (color→shade) | Buried in text |

### Ground Truth
All classifications are deterministic given the rule and the word.

### Scoring
- **Pre-switch accuracy**: correct classifications before rule change
- **Post-switch accuracy**: correct classifications after rule change
- **Switch cost**: pre-switch accuracy - post-switch accuracy (>0 means shifting hurts)
- **Perseveration errors**: items after switch classified using OLD rule
- **Perseveration rate**: perseveration errors / total post-switch items
- kbench assertion: `assert_contains_regex` for each item's correct classification

### Procedural Generation
- Word pools: 200+ common English words with known properties (category, first letter, syllable count, length)
- Rules: randomly chosen from pool of 10+ classification rules
- Rule pairs: pre-selected to have varying similarity levels
- Switch point: random position (not first or last item)
- Multiple switches for Expert difficulty

---

## Task E: Anomaly Detection (Stimulus-Driven Attention)

### Cognitive Construct
Tests whether the model notices unexpected/anomalous items during a routine task.
Maps to §7.3.3: "attention to be directed in a 'bottom-up' way toward new stimuli."

### Task Description
The model performs a routine primary task (e.g., counting items, summing numbers).
An anomalous element is embedded that is unrelated to the primary task but clearly unusual.
The model is asked about the primary task AND whether it noticed anything unusual.

### Prompt Template
```
PRIMARY TASK: Count how many times the word "the" appears in the passage below.

---
{passage with "the" appearing N times, plus one anomalous sentence}
---

1. How many times does "the" appear? ANSWER: [number]
2. Did you notice anything unusual or out of place in the passage? If yes, describe it.
   ANSWER: [description or "nothing unusual"]
```

### The Anomaly Types
| Type | Example | Detectability |
|------|---------|--------------|
| Language switch | One sentence in French among English | High |
| Format break | `<code>print("hello")</code>` in a narrative | High |
| Factual absurdity | "The sun set in the east as usual" | Medium |
| Semantic anomaly | A sentence about space in a cooking recipe | Medium |
| Subtle inconsistency | A character's name changes mid-paragraph | Low |
| Numerical anomaly | All prices are $5-$50 except one at $50,000 | Medium |

### Difficulty Scaling
| Level | Primary Task Complexity | Anomaly Saliency | Anomaly Count |
|-------|------------------------|-------------------|---------------|
| Easy | Simple counting | High (language switch) | 1 |
| Medium | Multi-step counting | Medium (format break) | 1 |
| Hard | Counting + filtering | Low (subtle inconsistency) | 1 |
| Expert | Complex primary task | Very Low + primary task demanding | 2 |

### Ground Truth
- Primary task answer: programmatically computed
- Anomaly: known type and location

### Scoring
- **Primary task accuracy**: exact match on count/sum
- **Anomaly detection rate**: did the model identify the anomaly? (binary)
- **Anomaly precision**: if model reports anomalies, are they the real ones?
- **Dual-task score**: primary_accuracy × anomaly_detection_rate
- kbench assertion: `assert_true` for primary task; `assert_contains_regex` for anomaly keywords

### Procedural Generation
- Passage: procedurally generated narrative (500-3000 words)
- Primary task: randomly chosen (count word X, sum numbers, list proper nouns)
- Anomaly: randomly selected type, randomly positioned
- Primary task answer: computed during generation

---

## Cross-Cutting Design Principles

### Contamination Resistance
1. All content is procedurally generated — no instance exists in training data
2. Entity names, numbers, locations randomized per instance
3. Include canary string: `<!-- CogAttention-v1-{instance_hash} -->` (detects training leakage)

### Position Bias Control
- Target/signal positions are uniformly randomized across the document
- Report position-stratified metrics (quintile accuracy)
- Control for U-shape bias (Lost in the Middle)

### Lexical Overlap Prevention
- Query terms and target answers have minimal lexical overlap (NoLiMa lesson)
- Distractors may share keywords with query to test semantic (not lexical) attention

### Statistical Rigor
- 32 items per task × 5 tasks = 160 total items
- Confidence intervals reported for all metrics
- Random baseline computed for each task
- Effect sizes (Cohen's d) between models

### Output Format Standardization
All tasks require answers in `ANSWER:` format for reliable regex extraction:
```python
ANSWER_PATTERN = r"ANSWER:\s*(.+?)(?:\n|$)"
```

---

## Composite Metric: Cognitive Attention Score (CAS)

```
CAS = 0.20 × capacity_score
    + 0.20 × sustained_score
    + 0.25 × selective_score    (weighted higher — most diagnostic)
    + 0.20 × shifting_score
    + 0.15 × stimulus_driven_score  (weighted lower — more exploratory)
```

Each sub-score is normalized to [0, 1] across difficulty levels.

CAS enables radar-chart cognitive profiling per model, as described in DeepMind paper Figure 2.
