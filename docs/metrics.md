# CogAttention — Metrics Specification

## Per-Task Metrics

### Task A: Thread Tracking (Capacity)
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Assignment Accuracy | correct_assignments / total_people | [0,1] | Core accuracy |
| Capacity Breakpoint | N where accuracy < 0.5 | integer | Model's capacity limit |
| Capacity Curve AUC | Area under accuracy-vs-N curve | [0,1] | Overall capacity |

### Task B: Vigilance Probe (Sustained)
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Target Recall | targets_found / total_targets | [0,1] | Ability to find targets |
| Intrusion Rate | false_targets / total_reported | [0,1] | Noise leakage (lower=better) |
| Vigilance Decrement | recall_Q1 - recall_Q5 | [-1,1] | Attention decay (>0=decrement) |
| Position Uniformity | std(recall_per_quintile) | [0,0.5] | Position bias (lower=better) |

### Task C: Distractor Filtering (Selective)
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Signal Recall | signal_found / total_signals | [0,1] | Finding relevant info |
| Distractor Intrusion | distractors_included / total_distractors | [0,1] | Noise leakage |
| Attention Precision | signal_found / total_reported | [0,1] | Selectivity |
| SAS Score | recall × (1 - intrusion) | [0,1] | Composite selective attention |

### Task D: Rule Shift (Shifting)
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Pre-Switch Accuracy | correct_pre / total_pre | [0,1] | Baseline accuracy |
| Post-Switch Accuracy | correct_post / total_post | [0,1] | Accuracy after shift |
| Switch Cost | pre_accuracy - post_accuracy | [-1,1] | Cost of shifting (>0=cost) |
| Perseveration Rate | old_rule_errors / total_post | [0,1] | Stuck on old rule |

### Task E: Anomaly Detection (Stimulus-Driven)
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Primary Task Accuracy | exact_match(answer, gt) | {0,1} | Main task correctness |
| Anomaly Detection Rate | anomaly_detected / total_anomalies | [0,1] | Bottom-up attention |
| Dual-Task Score | primary_acc × anomaly_rate | [0,1] | Combined performance |

### Task D (extended): Attentional Residue Classification
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Residue Error Count | post_answers matching any pre-switch answer | integer | Stuck on old context |
| Residue Rate | residue_errors / total_post | [0,1] | Proportion of residue errors |
| Random Error Count | post errors not perseveration or residue | integer | True random failures |
| Random Error Rate | random_errors / total_post | [0,1] | Proportion of random errors |

### Task C (extended): Noise-Ratio Degradation
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Noise Ratio | n_distractors / (n_signals + n_distractors) | [0,1] | Difficulty proxy |
| Degradation Coefficient (δ) | Power-law: E(m) ~ a × m^δ | [0,∞) | Vulnerability to distractors |

### Task F: Visual Stroop (Multimodal VLM Extension)
| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Accuracy | correct_color / total | [0,1] | Ink color identification |
| Stroop Error Rate | word_read_errors / total | [0,1] | OCR overpowering color |
| Stroop Resistance | 1 - stroop_error_rate | [0,1] | Perceptual inhibition |

## Composite: Cognitive Attention Score (CAS)

**Arithmetic CAS** (compensatory — high scores on easy tasks can mask failures):
```
CAS = Σ(weight_i × score_i) / Σ(weight_i)
```

Weights (13 task types): capacity(0.07), interference(0.07), blink(0.03), sustained(0.07), stream_segregation(0.07), context_dilution(0.08), semantic_niah(0.08), multihop(0.08), selective(0.07), stroop(0.07), flanker(0.04), shifting(0.07), inhibition_return(0.07), anomaly(0.13).

**Geometric CAS** (non-compensatory — zero on any task tanks the composite):
```
CAS_geo = exp(Σ(weight_i × ln(score_i + ε)) / Σ(weight_i))
```

## Statistical Analysis

| Method | Implementation | Purpose |
|--------|---------------|---------|
| Bootstrapped CIs | `src/analysis/bootstrap.py` (10,000 resamples) | Significance testing |
| Cohen's d | `src/analysis/effect_size.py` | Between-model effect sizes |
| Rank-biserial | `src/analysis/effect_size.py` | Non-parametric effect size |
| 2PL IRT | `src/analysis/irt.py` (scipy.optimize) | Item difficulty & discrimination |
| Power-law fit | `src/analysis/degradation.py` | Selective attention degradation |
| Position bias | `src/analysis/position_bias.py` | U-shape / Lost-in-the-Middle |

## Random Baselines

| Task | Random Baseline | Method |
|------|----------------|--------|
| A: Thread Tracking | 1/N! (permutation) | ~0.17 for N=3 |
| B: Vigilance Probe | Proportional to target density | ~0.05 |
| C: Distractor Filtering | Equal probability signal/distractor | ~0.33 |
| D: Rule Shift | 1/num_categories | ~0.33 |
| E: Anomaly Detection | 0.5 (binary guess) × primary_random | ~0.10 |
| F: Visual Stroop | 1/num_colors | ~0.125 |
