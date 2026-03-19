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

## Composite: Cognitive Attention Score (CAS)

```
CAS = 0.20 × normalize(capacity_AUC)
    + 0.20 × normalize(1 - vigilance_decrement + recall)
    + 0.25 × normalize(SAS)
    + 0.20 × normalize(post_switch_accuracy)
    + 0.15 × normalize(dual_task_score)
```

## Random Baselines

| Task | Random Baseline | Method |
|------|----------------|--------|
| A: Thread Tracking | 1/N! (permutation) | ~0.17 for N=3 |
| B: Vigilance Probe | Proportional to target density | ~0.05 |
| C: Distractor Filtering | Equal probability signal/distractor | ~0.33 |
| D: Rule Shift | 1/num_categories | ~0.33 |
| E: Anomaly Detection | 0.5 (binary guess) × primary_random | ~0.10 |
