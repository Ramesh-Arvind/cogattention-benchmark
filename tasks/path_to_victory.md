# Path to Victory — 1st Place Master Plan

**Status**: Active | **Created**: 2026-03-19

---

## Strategy Overview

The CogAttention benchmark (560 items, 5 notebooks) is functionally complete but scores 1.00
across all frontier models on the Kaggle leaderboard. This plan upgrades scoring, adds
mechanistic analysis, statistical rigor, and a showstopper public notebook.

---

## Phase A: Non-Compensatory Scoring (Geometric Mean CAS)

**Why**: Arithmetic CAS allows a model to compensate for total failure on one ability with
high scores elsewhere. Geometric mean penalizes this — a zero on any task tanks the composite.

- Add `cas_geometric` alongside existing `cas_score` in `compute_cas()`
- Geometric weighted mean: `exp(sum(w_i * ln(s_i + eps)) / sum(w_i))`
- Backward compatible: `cas_score` key unchanged

## Phase B: Attentional Residue Classification

**Why**: Current shifting scorer lumps all errors together. Distinguishing *residue errors*
(answering with a pre-switch correct answer) from *random errors* reveals whether the model
is still mentally "stuck" on the old task.

- Split "incorrect" into "residue" (matches any pre-switch answer) and "random"
- Add metrics: `residue_error_count`, `residue_rate`, `random_error_count`, `random_error_rate`

## Phase C: Degradation Coefficient (Power-Law Fit)

**Why**: GSM-DC (EMNLP 2025) showed error rate follows `E(m) ~ m^delta` where m is noise ratio.
Our selective attention data can fit this. Delta is a single number that characterizes how
vulnerable a model is to distractors.

- Power-law fit: `ln(error_rate) = ln(a) + delta * ln(noise_ratio)` via numpy
- Report `delta`, `r_squared`, `a` per model

## Phase D: Bootstrapped Confidence Intervals

**Why**: BetterBench found 58% of benchmarks report NO statistical significance. CIs alone
differentiate us from most submissions.

- `bootstrap_ci(values, n_bootstrap=10000, ci_level=0.95)` → point estimate + CI
- Apply to CAS and per-task scores

## Phase E: Position Bias Analysis (U-Shaped Attention)

**Why**: Lost in the Middle (TACL 2024) showed models have U-shaped attention. Showing this
pattern in our data adds analytical depth.

- Bin items by target position in context (quintiles)
- Report accuracy per position bin

## Phase F: Visualization Upgrades

4 new figures matching existing dark theme / 300 DPI:
1. Power-law degradation scatter + fit line (log-log)
2. Arithmetic vs geometric CAS (grouped bars)
3. Shifting error breakdown (stacked bar: perseveration vs residue vs random)
4. Position bias U-curve (accuracy vs position)

## Phase G: Writeup Discussion Section

~300 words mapping cognitive failures to transformer architecture:
- Vigilance decrement → attention weight dilution over long sequences
- Perseveration → residual connections carrying stale representations
- Distractor intrusion → softmax probability mass on irrelevant tokens
- Capacity limit → fixed number of attention heads per layer
- Inattentional blindness → masked self-attention ignoring unexpected patterns

## Phase H: Human Baseline Scaffolding

Tooling only (no external resources needed):
- `sample_human_tasks()` — sample representative items for human eval
- `format_for_qualtrics()` — export to survey format

## Phase I: Kaggle Environment Hardening

- OOM guard wrapper in generated notebook code cells
- Progress estimation (items/sec, ETA)
- 8.5-hour hard timeout with graceful partial submission

## Phase J: IRT Analysis (if time permits)

- 2PL Item Response Theory model
- Extract difficulty and discrimination parameters per item
- Fallback to scipy.optimize if `girth` not installed

---

## Priority Order

1. Phases A-E (scoring + analysis) — highest impact, pure code
2. Phase F (visualization) — showcase results
3. Phase G (writeup) — judge-facing
4. Phases H-J — stretch goals
