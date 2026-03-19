# CogAttention: Testing Whether Language Models Can Pay Attention

**Kaggle Community Benchmarks — Attention Track**

A comprehensive cognitive attention benchmark for LLMs, adapting 13 cognitive psychology paradigms into 15 procedurally generated task types across 5 attention abilities.

## Key Numbers

| Metric | Value |
|--------|-------|
| Task types | 15 (across 5 cognitive abilities) |
| Total items | 560 |
| Difficulty tiers | 5 (Easy, Medium, Hard, Expert, Frontier) |
| Fine-grained assertions | 1,168 |
| Kaggle notebooks | 5 self-contained .ipynb files |
| Unit tests | 124 passing |

## Cognitive Abilities Tested

| Ability | Tasks | Paradigm Source |
|---------|-------|----------------|
| **Attention Capacity** | Thread Tracking, Proactive Interference, Attentional Blink | MOT (Pylyshyn), PI-LLM (Wang & Sun) |
| **Sustained Attention** | Vigilance Probe, Stream Segregation, Context Dilution, Semantic NIAH, Multi-hop | CPT (Mackworth), Dichotic Listening (Cherry) |
| **Selective Attention** | Distractor Filtering, Semantic Stroop, Flanker | SiN, Stroop (1935), Eriksen Flanker |
| **Attention Shifting** | Rule Shift, Inhibition of Return | WCST (Monsell), IOR |
| **Stimulus-Driven** | Anomaly Detection | Inattentional Blindness (Simons & Chabris) |

## Local Validation Results

| Model | CAS Score | 95% CI |
|-------|-----------|--------|
| Qwen2.5-72B-Instruct | 0.81 | [0.71, 0.86] |
| Llama-3.1-8B-Instruct | 0.73 | [0.62, 0.81] |
| Phi-3.5-mini-instruct | 0.57 | [0.45, 0.61] |

Cohen's d between Phi-3.5 and Qwen-72B: **0.72 (medium effect)**.

## Repository Structure

```
src/
  generators/       # 15 procedural task generators (all seeded, contamination-resistant)
  scorers/           # 6 scoring modules + composite CAS aggregator
  analysis/          # Statistical analysis (bootstrap CI, effect sizes, IRT, degradation, position bias)
  human_baseline/    # Scaffolding for human evaluation (Prolific/MTurk export)
  visualize.py       # 9 publication-quality figures (dark theme, 300 DPI)
  inference_engine.py # vLLM wrapper with OOM resilience
  run_local_eval.py  # Main evaluation harness with checkpoint/resume

notebooks/           # 5 Kaggle SDK notebooks (560 items, 1168 assertions)
tests/               # 124 unit tests (generators, scorers, analysis)
docs/                # Competition writeup
figures/             # Generated visualizations
results/             # Pilot evaluation results (3 models)
tasks/               # Planning docs and progress tracker
```

## Scoring Architecture

- **Arithmetic CAS** — weighted mean across 13 task types (compensatory)
- **Geometric CAS** — weighted geometric mean (non-compensatory: zero on any ability tanks composite)
- **Per-task metrics** — recall, precision, intrusion rate, switch cost, perseveration rate, etc.
- **Attentional residue classification** — errors split into perseveration, residue, and random
- **Bootstrapped 95% CIs** — 10,000 resamples for all metrics
- **Power-law degradation** — `E(m) ~ a * m^delta` fit for selective attention

## Figures

9 publication-quality visualizations:
1. CAS scores by model (bar chart)
2. Task performance heatmap (model x task)
3. Difficulty degradation curves (line chart)
4. Cognitive attention profile (radar chart)
5. Power-law degradation (log-log scatter + fit)
6. Arithmetic vs geometric CAS (grouped bar)
7. Shifting error breakdown (stacked bar: perseveration/residue/random)
8. Position bias U-curve (Lost in the Middle)
9. Selectivity frontier (recall vs intrusion scatter)

## Quick Start

```bash
# Run tests
python3 -m pytest tests/ -v

# Generate all 560 benchmark instances
python3 -c "from src.run_local_eval import generate_all_instances; print(len(generate_all_instances(2026)))"

# Generate figures (from pilot data)
python3 src/visualize.py

# Run local evaluation (requires GPU + vLLM)
python3 -m src.run_local_eval --model phi3 --pilot-n 2
```

## References

- Cherry, E.C. (1953). Some experiments on the recognition of speech. *JASA*.
- Liu, N.F. et al. (2024). Lost in the middle. *TACL*.
- Mackworth, N.H. (1948). Breakdown of vigilance. *QJEP*.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*.
- Pylyshyn, Z.W. & Storm, R.W. (2001). Tracking multiple independent targets. *Spatial Vision*.
- Simons, D.J. & Chabris, C.F. (1999). Gorillas in our midst. *Perception*.
- Stroop, J.R. (1935). Interference in serial verbal reactions. *JEP*.
- Wang, Y. & Sun, Y. (2025). PI-LLM. *ICML Workshop*.
- Yang, Z. et al. (2025). Distractor-induced degradation. *EMNLP*.
