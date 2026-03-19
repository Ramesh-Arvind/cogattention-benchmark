# CogAttention: Testing Whether Language Models Can Pay Attention

**Kaggle Community Benchmarks — Attention Track**

A comprehensive cognitive attention benchmark for LLMs, adapting 13 cognitive psychology paradigms into 16 procedurally generated task types across 5 attention abilities — including a multimodal Visual Stroop extension for VLMs.

## Key Numbers

| Metric | Value |
|--------|-------|
| Task types | 16 (15 text + 1 visual, across 5 cognitive abilities) |
| Total items | 600 (560 text + 40 visual Stroop) |
| Difficulty tiers | 5 (Easy, Medium, Hard, Expert, Frontier) |
| Fine-grained assertions | 1,168 |
| Kaggle notebooks | 5 task notebooks + 1 interactive public demo |
| Unit tests | 124 passing |
| Scoring variants | Arithmetic CAS + Geometric CAS (non-compensatory) |

## Cognitive Abilities Tested

| Ability | Tasks | Paradigm Source |
|---------|-------|----------------|
| **Attention Capacity** | Thread Tracking, Proactive Interference, Attentional Blink | MOT (Pylyshyn), PI-LLM (Wang & Sun) |
| **Sustained Attention** | Vigilance Probe, Stream Segregation, Context Dilution, Semantic NIAH, Multi-hop | CPT (Mackworth), Dichotic Listening (Cherry) |
| **Selective Attention** | Distractor Filtering, Semantic Stroop, Flanker, **Visual Stroop (VLM)** | SiN, Stroop (1935), Eriksen Flanker |
| **Attention Shifting** | Rule Shift, Inhibition of Return | WCST (Monsell), IOR |
| **Stimulus-Driven** | Anomaly Detection | Inattentional Blindness (Simons & Chabris) |

## Local Validation Results (with Frontier Tier)

Evaluated on 140 items per model (2 per difficulty × 15 tasks × 5 tiers) using vLLM with deterministic generation (temperature=0, top_p=1):

| Model | Parameters | CAS (Arithmetic) | CAS (Geometric) | 95% CI |
|-------|-----------|-------------------|-----------------|--------|
| Qwen2.5-72B-Instruct | 72B | **0.833** | 0.806 | [0.785, 0.878] |
| Llama-3.1-8B-Instruct | 8B | **0.685** | 0.640 | [0.621, 0.747] |
| Phi-3.5-mini-instruct | 3.8B | **0.567** | 0.479 | [0.513, 0.619] |

**Effect sizes:** Phi vs Qwen d=0.68 (medium), Llama vs Qwen d=0.41 (small). CIs do not overlap between top and bottom models.

**Frontier tier works:** Context dilution drops to 0.0 for all models at Frontier. Capacity Frontier=0.19 for Qwen (vs 1.0 Easy). Anomaly detection: Phi=0.16 overall.

**Geometric CAS** penalizes models that completely fail on any ability — Phi drops from 0.567 (arithmetic) to 0.479 (geometric), exposing hidden weaknesses.

## Repository Structure

```
src/
  generators/       # 16 procedural task generators (all seeded, contamination-resistant)
    visual_selective.py  # PIL-generated Visual Stroop images for VLMs
  scorers/           # 7 scoring modules + composite CAS aggregator
    visual_selective.py  # VLM Stroop scorer with error classification
  analysis/          # Statistical analysis suite
    bootstrap.py       # Bootstrapped CIs (10,000 resamples)
    effect_size.py     # Cohen's d + rank-biserial correlation
    irt.py             # 2PL Item Response Theory (scipy.optimize)
    degradation.py     # Power-law degradation coefficient
    position_bias.py   # U-shape / Lost-in-the-Middle detection
    discrimination.py  # Cross-model spread + ceiling/floor detection
  human_baseline/    # Scaffolding for human evaluation (Prolific/MTurk export)
  visualize.py       # 9 publication-quality figures (dark theme, 300 DPI)
  inference_engine.py # vLLM wrapper with OOM resilience
  run_local_eval.py  # Main evaluation harness with checkpoint/resume

notebooks/           # 5 Kaggle SDK task notebooks + 1 interactive public demo
tests/               # 124 unit tests (generators, scorers, analysis)
docs/                # Competition writeup + metrics spec + task specs
figures/             # 9 generated visualizations + Visual Stroop samples
results/             # Evaluation results (3 models × 140 items)
tasks/               # Planning docs and progress tracker
```

## Scoring Architecture

- **Arithmetic CAS** — weighted mean across 14 task types (compensatory)
- **Geometric CAS** — weighted geometric mean (non-compensatory: zero on any ability tanks composite)
- **Per-task metrics** — recall, precision, intrusion rate, switch cost, perseveration rate, Stroop resistance, etc.
- **Attentional residue classification** — shifting errors split into perseveration, residue, and random
- **Bootstrapped 95% CIs** — 10,000 resamples for all metrics
- **Cohen's d + rank-biserial** — pairwise effect sizes between models
- **2PL IRT** — item difficulty and discrimination parameters
- **Power-law degradation** — `E(m) ~ a * m^delta` fit for selective attention
- **Position bias** — U-shape detection (Lost in the Middle)

## Figures

9 publication-quality visualizations (dark theme, 300 DPI):
1. CAS scores by model (bar chart)
2. Task performance heatmap (model x task)
3. Difficulty degradation curves (line chart)
4. Cognitive attention profile (radar chart)
5. Power-law degradation (log-log scatter + fit)
6. Arithmetic vs geometric CAS (grouped bar)
7. Shifting error breakdown (stacked bar: perseveration/residue/random)
8. Position bias U-curve (Lost in the Middle)
9. Selectivity frontier (recall vs intrusion scatter)

Plus: Visual Stroop sample images (`figures/visual_stroop_sample_*.png`)

## Installation

```bash
# Clone and install (editable mode)
git clone https://github.com/Ramesh-Arvind/cogattention-benchmark.git
cd cogattention-benchmark
pip install -e .

# With visualization support (matplotlib + plotly)
pip install -e ".[viz]"

# With GPU evaluation support (torch + vLLM + transformers)
pip install -e ".[eval]"

# With multimodal Visual Stroop support (Pillow)
pip install -e ".[vision]"

# Everything
pip install -e ".[all]"

# Development (includes pytest)
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run all 124 tests
python3 -m pytest tests/ -v

# Generate all 600 benchmark instances (560 text + 40 visual)
python3 -c "
from src.generators.capacity import generate_capacity_dataset
from src.generators.sustained import generate_sustained_dataset
from src.generators.selective import generate_selective_dataset
from src.generators.shifting import generate_shifting_dataset
from src.generators.anomaly import generate_anomaly_dataset
from src.generators.visual_selective import generate_visual_stroop_dataset

total = sum(len(g(seed=2026)) for g in [
    generate_capacity_dataset, generate_sustained_dataset,
    generate_selective_dataset, generate_shifting_dataset,
    generate_anomaly_dataset, generate_visual_stroop_dataset,
])
print(f'Total instances: {total}')
"

# Generate all 9 figures (from pilot data)
python3 src/visualize.py

# Run local evaluation (requires GPU + vLLM)
python3 -m src.run_local_eval --model phi3 --pilot-n 2

# Run full evaluation on all models
python3 -m src.run_local_eval --model all --full
```

## Usage as a Library

```python
# Generate benchmark instances
from src.generators.capacity import generate_capacity_dataset
from src.generators.visual_selective import generate_visual_stroop_dataset

capacity_items = generate_capacity_dataset(seed=42)
visual_items = generate_visual_stroop_dataset(seed=42)

# Score responses
from src.scorers.capacity import score_capacity
from src.scorers.composite import compute_cas

result = score_capacity(capacity_items[0], "ANSWER:\n- Alice: red key\n- Bob: blue book")
print(f"Accuracy: {result.metrics['accuracy']}")

# Statistical analysis
from src.analysis.bootstrap import bootstrap_ci
from src.analysis.effect_size import cohens_d
from src.analysis.irt import fit_irt_model

ci = bootstrap_ci([0.8, 0.7, 0.9, 0.85], n_bootstrap=10000)
print(f"Mean: {ci['point_estimate']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

d = cohens_d([0.9, 0.85, 0.88], [0.6, 0.55, 0.58])
print(f"Cohen's d: {d:.2f}")
```

## Integration with lm-evaluation-harness

The package structure supports integration with EleutherAI's lm-evaluation-harness:

```python
# Each generator returns TaskInstance objects with .prompt and .gold_answer
# Each scorer takes (TaskInstance, response_str) and returns ScoreResult
# ScoreResult.metrics is a dict of float scores

from src.generators.base import TaskInstance
from src.scorers.base import ScoreResult
from src.eval_config import get_task_registry

# Get all task types with their generators and scorers
registry = get_task_registry()
for task_type, (gen_fn, score_fn) in registry.items():
    instances = gen_fn(seed=2026)
    # Feed instance.prompt to your model, get response
    # result = score_fn(instance, model_response)
```

## Discussion: Cognitive Failures Map to Transformer Architecture

| Cognitive Failure | Transformer Mechanism | Our Evidence |
|-------------------|----------------------|-------------|
| Vigilance decrement | Softmax dilution over long contexts | U-shaped accuracy curve |
| Perseveration | Residual connections carrying stale KV states | 60%+ of shifting errors are perseveration |
| Distractor intrusion | MLP pre-training priors overpower in-context attention | Factual distractors intrude more than nonsense |
| Capacity limit | Fixed attention heads per layer | Accuracy cliff at 4+ tracked objects |
| Inattentional blindness | Causal mask prevents backward anomaly detection | Phi-3.5 scores 0.16 on dual-task anomaly |

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
