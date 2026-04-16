# CogAttention: Testing Whether Language Models Can Pay Attention

**Kaggle Community Benchmarks — Attention Track**

A comprehensive cognitive attention benchmark for LLMs, adapting 13 cognitive psychology paradigms into 16 procedurally generated task types across 5 attention abilities — including a multimodal Visual Stroop extension for VLMs.

## Key Numbers

| Metric | Value |
|--------|-------|
| Task types | 16 (14 text + 2 visual, across 5 cognitive abilities) |
| Total items | 860 (560 text + 150 Visual Stroop + 150 Visual Inattentional) |
| Difficulty tiers | 5 (Easy, Medium, Hard, Expert, Frontier) |
| Kaggle benchmark tasks | 19 tasks across 17 notebooks on Kaggle Benchmarks |
| Frontier models evaluated | 7 (DeepSeek-R1, Gemini 2.5 Flash, Claude Opus 4.6, Claude Sonnet 4.5, GPT-OSS-20B, Qwen3-Next-80B, Gemma-3-27B) |
| Unit tests | 148 passing |
| Scoring variants | Arithmetic CAS + Geometric CAS (non-compensatory) |

## Cognitive Abilities Tested

| Ability | Tasks | Paradigm Source |
|---------|-------|----------------|
| **Attention Capacity** | Thread Tracking, Proactive Interference, Attentional Blink | MOT (Pylyshyn), PI-LLM (Wang & Sun) |
| **Sustained Attention** | Vigilance Probe, Stream Segregation, Context Dilution, Semantic NIAH, Multi-hop | CPT (Mackworth), Dichotic Listening (Cherry) |
| **Selective Attention** | Distractor Filtering, Semantic Stroop, Flanker, **Visual Stroop (VLM)** | SiN, Stroop (1935), Eriksen Flanker |
| **Attention Shifting** | Rule Shift, Inhibition of Return | WCST (Monsell), IOR |
| **Stimulus-Driven** | Anomaly Detection, **Visual Inattentional Blindness (VLM)** | Inattentional Blindness (Simons & Chabris) |

## Kaggle Benchmarks Leaderboard (Frontier Models)

Evaluated on the Kaggle Community Benchmarks platform across 19 tasks (April 2026):

| Model | Score | Key Failures |
|-------|-------|-------------|
| DeepSeek-R1-0528 | **0.895** | visual stroop, visual inattentional |
| Claude Opus 4.6 | **0.895** | visual stroop, visual inattentional |
| Gemini 2.5 Flash | 0.842 | blink, visual stroop, visual inattentional |
| Claude Sonnet 4.5 | 0.842 | shifting, visual stroop, visual inattentional |
| GPT-OSS-20B | 0.842 | blink, visual stroop, visual inattentional |
| Qwen3-Next-80B | 0.737 | blink, shifting, anomaly, visual stroop, visual inattentional |
| Gemma-3-27B | **0.684** | blink, capacity, shifting, anomaly, visual stroop, visual inattentional |

**21-point spread** across 7 frontier models. Point-biserial discrimination (CTT, Pearson `r_pb` between task pass/fail and total CAS): **anomaly r_pb=0.94**, **shifting r_pb=0.77**, **capacity r_pb=0.75**, **blink r_pb=0.68** — the four discriminating tasks (3/7 fail shifting, 2/7 fail anomaly, 1/7 fails capacity, 4/7 fail blink). Ten remaining text-only tasks are at ceiling; two visual tasks at floor (all 7 fail).

## Local Validation Results (Open Models)

Evaluated on 140 items per model (2 per difficulty × 14 text tasks × 5 tiers) using vLLM with deterministic generation (temperature=0, top_p=1):

| Model | Parameters | CAS (Arithmetic) | CAS (Geometric) | 95% CI |
|-------|-----------|-------------------|-----------------|--------|
| Qwen2.5-72B-Instruct | 72B | **0.833** | 0.806 | [0.785, 0.878] |
| Llama-3.1-8B-Instruct | 8B | **0.685** | 0.640 | [0.621, 0.747] |
| Phi-3.5-mini-instruct | 3.8B | **0.567** | 0.479 | [0.513, 0.619] |

**Effect sizes:** Phi vs Qwen d=0.68 (medium), Llama vs Qwen d=0.41 (small). CIs do not overlap between top and bottom models.

**Frontier tier works:** Context dilution drops to 0.0 for all models at Frontier. Capacity Frontier=0.19 for Qwen (vs 1.0 Easy). Anomaly detection: Phi=0.16 overall.

**Geometric CAS** penalizes models that completely fail on any ability — Phi drops from 0.567 (arithmetic) to 0.479 (geometric), exposing hidden weaknesses.

## What You Can Do With CogAttention

| Use Case | How | Entry Point |
|----------|-----|-------------|
| **Benchmark a frontier model on Kaggle** | Submit any of 19 notebooks to the Kaggle Community Benchmarks platform and get a score on the public leaderboard | `notebooks/task_*.ipynb` |
| **Evaluate a local/open model with vLLM** | Run the full 14-task text suite (or a subset via `--tasks`) with OOM-safe checkpointed inference | `python -m src.run_local_eval --model <alias> --full` |
| **Plug into lm-evaluation-harness** | Each generator yields `TaskInstance` objects; each scorer returns a `ScoreResult` — drop into any external harness | `src/eval_config.py::get_task_registry()` |
| **Diagnose a model's attention profile** | Geometric CAS + radar chart expose which of the 5 abilities (capacity / sustained / selective / shifting / stimulus-driven) collapse | `src/scorers/composite.py`, `src/visualize.py` |
| **Compare two models with stats** | Bootstrapped 95% CIs, Cohen's d, rank-biserial, and point-biserial CTT discrimination | `src/analysis/` |
| **Extend to new paradigms** | All generators are procedural + seeded — fork a `generator.py` + `scorer.py` pair and register it | `src/generators/base.py`, `src/scorers/base.py` |
| **Test multimodal attention (VLMs)** | PIL-generated Visual Stroop and Inattentional Blindness items with programmatic ground truth | `src/generators/visual_*.py` |
| **Reproduce our 7-model leaderboard** | All seeds fixed (`seed=2026`); regenerate identical items, rescore, and replot the figures | `python src/visualize.py` |

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
    point_biserial.py  # CTT task-level discrimination against total CAS
    degradation.py     # Power-law degradation coefficient
    position_bias.py   # U-shape / Lost-in-the-Middle detection
    discrimination.py  # Cross-model spread + ceiling/floor detection
  human_baseline/    # Human baseline data (25 participants, Excel-based collection)
  visualize.py       # 9 publication-quality figures (dark theme, 300 DPI)
  inference_engine.py # vLLM wrapper with OOM resilience
  run_local_eval.py  # Main evaluation harness with checkpoint/resume

notebooks/           # 19 Kaggle SDK task notebooks (1 task per notebook)
tests/               # 148 unit tests (generators, scorers, analysis)
docs/                # Kaggle writeup (cogattention_v2.md) + detailed writeup (writeup.md) + metrics/task specs
figures/             # 9 generated visualizations + Visual Stroop samples
results/             # Evaluation results (3 models × 140 items)
tasks/               # Planning docs and progress tracker
```

## Kaggle Benchmark Notebooks

Each notebook contains exactly one `@kbench.task` (required by Kaggle Benchmarks platform):

| Notebook | Kaggle Task | Cognitive Ability | Items |
|----------|-------------|-------------------|-------|
| `task_a1_capacity.ipynb` | cogattention_capacity | Attention Capacity | 40 |
| `task_a2_interference.ipynb` | cogattention_interference | Attention Capacity | 40 |
| `task_a3_blink.ipynb` | cogattention_blink | Attention Capacity | 40 |
| `task_b1a_vigilance.ipynb` | cogattention_sustained | Sustained Attention | 40 |
| `task_b1b_stream.ipynb` | cogattention_stream_segregation | Sustained Attention | 40 |
| `task_b2_dilution_emh.ipynb` | cogattention_context_dilution | Sustained Attention | 24 |
| `task_b2_dilution_expert.ipynb` | cogattention_context_dilution | Sustained Attention | 8 |
| `task_b2_dilution_frontier_a.ipynb` | cogattention_context_dilution | Sustained Attention | 4 |
| `task_b2_dilution_frontier_b.ipynb` | cogattention_context_dilution | Sustained Attention | 4 |
| `task_b3_niah.ipynb` | cogattention_semantic_niah | Sustained Attention | 40 |
| `task_b4_multihop.ipynb` | cogattention_multihop | Sustained Attention | 40 |
| `task_c1_selective.ipynb` | cogattention_selective | Selective Attention | 40 |
| `task_c2_stroop.ipynb` | cogattention_stroop | Selective Attention | 40 |
| `task_c3_flanker.ipynb` | cogattention_flanker | Selective Attention | 40 |
| `task_d1_shifting.ipynb` | cogattention_shifting | Attention Shifting | 40 |
| `task_d2_inhibition_return.ipynb` | cogattention_inhibition_return | Attention Shifting | 40 |
| `task_e_anomaly.ipynb` | cogattention_anomaly | Stimulus-Driven | 40 |
| `task_f_visual_stroop.ipynb` | cogattention_visual_stroop | Selective (VLM) | 150 |
| `task_g_visual_inattentional.ipynb` | cogattention_visual_inattentional | Stimulus-Driven (VLM) | 150 |

## Scoring Architecture

- **Arithmetic CAS** — weighted mean across 14 task types (compensatory)
- **Geometric CAS** — weighted geometric mean (non-compensatory: zero on any ability tanks composite)
- **Per-task metrics** — recall, precision, intrusion rate, switch cost, perseveration rate, Stroop resistance, etc.
- **Attentional residue classification** — shifting errors split into perseveration, residue, and random
- **Bootstrapped 95% CIs** — 10,000 resamples for all metrics
- **Cohen's d + rank-biserial** — pairwise effect sizes between models
- **Point-biserial discrimination (CTT)** — Pearson `r_pb` between task pass/fail and total CAS; 4 of 16 deduplicated tasks pass the `r_pb >= 0.3` threshold (anomaly 0.94, shifting 0.77, capacity 0.75, blink 0.68)
- **Power-law degradation** — `E(m) ~ a * m^delta` fit for selective attention
- **Position bias** — U-shape detection (Lost in the Middle)

## Figures

11 publication-quality visualizations (dark theme, 300 DPI):
1. CAS scores by model (bar chart)
2. Task performance heatmap (model x task)
3. Difficulty degradation curves (line chart)
4. Cognitive attention profile (radar chart)
5. Power-law degradation (log-log scatter + fit)
6. Arithmetic vs geometric CAS (grouped bar)
7. Shifting error breakdown (stacked bar: perseveration/residue/random)
8. Position bias U-curve (Lost in the Middle)
9. Selectivity frontier (recall vs intrusion scatter)

10. Kaggle leaderboard bar chart (7 frontier models)
11. Kaggle radar chart (7 frontier models + human baseline)

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
# Run all 148 tests
python3 -m pytest tests/ -v

# Generate all 860 benchmark instances (560 text + 300 visual) across 16 task types
python3 -c "
from src.generators.capacity import generate_capacity_dataset
from src.generators.novel_capacity import generate_interference_dataset
from src.generators.attentional_blink import generate_blink_dataset
from src.generators.sustained import generate_sustained_dataset
from src.generators.novel_sustained import generate_stream_dataset
from src.generators.context_dilution import generate_dilution_dataset
from src.generators.semantic_niah import generate_sniah_dataset
from src.generators.multihop_attention import generate_multihop_dataset
from src.generators.selective import generate_selective_dataset
from src.generators.novel_selective import generate_stroop_dataset
from src.generators.flanker import generate_flanker_dataset
from src.generators.shifting import generate_shifting_dataset
from src.generators.inhibition_return import generate_ior_dataset
from src.generators.anomaly import generate_anomaly_dataset
from src.generators.visual_selective import generate_visual_stroop_dataset
from src.generators.visual_inattentional import generate_visual_inattentional_dataset

total = sum(len(g(seed=2026)) for g in [
    generate_capacity_dataset, generate_interference_dataset, generate_blink_dataset,
    generate_sustained_dataset, generate_stream_dataset, generate_dilution_dataset,
    generate_sniah_dataset, generate_multihop_dataset,
    generate_selective_dataset, generate_stroop_dataset, generate_flanker_dataset,
    generate_shifting_dataset, generate_ior_dataset, generate_anomaly_dataset,
    generate_visual_stroop_dataset, generate_visual_inattentional_dataset,
])
print(f'Total instances: {total}')  # expect 860
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

- Agrawal, N. (2026). Executive Functions: Cognitive Control Suite. *Kaggle Benchmarks*. kaggle.com/benchmarks/naivedhyaagrawal/executive-functions-the-cognitive-control-suite
- Barzykowski, K. et al. (2022). Cognitive inhibition behavioral tasks: online and laboratory data. *Data in Brief*, 43, 108398.
- Burnell, R., Kelly, O. et al. (2026). *Measuring Progress Toward AGI: A Cognitive Framework.* Google DeepMind.
- Cherry, E. C. (1953). Some experiments on the recognition of speech, with one and with two ears. *Journal of the Acoustical Society of America*, 25(5), 975--979.
- Eriksen, B. A., & Eriksen, C. W. (1974). Effects of noise letters upon the identification of a target letter in a nonsearch task. *Perception & Psychophysics*, 16(1), 143--149.
- Fernandez-Duque, D., Baird, J. A., & Posner, M. I. (2000). Executive attention and metacognitive regulation. *Consciousness and Cognition*, 9(2), 288--307.
- Liu, N. F. et al. (2024). Lost in the middle: How language models use long contexts. *TACL*.
- Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. *Quarterly Journal of Experimental Psychology*, 1(1), 6--21.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134--140.
- Norman, D. A., & Shallice, T. (1986). Attention to action: Willed and automatic control of behavior. *Consciousness and Self-Regulation*, 4, 1--18.
- Posner, M. I., & Cohen, Y. (1984). Components of visual orienting. *Attention and Performance X*, 531--556.
- Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. *Annual Review of Neuroscience*, 13, 25--42.
- Pylyshyn, Z. W., & Storm, R. W. (1988). Tracking multiple independent targets: Evidence for a parallel tracking mechanism. *Spatial Vision*, 3(3), 179--197.
- Raymond, J. E., Shapiro, K. L., & Arnell, K. M. (1992). Temporary suppression of visual processing in an RSVP task: An attentional blink? *Journal of Experimental Psychology: HPP*, 18(3), 849--860.
- Reuel, A. et al. (2024). BetterBench: Assessing AI benchmarks. *NeurIPS 2024.*
- Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst: Sustained inattentional blindness for dynamic events. *Perception*, 28(9), 1059--1074.
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643--662.
- Wang, C., & Sun, J. V. (2025). Unable to forget: Proactive interference reveals working memory limits in LLMs beyond context length (PI-LLM). *ICML 2025 Workshop on Long Context Foundation Models*. arXiv:2506.08184.
