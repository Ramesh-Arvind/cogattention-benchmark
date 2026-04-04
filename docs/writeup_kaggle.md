### CogAttention

### Ramesh Arvind

### Problem Statement

We gave 7 frontier AI models the cognitive equivalent of the gorilla experiment. Five failed to detect two targets in a rapid word stream. Three could not switch classification rules mid-task. Two missed embedded anomalies entirely. Meanwhile, humans outperform every frontier model on anomaly detection and Stroop resistance — but LLMs dominate on interference resistance and sequential recall. The cognitive profiles are inverted.

CogAttention is a benchmark that decomposes "attention" into 5 distinct sub-abilities from cognitive psychology: Attention Capacity (tracking multiple things), Sustained Attention (staying focused over long text), Selective Attention (filtering signal from noise), Attention Shifting (switching rules), and Stimulus-Driven Attention (noticing the unexpected). On the Kaggle platform, it produces a 21-point spread across 7 frontier models (DeepSeek-R1 at 0.895, Gemma-3-27B at 0.684). These are not the same ability — a model good at filtering noise can be bad at switching rules. CogAttention measures each one separately.

### Task & Benchmark Construction

We built 16 task types across 5 cognitive abilities, organized into 7 Kaggle benchmark tasks with 13 supporting notebooks. All tasks are procedurally generated from a seed with programmatic ground truth — no static datasets, no ambiguity.

**Capacity.** Thread Tracking: N people swap items; report who holds what (Pylyshyn & Storm, 1988). Proactive Interference: repeated value updates; report only the final value (Wang & Sun, 2025). Attentional Blink: identify two targets in rapid serial presentation (Raymond et al., 1992).

**Sustained Attention.** Vigilance Probe: find scattered targets in long documents with near-miss distractors (Mackworth, 1948). Stream Segregation: follow one interleaved conversation while ignoring another (Cherry, 1953). Context Dilution: extract a target fact from 50K+ token passages. Semantic NIAH: needle-in-a-haystack with zero keyword overlap. Multihop: chain distributed facts across long context.

**Selective Attention.** Distractor Filtering: extract verified facts while ignoring unverified noise. Semantic Stroop: report what a sentence literally says despite factual errors (Stroop, 1935). Flanker: extract a target surrounded by conflicting values (Eriksen & Eriksen, 1974).

**Shifting.** Rule Shift: classify words by one rule, then switch mid-task (Monsell, 2003). Inhibition of Return: answer questions about a modified passage, suppressing prior answers (Posner & Cohen, 1984).

**Stimulus-Driven.** Anomaly Detection: perform a counting task while an anomaly (foreign language, code, factual absurdity) is embedded; report it (Simons & Chabris, 1999).

**Multimodal.** Visual Stroop: name the ink color of a color word in conflicting ink. Visual Inattentional Blindness: count shapes while detecting an unexpected stimulus in a busy scene.

### Dataset

860 total items: 560 text-only + 150 Visual Stroop images + 150 Visual Inattentional scenes. All procedurally generated with 7 layers of contamination resistance: (1) combinatorial procedural generation, (2) canary strings, (3) seed-based regeneration, (4) zero lexical overlap in Semantic NIAH, (5) dynamic entity pools, (6) parametric difficulty scaling, (7) private scoring logic. Difficulty spans 5 tiers (Easy→Frontier), each changing concrete parameters: tracked items, passage length, distractor density, cue clarity.

### Technical Details

**Assertions.** Built on `assert_contains_regex` from the Kaggle Benchmarks SDK. Multi-target tasks use aggregate assertions with minimum thresholds (e.g., tracking 5 people passes if ≥4 correct), preventing stochastic formatting misses from invalidating correct responses. Single-target tasks (NIAH, dilution, anomaly) use direct match. Visual Stroop uses color-variant-aware matching (accepts "navy" for blue, "crimson" for red).

**Composite Scoring (CAS).** Two scoring modes: Arithmetic CAS (weighted mean — compensatory, a model can offset weakness in one ability with strength in another) and Geometric CAS (non-compensatory — a zero on any ability tanks the composite, exposing true weaknesses). This dual scoring follows BetterBench recommendations.

**Statistical Rigor.** Following BetterBench (NeurIPS 2024), we apply: bootstrapped 95% CIs (10,000 resamples) confirming non-overlapping intervals between models; Cohen's d and rank-biserial effect sizes; power-law degradation fits for selective attention; 2PL Item Response Theory for item discrimination; position bias analysis detecting U-shaped attention curves (Liu et al. 2024); and attentional residue classification showing 60-70% of shifting errors are systematic perseveration, not random.

### Results, Insights, and Conclusions

**Kaggle Benchmarks platform** — 7 frontier models evaluated across all 16 task types (April 2026):

| Model | Score | Key Failures |
|-------|-------|-------------|
| DeepSeek-R1-0528 | **0.895** | visual stroop, visual inattentional |
| Gemini 2.5 Flash | 0.842 | blink, visual tasks |
| Claude Opus 4.6 | 0.842 | blink, visual tasks |
| Claude Sonnet 4.5 | 0.789 | blink, shifting, visual tasks |
| GPT-OSS-20B | 0.778 | flanker, inhibition of return, visual tasks |
| Qwen3-Next-80B | 0.737 | blink, shifting, anomaly, visual tasks |
| Gemma-3-27B | **0.684** | blink, capacity, shifting, anomaly, visual tasks |

21-point spread across 7 frontier models. Key discriminators: shifting (3/7 fail), anomaly (2/7 fail), blink (5/7 fail). Visual tasks remain unsolved by all models.

**Local validation** against three open models (vLLM, temperature=0):

| Model | CAS (Arithmetic) | CAS (Geometric) | 95% CI |
|-------|-------------------|-----------------|--------|
| Qwen2.5-72B | 0.833 | 0.806 | [0.785, 0.878] |
| Llama-3.1-8B | 0.685 | 0.640 | [0.621, 0.747] |
| Phi-3.5-mini | 0.567 | 0.479 | [0.513, 0.619] |

Non-overlapping bootstrapped CIs confirm statistically significant separation between all model tiers. Cohen's d = 0.68 (medium effect) between Phi-3.5 and Qwen-72B; d = 0.41 (small) between Llama-8B and Qwen-72B. Geometric CAS reveals hidden weaknesses: Phi drops from 0.567 → 0.479 due to near-zero anomaly detection (0.16) and multihop (0.20).

**Human baseline** from 25 participants (13 psychology + 12 general students, 207 responses): overall accuracy 0.758, CAS estimate 0.767. Human performance degrades predictably: Easy 0.943, Medium 0.722, Hard 0.600.

**Five key findings:**

1. **Shifting is the strongest frontier discriminator.** 3 of 7 frontier models fail rule-shift (Claude Sonnet, Qwen3, Gemma) — 60-70% of errors are perseveration or attentional residue, not random. Causal self-attention mechanically anchors to earlier context.

2. **Anomaly detection separates model tiers.** Qwen3-Next and Gemma fail while larger models pass. Locally, Phi-3.5 scores 0.16; humans score 0.676. Fixed attention head counts create a hard capacity constraint under dual-task load.

3. **Attentional blink is nearly unsolved.** Only DeepSeek-R1 passes; all other frontier models fail. Temporal bottlenecks in sequential processing remain a fundamental challenge.

4. **Vigilance degrades with context length.** All models show lower detection in later portions of long documents — the same vigilance decrement pattern observed in humans since Mackworth (1948), driven by softmax attention dilution and RoPE decay.

5. **Humans and LLMs have inverted profiles.** Humans excel at anomaly detection (0.676) and Stroop resistance (0.852) but struggle with interference (0.640). All 7 frontier models pass interference and stroop, but fail on shifting, anomaly, and blink — confirming the inversion is a universal Transformer trait.

### Gradient of Performance & Discriminatory Power

A benchmark where all models score 100% or 0% reveals nothing. CogAttention produces a meaningful gradient at every level of analysis:

**Across frontier models (Kaggle platform).** Scores span a 21-point range (DeepSeek 0.895 to Gemma 0.684) across 7 frontier models, with every model receiving a unique score. Locally, CAS scores span a 27-point range (Qwen 0.833, Llama 0.685, Phi 0.567) with non-overlapping 95% CIs, confirming statistically significant separation. Cohen's d = 0.68 (medium effect) between the best and worst models. Geometric CAS amplifies the gradient further: Phi drops 15.6% (0.567→0.479) due to near-zero scores on anomaly detection and multihop, while Qwen drops only 3.2%.

**Across difficulty tiers (within models).** Every model shows monotonic degradation from Easy to Frontier:

| Tier | Qwen-72B | Llama-8B | Phi-3.5 |
|------|----------|----------|---------|
| Easy | 0.887 | 0.881 | 0.740 |
| Medium | 0.920 | 0.849 | 0.639 |
| Hard | 0.790 | 0.741 | 0.631 |
| Expert | 0.788 | 0.606 | 0.553 |
| Frontier | 0.637 | 0.493 | 0.405 |

No tier produces uniform 0% or 100%. The Easy tier validates the construct (models can do the task); the Frontier tier separates models that otherwise ceiling.

**Across tasks (diagnostic level).** The benchmark reveals distinct capability profiles, not a single "attention" score. Cross-task variance is high:

- Qwen scores 1.0 on proactive interference but 0.48 on anomaly detection — a 52-point gap within the same model.
- Llama scores 0.876 on selective attention but 0.273 on inhibition of return.
- The largest cross-model gap on a single task is proactive interference: Qwen 1.0 vs Llama 0.388 (0.612 gap).

**Ceiling tasks are intentional, not a flaw.** Flanker, Semantic NIAH, and Stroop hit 100% for Qwen — confirming these abilities are solved for frontier models. This matters: it establishes that attention is not uniformly hard; specific sub-abilities (capacity tracking, anomaly detection, context dilution) remain unsolved and are where the benchmark provides discriminatory signal.

**What this benchmark reveals that existing benchmarks cannot:** Standard NIAH and long-context benchmarks test one dimension (retrieval over length). CogAttention decomposes attention into 5 cognitive sub-abilities and shows models have inverted profiles compared to humans — strong where humans are weak (interference resistance, Stroop) and weak where humans are strong (anomaly detection, divided attention). This diagnostic granularity lets researchers target specific architectural bottlenecks rather than chasing a single composite score.

Full benchmark suite (19 notebooks, 860 items), figures, human baseline data, and analysis code: github.com/Ramesh-Arvind/cogattention-benchmark

### Organizational Affiliations

Independent researcher.

### References & Citations

- Cherry, E.C. (1953). Some experiments on the recognition of speech. *JASA*, 25(5), 975-979.
- Eriksen, B.A. & Eriksen, C.W. (1974). Effects of noise letters upon target identification. *Perception & Psychophysics*, 16(1), 143-149.
- Liu, N.F. et al. (2024). Lost in the middle. *TACL*.
- Mackworth, N.H. (1948). The breakdown of vigilance. *QJEP*, 1(1), 6-21.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134-140.
- Posner, M.I. & Cohen, Y. (1984). Components of visual orienting. *Attention and Performance X*.
- Pylyshyn, Z.W. & Storm, R.W. (1988). Tracking multiple independent targets. *Spatial Vision*, 3(3), 179-197.
- Raymond, J.E. et al. (1992). Temporary suppression of visual processing. *JEP:HPP*, 18(3), 849-860.
- Simons, D.J. & Chabris, C.F. (1999). Gorillas in our midst. *Perception*, 28(9), 1059-1074.
- Stroop, J.R. (1935). Studies of interference. *JEP*, 18(6), 643-662.
- Wang, C. & Sun, J.V. (2025). Unable to forget (PI-LLM). *ICML ICFM Workshop*. arXiv:2506.08184.
