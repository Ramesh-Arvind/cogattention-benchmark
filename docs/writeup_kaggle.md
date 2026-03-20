### CogAttention

### Ramesh Arvind

### Problem Statement

Most LLM benchmarks test knowledge or reasoning — a model that memorized enough facts scores well. But attention is different. It is about what the model does with information right in front of it. Cognitive psychology has studied attention for decades and identified distinct sub-abilities. We adapted five for language models: Attention Capacity (tracking multiple things), Sustained Attention (staying focused over long text), Selective Attention (filtering signal from noise), Attention Shifting (switching rules), and Stimulus-Driven Attention (noticing the unexpected). These are not the same ability — a model good at filtering noise can be bad at switching rules. CogAttention measures each one separately.

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

**Statistical Rigor.** We apply five layers of statistical analysis that 58% of existing benchmarks lack (per BetterBench, NeurIPS 2024):
- **Bootstrapped 95% confidence intervals** (10,000 resamples) on both CAS and per-task scores, confirming non-overlapping CIs between model tiers.
- **Effect sizes**: Cohen's d (0.68 medium between Phi-3.5 and Qwen-72B) and rank-biserial correlation (0.35), quantifying practical significance beyond p-values.
- **Power-law degradation coefficient** (δ): fits `E(m) ~ a·m^δ` to selective attention error rates as noise ratio increases, yielding a single number characterizing distractor vulnerability per model.
- **2PL Item Response Theory**: extracts difficulty and discrimination parameters per item, identifying which items best separate model abilities and flagging items with poor discrimination for removal.
- **Position bias analysis**: bins sustained attention items by target position (quintiles) to detect U-shaped attention curves (Lost in the Middle, Liu et al. 2024).
- **Attentional residue classification**: categorizes shifting errors as perseveration (old rule), residue (old context), or random — revealing that 60-70% of errors are systematic, not stochastic.

### Results, Insights, and Conclusions

**Local validation** against three open models (vLLM, temperature=0):

| Model | CAS (Arithmetic) | CAS (Geometric) | 95% CI |
|-------|-------------------|-----------------|--------|
| Qwen2.5-72B | 0.833 | 0.806 | [0.785, 0.878] |
| Llama-3.1-8B | 0.685 | 0.640 | [0.621, 0.747] |
| Phi-3.5-mini | 0.567 | 0.479 | [0.513, 0.619] |

Non-overlapping bootstrapped CIs confirm statistically significant separation between all model tiers. Cohen's d = 0.68 (medium effect) between Phi-3.5 and Qwen-72B; d = 0.41 (small) between Llama-8B and Qwen-72B. Geometric CAS reveals hidden weaknesses: Phi drops from 0.567 → 0.479 due to near-zero anomaly detection (0.16) and multihop (0.20).

**Human baseline** from 25 participants (13 psychology + 12 general students, 207 responses): overall accuracy 0.758, CAS estimate 0.767. Human performance degrades predictably: Easy 0.943, Medium 0.722, Hard 0.600.

**Five key findings:**

1. **Interference resistance scales with size.** Qwen-72B scores 1.0 on proactive interference; Llama-8B collapses to 0.0 at Expert. This reflects KV-cache attention sinks anchoring to initial token states.

2. **Anomaly detection is systematically hard.** Phi-3.5 scores 0.16, meaning it almost never notices embedded anomalies. Humans score 0.676. Fixed attention head counts create a hard capacity constraint under dual-task load.

3. **Vigilance degrades with context length.** All models show lower detection in later portions of long documents — the same vigilance decrement pattern observed in humans since Mackworth (1948), driven by softmax attention dilution and RoPE decay.

4. **Shifting errors are systematic.** 60-70% of post-switch errors are perseveration or attentional residue, not random hallucination. Causal self-attention mechanically anchors to earlier context.

5. **Humans and LLMs have inverted profiles.** Humans excel at anomaly detection (0.676) and Stroop resistance (0.852) but struggle with interference (0.640). LLMs show the opposite. This inversion reflects fundamentally different architectures.

**Discriminatory power.** The Frontier tier produces meaningful separation: context dilution drops to 0.0 for all models; capacity at Frontier: Qwen 0.19 vs 1.0 at Easy. Geometric CAS penalizes weaknesses: Phi drops from 0.567→0.479.

Full benchmark suite, figures, human baseline data, and analysis code: github.com/Ramesh-Arvind/cogattention-benchmark

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
