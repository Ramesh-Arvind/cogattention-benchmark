# CogAttention: A Three-Tier Decomposition of Attention in Frontier Models

**Ramesh Arvind · TU Chemnitz**

### Problem Statement

We adapted the gorilla experiment (Simons & Chabris, 1999) and 15 other classical attention tasks for frontier models. Three of seven could not switch classification rules mid-task. Two missed embedded anomalies entirely. None could reliably name ink colors in conflicting color words. The same models passed classical distractor-filtering and interference-resistance tasks. These patterns are not visible in aggregate benchmark scores or in retrieval-focused long-context tests like NIAH or RULER.

CogAttention measures attention as DeepMind's cognitive taxonomy (Burnell et al., 2026, §7.3) defines it: three separable faculties: Attention Capacity, Selective Attention (with sub-abilities for Sustained focus, Perceptual Inhibition, and Shifting), and Stimulus-Driven Attention. Across 16 task types and 860 items, we find a 21-point spread across models, human-model profile inversions on specific sub-abilities, and cross-benchmark differences consistent with the framework's separability claim. Frontier models with near-identical aggregate scores can have opposite cognitive profiles.

### Alignment to the DeepMind Cognitive Framework

Under the framework's three-tier hierarchy (§7.3):

**Attention Capacity (§7.3.1).** Thread Tracking (Pylyshyn & Storm, 1988): N agents swap items; the model reports final holdings. Proactive Interference (Wang & Sun, 2025): repeated value updates with only the final value as target. Attentional Blink (Raymond et al., 1992): identify two targets in rapid serial presentation.

**Selective Attention / Attentional Control (§7.3.2).** This faculty is goal-directed, top-down focus, and the framework decomposes it into three sub-abilities, each of which we measure directly:

- *Sustained Attention*: Vigilance Probe (Mackworth, 1948), Stream Segregation (Cherry, 1953), Context Dilution (50K+ tokens), Semantic NIAH (zero lexical overlap), Multihop.
- *Perceptual Inhibition*: Distractor Filtering, Semantic Stroop (Stroop, 1935), Flanker (Eriksen & Eriksen, 1974).
- *Shifting*: Rule Shift (Monsell, 2003), Inhibition of Return (Posner & Cohen, 1984).

**Stimulus-Driven Attention (§7.3.3).** Anomaly Detection (Simons & Chabris, 1999): embedded foreign language, code, or factual absurdity must be flagged during a distractor task.

**Multimodal.** Visual Stroop and Visual Inattentional Blindness probe Selective and Stimulus-Driven attention under visual-linguistic conflict.

### Dataset and Contamination Resistance

860 items: 560 text-only, 150 Visual Stroop, 150 Visual Inattentional. All items are procedurally generated from a seed with programmatic ground truth. Contamination resistance via combinatorial pools, canary strings, seed-based regeneration, zero lexical overlap, and private scoring logic held out of model contexts. Difficulty spans five tiers (Easy to Frontier) by varying tracked-item count, passage length, distractor density, and cue clarity.

### Methodology

Assertions use `assert_contains_regex` with task-specific scoring; multi-target tasks pass at ≥80% target identification (e.g., ≥4 of 5 tracked items).

**Composite Attention Score (CAS).** Following BetterBench (Reuel et al., 2024), we report Arithmetic CAS (compensatory weighted mean) and Geometric CAS (non-compensatory, penalizing uneven profiles).

**Statistics.** With 7 models officially benchmarked on Kaggle (extended to 12 in local validation), we use Classical Test Theory rather than IRT, for which our model pool is too small. We report per-task difficulty (proportion of models passing), task-level point-biserial discrimination against total CAS, bootstrapped 95% confidence intervals (10,000 resamples), Cohen's d effect sizes between adjacent models, and position-bias analysis for the U-shaped attention curves documented by Liu et al. (2024). Shifting errors are further classified as perseveration vs. random via an attentional-residue analysis.

### Human Baseline

25 participants contributed 207 responses; we anchor against Barzykowski et al. (2022) normative data (N=485) on Stroop, SART, and Flanker. Human CAS = 0.767 overall (0.943 Easy to 0.600 Hard). This gives a comparison profile across difficulty, not a single ceiling.

### Results: Seven Frontier Models, April 2026

| Model | CAS | Primary Failure Modes |
|---|---|---|
| DeepSeek-R1-0528 | 0.89 | Visual Stroop, Visual Inattentional |
| Claude Opus 4.6 | 0.89 | Visual Stroop, Visual Inattentional |
| Gemini 2.5 Flash | 0.84 | Visual Stroop, Visual Inattentional |
| Claude Sonnet 4.5 | 0.84 | Shifting, Visual |
| GPT-OSS-20B | 0.84 | Visual Stroop, Visual Inattentional |
| Qwen3-Next-80B | 0.74 | Shifting, Anomaly, Visual |
| Gemma-3-27B | 0.68 | Capacity, Shifting, Anomaly, Visual |

Task-level r_pb shows 4 discriminating tasks (shifting, anomaly, capacity, blink), with the remaining 10 text-only tasks at ceiling and 2 visual tasks at floor. This saturation pattern is why sub-faculty decomposition surfaces differences that a single aggregate score hides.

### Five findings that aggregate scores obscure

**1. Shifting is the strongest discriminator within Selective Attention.** Three of seven models fail rule-shift (point-biserial r_pb = 0.86 against total CAS). Of these errors, 60–70% are classified as perseveration: traces show continued application of the prior classification rule rather than random noise, consistent with goal-set reconfiguration failure (Monsell, 2003).

**2. Attention Capacity separates only the bottom tier.** Gemma-3-27B fails capacity tasks that all other models pass (r_pb = 0.73), while top models are at ceiling. Capacity acts as a floor constraint, not a scaling axis, at the frontier.

**3. Anomaly Detection (Stimulus-Driven) separates tiers not captured by Selective metrics.** Qwen3-Next and Gemma fail (r_pb = 0.88, the highest in the benchmark). Humans score 0.676; the weakest models approach floor. A model that misses an embedded anomaly has no signal that something unexpected occurred, which is relevant to downstream metacognitive monitoring (Fernandez-Duque et al., 2000).

**4. Vigilance degrades predictably with length.** All seven models show the classical vigilance decrement (Mackworth, 1948): detection rates drop monotonically in later document quartiles. Long-context degradation follows a pattern also documented in humans.

**5. Human-model profile inversion.** Humans outperform every tested model on Anomaly Detection and Stroop but underperform on Proactive Interference and sequential recall. This is not a gap in overall ability; it is an inversion. The sub-abilities where humans do best are the ones where frontier models do worst.

### Cross-Track Evidence

Cognitive science has long treated attention as an upstream gate on downstream processing (Posner & Petersen, 1990), a hypothesis our cross-track comparisons can test. Cross-referencing Agrawal's *Executive Functions: Cognitive Control Suite* (Kaggle Benchmarks, 2026):

- **DeepSeek-R1 tops CogAttention (0.89) yet scores near-floor on executive inhibitory control in Agrawal's suite.** §7.3 and §7.4 behave independently here.
- **Gemma-3 fails Shifting here and cognitive flexibility in Agrawal's Executive Functions tasks.** Same family, same deficit across two independent benchmarks, consistent with Norman & Shallice (1986).

Two benchmarks grounded in the same taxonomy produce a cognitive profile, not a single score.

### Limitations

Our 7-model (12 local) pool supports CTT descriptive analysis but defers IRT to a larger pool. Visual Stroop in VLMs mixes perceptual encoding with interference; we flag this where reported. The 25-participant pool validates construct, not population norms; we anchor to Barzykowski et al. (2022).

### What a sub-faculty decomposition surfaces

Standard long-context benchmarks measure retrieval across length. CogAttention decomposes attention into sub-faculties and surfaces three results a single score hides: (1) models with identical aggregate scores have opposite profiles (Sonnet 4.5 and GPT-OSS-20B both score 0.84 but fail different sub-abilities); (2) failure modes are systematic (perseveration dominates shifting errors, vigilance decrement is monotonic); (3) cross-track comparison shows the framework's sub-faculties coming apart in behavior.

Sub-scores are diagnostic tools. For labs, two concrete uses: (a) regression-test a model family across releases (did shifting degrade from v3 to v4?); (b) evaluate architectural ablations (does a new KV-cache design improve sustained attention without breaking anomaly detection?). For the framework, they give behavioral evidence that its sub-faculty categories come apart in practice.

**Full benchmark (19 notebooks, 860 items), stimuli, human baseline data, and analysis code:** github.com/Ramesh-Arvind/cogattention-benchmark

### References

Agrawal, N. (2026). *Executive Functions: Cognitive Control Suite.* Kaggle Benchmarks.
Barzykowski, K. et al. (2022). Cognitive inhibition behavioral tasks: online and laboratory data. *Data in Brief*, 43, 108398.
Burnell, R., Kelly, O. et al. (2026). *Measuring Progress Toward AGI: A Cognitive Framework.* Google DeepMind.
Cherry, E.C. (1953). Some experiments on the recognition of speech. *JASA*, 25(5), 975–979.
Eriksen, B.A. & Eriksen, C.W. (1974). Effects of noise letters. *Perception & Psychophysics*, 16(1), 143–149.
Fernandez-Duque, D. et al. (2000). Executive attention and metacognitive regulation. *Consciousness and Cognition*, 9(2), 288–307.
Liu, N.F. et al. (2024). Lost in the middle. *TACL*.
Mackworth, N.H. (1948). The breakdown of vigilance. *QJEP*, 1(1), 6–21.
Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134–140.
Norman, D.A. & Shallice, T. (1986). Attention to action. *Consciousness and Self-Regulation*, 4.
Posner, M.I. & Cohen, Y. (1984). Components of visual orienting. *Attention and Performance X*.
Posner, M.I. & Petersen, S.E. (1990). The attention system of the human brain. *Annual Review of Neuroscience*, 13, 25–42.
Pylyshyn, Z.W. & Storm, R.W. (1988). Tracking multiple independent targets. *Spatial Vision*, 3(3), 179–197.
Raymond, J.E. et al. (1992). Temporary suppression of visual processing. *JEP:HPP*, 18(3), 849–860.
Reuel, A. et al. (2024). BetterBench: Assessing AI benchmarks. *NeurIPS 2024.*
Simons, D.J. & Chabris, C.F. (1999). Gorillas in our midst. *Perception*, 28(9), 1059–1074.
Stroop, J.R. (1935). Studies of interference. *JEP*, 18(6), 643–662.
Wang, C. & Sun, J.V. (2025). Unable to forget (PI-LLM). *ICML ICFM Workshop.* arXiv:2506.08184.
