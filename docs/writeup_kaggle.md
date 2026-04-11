### CogAttention

### Ramesh Arvind

### Problem Statement

We gave 7 frontier AI models the cognitive equivalent of the gorilla experiment. Three could not switch classification rules mid-task. Two missed embedded anomalies entirely. None could name ink colors in conflicting color words or detect an unexpected shape while counting. Meanwhile, humans outperform every frontier model in anomaly detection and Stroop resistance, but LLMs dominate in interference resistance and sequential recall. The cognitive profiles are inverted.

CogAttention decomposes "attention" into 5 sub-abilities from cognitive psychology: Capacity (tracking multiple things), Sustained (staying focused over long stretches of text), Selective (filtering out noise), Shifting (switching rules), and Stimulus-Driven (noticing the unexpected). It produces a 21-point spread across 7 frontier models (DeepSeek-R1 at 0.89, Gemma-3-27B at 0.68). These are not the same ability - a model good at filtering noise can be bad at switching rules. CogAttention measures each one separately and reveals failure modes that raw accuracy benchmarks cannot see.

### Task & Benchmark Construction

We built 16 task types across 5 cognitive abilities, organized into 19 Kaggle benchmark tasks across 17 notebooks (one task per notebook). All tasks are procedurally generated from a seed with programmatic ground truth - no static datasets, no ambiguity.

**Capacity.** Thread Tracking: N people swap items; report who holds what (Pylyshyn & Storm, 1988). Proactive Interference: repeated value updates; report only the final value (Wang & Sun, 2025). Attentional Blink: identify two targets in rapid serial presentation (Raymond et al., 1992).

**Sustained Attention.** Vigilance Probe: find targets in long documents with near-miss distractors (Mackworth, 1948). Stream Segregation: follow one interleaved conversation (Cherry, 1953). Context Dilution: extract a fact from 50K+ token passages. Semantic NIAH: needle-in-a-haystack with zero keyword overlap. Multihop: chain distributed facts across long context.

**Selective Attention.** Distractor Filtering: extract verified facts ignoring unverified noise. Semantic Stroop: report literal meaning despite factual errors (Stroop, 1935). Flanker: extract a target surrounded by conflicting values (Eriksen & Eriksen, 1974).

**Shifting.** Rule Shift: classify by one rule, then switch mid-task (Monsell, 2003). Inhibition of Return: answer questions about a modified passage, suppressing prior answers (Posner & Cohen, 1984).

**Stimulus-Driven.** Anomaly Detection: perform a counting task while an embedded anomaly (foreign language, code, factual absurdity) must be reported (Simons & Chabris, 1999).

**Multimodal.** Visual Stroop: name ink color of a color word in conflicting ink. Visual Inattentional Blindness: count shapes while detecting an unexpected stimulus.

### Dataset

860 items: 560 text-only + 150 Visual Stroop images + 150 Visual Inattentional scenes. Procedurally generated with 7 contamination resistance layers: combinatorial generation, canary strings, seed-based regeneration, zero lexical overlap, dynamic entity pools, parametric difficulty scaling, private scoring logic. Difficulty spans 5 tiers (Easy→Frontier), varying tracked items, passage length, distractor density, and cue clarity.

### Technical Details

**Assertions.** Built on `assert_contains_regex`. Multi-target tasks use aggregate assertions with minimum thresholds (e.g., tracking 5 people passes if ≥4 correct). Single-target tasks use direct match. Visual Stroop uses color-variant-aware matching.

**Composite Scoring (CAS).** Two modes: Arithmetic CAS (compensatory weighted mean) and Geometric CAS (non-compensatory - a zero on any ability tanks the composite). Dual scoring follows BetterBench recommendations.

**Statistical Rigor.** Following BetterBench (NeurIPS 2024): bootstrapped 95% CIs (10,000 resamples), Cohen's d effect sizes, 2PL IRT, position bias analysis detecting U-shaped attention curves (Liu et al. 2024), and attentional residue classification showing 60-70% of shifting errors are systematic perseveration.

### Results, Insights, and Conclusions

**Kaggle Benchmarks platform** - 7 frontier models evaluated across all 16 task types (April 2026):

| Model | Score | Key Failures |
|-------|-------|-------------|
| DeepSeek-R1-0528 | **0.89** | visual stroop, visual inattentional |
| Claude Opus 4.6 | **0.89** | visual stroop, visual inattentional |
| Gemini 2.5 Flash | 0.84 | visual stroop, visual inattentional |
| Claude Sonnet 4.5 | 0.84 | shifting, visual tasks |
| GPT-OSS-20B | 0.84 | visual stroop, visual inattentional |
| Qwen3-Next-80B | 0.74 | shifting, anomaly, visual tasks |
| Gemma-3-27B | **0.68** | capacity, shifting, anomaly, visual tasks |

21-point spread (DeepSeek/Opus 0.89 to Gemma 0.68). Key discriminators: shifting (3/7 fail), anomaly (2/7 fail). Visual tasks unsolved by all. Top 5 models pass all text-only tasks; separation comes from shifting, anomaly, and capacity. Local validation confirms non-overlapping 95% CIs (Cohen's d = 0.68) and Geometric CAS exposes hidden weaknesses (Phi-3.5-mini drops 0.567 → 0.479 from near-zero anomaly detection).

**Human baseline** from 25 participants (207 responses): overall CAS 0.767, degrading predictably from Easy (0.943) to Hard (0.600).

**Five key findings:**

1. **Shifting is the strongest discriminator.** 3/7 frontier models fail rule-shift - 60-70% of errors are perseveration, not random.

2. **Anomaly detection separates tiers.** Qwen3-Next and Gemma fail; humans score 0.676 vs. Phi-3.5 at 0.16.

3. **Capacity separates the bottom tier.** Gemma fails capacity tasks that all other models pass - tracking multiple objects under load is a fundamental bottleneck for smaller models.

4. **Vigilance degrades with length.** All models show lower detection in later document portions - the vigilance decrement pattern from Mackworth (1948).

5. **Humans and LLMs have inverted profiles.** Humans excel at anomaly detection and Stroop resistance but struggle with interference. All frontier models show the opposite - confirming the inversion is a universal Transformer trait.

**Cross-track implications: attention as the upstream primitive.** Cognitive science treats attention as the gateway that gates all downstream processing (Posner & Petersen, 1990). Cross-referencing with the Executive Functions: Cognitive Control Suite benchmark (Agrawal, 2026; 16 tasks, 27 models) provides empirical support:

- *Shifting → Cognitive flexibility.* Gemma-3-27B fails our rule-shift task; Gemma-3-1B scores 0.00 on cognitive flexibility in the executive benchmark - the same family, same deficit, two independent benchmarks, exactly the cascade Norman & Shallice (1986) predict.
- *Anomaly detection → Metacognition.* A model that misses an embedded anomaly has no signal that something unexpected occurred. Without that signal, it cannot flag uncertainty - it will answer confidently on incomplete evidence, the core metacognitive monitoring failure (Fernandez-Duque et al., 2000).
- *Attention ≠ Executive control.* DeepSeek-R1 tops our benchmark (0.89) yet scores 0.00 on executive inhibitory control - confirming separable faculties, not a single "intelligence" factor.

CogAttention sub-scores thus serve as diagnostic primitives for the broader AGI framework: they predict specific cross-track failures while remaining dissociable from overall performance.

### Gradient of Performance & Discriminatory Power

CogAttention produces a meaningful gradient at every level:

**Across models.** Kaggle: 21-point range (DeepSeek/Opus 0.89 to Gemma 0.68). Local validation: 27-point range with non-overlapping 95% CIs. Geometric CAS amplifies separation.

**Across difficulty.** Monotonic degradation Easy→Frontier for every model (Qwen: 0.887→0.637; Phi: 0.740→0.405). Easy validates the construct; Frontier separates models that otherwise ceiling.

**Across tasks.** Qwen scores 1.0 on proactive interference but 0.48 on anomaly - a 52-point gap within one model. Ceiling tasks (Flanker, NIAH, Stroop) confirm solved abilities; discrimination comes from capacity, anomaly, and shifting.

**What existing benchmarks cannot show:** Standard NIAH tests retrieval over length. CogAttention decomposes attention into 5 sub-abilities with inverted human-model profiles, letting researchers target architectural bottlenecks.

### Implications for AGI Measurement

Attention is not one ability - it is five, and models have inverted profiles compared to humans. Three forward-looking claims:

1. **Fixing shifting and anomaly detection is prerequisite to reliable metacognition and executive control.** A model that cannot disengage from a prior rule or misses the unexpected cannot calibrate confidence or adjust plans - no downstream training compensates for attention-stage information loss.

2. **Labs can operationalize CogAttention:** (a) regression-test a model family across releases - did shifting degrade from v3 to v4? (b) evaluate architectural changes - does a new KV cache design improve sustained attention without breaking anomaly detection? (c) feed sub-scores into an internal AGI dashboard alongside metacognition and executive benchmarks to build composite cognitive profiles.

3. **Frontier models are at floor on visual and stimulus-driven attention** - visual Stroop and inattentional blindness remain unsolved by all 7 models. These will most move the needle toward robust AGI.

Full benchmark suite (19 notebooks, 860 items), figures, human baseline data, and analysis code: github.com/Ramesh-Arvind/cogattention-benchmark

### Organizational Affiliations

Independent researcher.

### References & Citations

- Agrawal, N. (2026). Executive Functions: Cognitive Control Suite. *Kaggle Benchmarks*. kaggle.com/benchmarks/naivedhyaagrawal/executive-functions-the-cognitive-control-suite
- Cherry, E.C. (1953). Some experiments on the recognition of speech. *JASA*, 25(5), 975-979.
- Eriksen, B.A. & Eriksen, C.W. (1974). Effects of noise letters. *Perception & Psychophysics*, 16(1), 143-149.
- Fernandez-Duque, D. et al. (2000). Executive attention and metacognitive regulation. *Consciousness and Cognition*, 9(2), 288-307.
- Liu, N.F. et al. (2024). Lost in the middle. *TACL*.
- Mackworth, N.H. (1948). The breakdown of vigilance. *QJEP*, 1(1), 6-21.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134-140.
- Norman, D.A. & Shallice, T. (1986). Attention to action. In *Consciousness and Self-Regulation* (Vol. 4).
- Posner, M.I. & Cohen, Y. (1984). Components of visual orienting. *Attention and Performance X*.
- Posner, M.I. & Petersen, S.E. (1990). The attention system of the human brain. *Annual Review of Neuroscience*, 13, 25-42.
- Pylyshyn, Z.W. & Storm, R.W. (1988). Tracking multiple independent targets. *Spatial Vision*, 3(3), 179-197.
- Raymond, J.E. et al. (1992). Temporary suppression of visual processing. *JEP:HPP*, 18(3), 849-860.
- Simons, D.J. & Chabris, C.F. (1999). Gorillas in our midst. *Perception*, 28(9), 1059-1074.
- Stroop, J.R. (1935). Studies of interference. *JEP*, 18(6), 643-662.
- Wang, C. & Sun, J.V. (2025). Unable to forget (PI-LLM). *ICML ICFM Workshop*. arXiv:2506.08184.
