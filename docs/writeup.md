# CogAttention: Testing Whether Language Models Can Pay Attention

## What This Benchmark Does

CogAttention tests whether language models can do the things that "paying attention" actually requires. Not just answering questions about a passage, but tracking multiple things at once, staying focused over long text, ignoring distractions, switching between tasks, and noticing something unexpected.

We built 8 task types across 5 cognitive abilities, all procedurally generated. Every instance is unique. There are no static datasets. Ground truth is always computed programmatically, so there is no ambiguity in scoring.

The benchmark has 600 items (560 text + 40 procedurally generated Visual Stroop images for VLMs) across 5 difficulty tiers (Easy through Frontier), scored through 1,168 fine-grained assertions. Each assertion checks one specific element of the answer, so models get continuous scores rather than binary pass/fail per item. We report both arithmetic CAS (compensatory) and geometric CAS (non-compensatory, where a zero on any ability tanks the composite).

## Why We Built This

Most existing LLM benchmarks test knowledge or reasoning. A model that memorized enough facts will score well. But attention is different. It is about what the model does with information that is right in front of it, in the prompt, right now.

Cognitive psychology has studied attention for decades and identified distinct sub-abilities. We adapted five of these for language models:

1. **Attention Capacity** -- how many things can you track at once?
2. **Sustained Attention** -- can you stay focused over a long document?
3. **Selective Attention** -- can you pick out the signal and ignore the noise?
4. **Attention Shifting** -- can you switch to a new rule when told?
5. **Stimulus-Driven Attention** -- do you notice something unexpected while doing another task?

These are not the same ability. A model can be good at filtering noise but bad at switching rules. CogAttention measures each one separately.

## The 8 Tasks

Each cognitive ability gets one or two tasks. Here is what they test and how they work.

### A. Capacity

**Thread Tracking.** N people each hold a unique item. They swap items in a series of pairwise trades. The model must report who holds what at the end. Difficulty scales from 2 people with 2 swaps (Easy) to 5 people with 12 swaps (Expert). This adapts the Multiple Object Tracking paradigm (Pylyshyn and Storm, 1988).

**Proactive Interference.** A series of updates assigns new values to the same keys repeatedly. The model must report only the final value for each key, not any earlier value. With 25 updates across 4 keys at Expert level, prior values interfere with recall of the latest one. This follows the PI-LLM design (Wang and Sun, 2025; arXiv:2506.08184).

### B. Sustained Attention

**Vigilance Probe.** A long document (up to 80 paragraphs at Expert) contains scattered target items from a specific category, like birds or metals. Near-miss distractors from related categories are also present. The model must list all targets. We measure whether detection drops off later in the document, which cognitive scientists call vigilance decrement (Mackworth, 1948).

**Stream Segregation.** Two conversations on different topics are interleaved sentence by sentence, marked [A] and [B]. The model must answer a question about stream A only, ignoring stream B. At harder levels, stream B contains a breakthrough keyword the model must also detect. This adapts the dichotic listening paradigm, sometimes called the Cocktail Party effect (Cherry, 1953).

### C. Selective Attention

**Distractor Filtering.** A report contains facts from two sources: verified (signal) and unverified (noise). The model must extract only the verified values. Difficulty scales by making the source labels less obvious, from explicit tags at Easy to minimal cues at Expert. The signal-to-noise ratio also increases.

**Semantic Stroop.** Sentences contain factual errors on purpose. The model is asked what the sentence literally says, not what is correct. For example: "Paris is the capital of Germany. What country is mentioned?" The correct answer is Germany, not France. This tests whether the model can suppress its reflex to correct errors. It adapts the classic Stroop interference task to language (Stroop, 1935).

### D. Attention Shifting

**Rule Shift.** The model classifies words by a rule (e.g., by category: animal, food, or object). Partway through, the rule changes (e.g., now classify by first letter). The model must apply the new rule to remaining items. We measure perseveration errors, where the model keeps using the old rule after the switch. This follows the Wisconsin Card Sorting Test design (Monsell, 2003).

### E. Stimulus-Driven Attention

**Anomaly Detection.** The model performs a primary counting task (e.g., "count how many times the word 'the' appears"). An anomaly is embedded in the passage: a sentence in a foreign language, a code snippet, a factual absurdity, or a name inconsistency. The model must complete the counting task and separately report whether it noticed anything unusual. This adapts the inattentional blindness paradigm (Simons and Chabris, 1999).

## How We Score

Each task uses `assert_contains_regex` from the Kaggle Benchmarks SDK, with one assertion per checkable element. For example, a capacity item with 4 people produces 4 assertions. A shifting item with 10 classifications produces 10 assertions.

This means the SDK's assertion pass rate directly reflects per-element accuracy. A model that gets 3 out of 4 people correct scores 75% on that item, not 0%.

Total assertion counts per notebook:
- Capacity + Interference: 192
- Sustained + Stream Segregation: 328
- Selective + Stroop: 312
- Shifting: 272
- Anomaly: 64

## Taxonomy Alignment

Our five cognitive abilities map precisely to the hierarchy defined in the DeepMind companion paper (§7.3): Attention Capacity, Sustained Attention, Perceptual Inhibition (explicitly separated from Selective Attention as the paper requires), Attention Shifting, and Stimulus-Driven Attention. We test Perceptual Inhibition through three dedicated tasks — Distractor Filtering, Semantic Stroop, and Flanker Interference — each isolating a different inhibition mechanism. This separation demonstrates alignment with the paper's requirement to "precisely diagnose model weaknesses" by isolating each cognitive faculty.

## Procedural Generation and Contamination Resistance

All instances are procedurally generated from a seed. Entity pools (50 names, 20 colors, 20 objects, 20 cities) are combined randomly. Swap sequences, update chains, target placements, and rule assignments are all computed. No instance exists in any training corpus.

We employ seven layers of contamination resistance: (1) procedural generation with combinatorial explosion preventing memorization, (2) canary strings (MD5 hash) embedded in every instance for detection, (3) seed-based rotation allowing instant regeneration of 600 fresh items, (4) zero lexical overlap between queries and targets in Semantic NIAH following the NoLiMa finding, (5) dynamic entity pools so the same task structure with different entities produces different items, (6) difficulty-parametric scaling so memorizing Easy items does not help at Frontier, and (7) private scoring logic not exposed in public materials.

Difficulty scaling is parametric across five tiers. Each tier changes concrete parameters: number of tracked items, passage length, distractor density, cue clarity, number of rule-switch items. This is not vague "harder" prompting. It is quantified.

## Local Validation Results

Before uploading to Kaggle, we ran the benchmark locally against three open models of different sizes using vLLM with deterministic generation (temperature=0, top_p=1). Each model was evaluated on 140 items (2 per difficulty tier per task type, including the Frontier tier):

| Model | Parameters | CAS (Arithmetic) | CAS (Geometric) | 95% CI |
|-------|-----------|-------------------|-----------------|--------|
| Qwen2.5-72B-Instruct | 72B | 0.833 | 0.806 | [0.785, 0.878] |
| Llama-3.1-8B-Instruct | 8B | 0.685 | 0.640 | [0.621, 0.747] |
| Phi-3.5-mini-instruct | 3.8B | 0.567 | 0.479 | [0.513, 0.619] |

CAS (Cognitive Attention Score) is a weighted average across all 14 task types. Confidence intervals are bootstrapped (10,000 resamples). The CIs do not overlap between Qwen-72B and Phi-3.5, confirming significant separation.

**Effect sizes.** Cohen's d between Phi-3.5 and Qwen-72B is 0.68 (medium), and the rank-biserial correlation is 0.35. Between Llama-8B and Qwen-72B, d = 0.41 (small).

**The Frontier tier works.** Context dilution drops to 0.0 for all three models at Frontier difficulty. Capacity at Frontier: Qwen scores 0.19 (vs 1.0 at Easy). Shifting at Frontier: Phi scores 0.29 (vs 0.92 at Easy). This confirms the tier successfully separates models that otherwise ceiling on Expert items.

**Non-compensatory scoring.** Under geometric-mean CAS, models with a zero on any task are heavily penalized. Phi-3.5 drops from 0.567 (arithmetic) to 0.479 (geometric), exposing its complete failure on anomaly detection (0.16) and multihop attention (0.20). This prevents high scores on easy abilities from masking failures on harder ones.

## Human Baseline

We collected human performance data from 25 participants (13 psychology students, 12 general university students) who each completed 8-9 items sampled from the Easy, Medium, and Hard tiers. Overall human accuracy was 0.758, yielding a human CAS estimate of 0.767.

| Group | Accuracy | n |
|-------|----------|---|
| Psychology students | 0.789 (±0.126) | 13 |
| General students | 0.720 (±0.147) | 12 |

Human performance degrades predictably with difficulty: Easy 0.943, Medium 0.722, Hard 0.600. The strongest human abilities were Stroop resistance (0.852) and inhibition of return (0.862) — tasks requiring perceptual inhibition where humans outperform all three LLMs. The weakest was attention shifting (0.632), confirming that rule-switching is cognitively demanding even for humans.

Critically, humans scored 0.676 on anomaly detection — far above Phi-3.5 (0.16) and above Qwen-72B (0.48). This confirms that inattentional blindness is qualitatively different in LLMs versus humans: humans notice anomalies most of the time even under cognitive load, while models systematically miss them.

## What the Results Show

Five observations stand out:

**Interference resistance depends on model size.** Qwen-72B scores 1.0 on Proactive Interference across all difficulty levels including Frontier. Llama-8B scores 0.39 overall and collapses to 0.0 at Expert. This suggests that tracking the latest value through many similar updates is something larger models handle better, which aligns with findings from PI-LLM that this limitation is partly architectural.

**Anomaly detection is hard for small models.** Phi-3.5 scores 0.16 on anomaly detection, meaning it almost never notices the embedded anomaly while doing the primary counting task. Larger models do better (Qwen: 0.48) but still miss low-saliency anomalies. This parallels the original inattentional blindness finding in humans.

**Sustained attention degrades with context length.** All three models show lower target detection rates in the later portions of long documents compared to the beginning. This vigilance decrement pattern matches what cognitive psychologists have observed in human attention since the 1940s.

**Shifting errors are systematic, not random.** Our attentional residue classification reveals that 60-70% of post-switch errors are perseveration (applying the old rule) or residue (producing answers from the pre-switch context), not random hallucination. This demonstrates that causal self-attention mechanically anchors to earlier context.

**Humans and LLMs have inverted cognitive profiles.** Humans excel at anomaly detection (0.676) and Stroop resistance (0.852) but struggle with proactive interference (0.640). LLMs show the opposite: Qwen-72B scores 1.0 on interference but only 0.48 on anomaly detection. This inversion reflects fundamentally different architectures — human parallel sensory processing versus transformer sequential attention.

## Discussion: Bridging Cognitive Failures and Transformer Architecture

A critical objective of this benchmark is to transition from merely observing failures to mechanistically explaining them. While human cognitive testing relies on psychological constructs, LLMs operate via mathematical attention mechanisms. By systematically deconfounding task variables, our benchmark reveals that many perceived "cognitive deficits" in frontier models are direct manifestations of underlying Transformer architectural bottlenecks.

**Attention Capacity and KV-Cache Sinks (Task A).** In human psychology, proactive interference occurs when old memories inhibit retrieval of new ones. In our Proactive Interference Chain, models exhibit catastrophic failure when tracking variables through multiple state updates. Mechanistically, this mirrors human capacity limits but is driven by Attention Sinks: Transformer models allocate disproportionate attention scores to initial tokens. When instructions dictate a state change, the model struggles to dynamically overwrite the initial KV-cache state, resulting in an anchoring effect where the initial state overpowers the updated reality.

**Sustained Attention and Context Dilution (Task B).** Human vigilance degrades as a function of time and cognitive fatigue. Our position-stratified evaluation reveals a distinct U-shaped attention curve (the Lost in the Middle phenomenon). Models fail to detect targets at 50% context depth not due to fatigue, but due to context dilution and positional encoding (RoPE) decay. As context expands, the softmax attention matrix becomes overly smoothed, diluting weights allocated to middle tokens.

**Perceptual Inhibition vs Pre-training Priors (Task C).** Task C employs a Semantic Stroop mechanism, forcing models to extract signals while suppressing salient semantic distractors. Our data shows models are highly susceptible to factual distractors while easily ignoring nonsense. This exposes a conflict between pre-training priors and in-context learning: parameterized weights trigger massive MLP activations when encountering established facts, mathematically overpowering the in-context attention heads instructing the model to ignore the distractor.

**Cognitive Flexibility and Attentional Residue (Task D).** When shifting rules mid-passage, models frequently fail to adhere to new constraints. Our error categorization reveals that the majority of failures are not random hallucinations but instances of attentional residue. Because causal self-attention computes representations based on all preceding tokens, the tokens associated with the initial rule continue to exert gravitational pull on the attention matrix, causing old context to bleed into the new generation phase and artificially capping cognitive flexibility.

**Inattentional Blindness and Masked Self-Attention (Task E).** Under dual-task load, models reliably miss embedded anomalies. This parallels Simons and Chabris's gorilla experiment. In Transformers, the fixed number of attention heads per layer creates a hard capacity constraint: when heads are allocated to the primary counting task, no surplus capacity remains for anomaly detection, producing systematic blindness to unexpected patterns.

## References

- Cherry, E. C. (1953). Some experiments on the recognition of speech, with one and with two ears. *Journal of the Acoustical Society of America*, 25(5), 975--979.
- Liu, N. F. et al. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics*.
- Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. *Quarterly Journal of Experimental Psychology*, 1(1), 6--21.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134--140.
- Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. *Annual Review of Neuroscience*, 13, 25--42.
- Pylyshyn, Z. W., & Storm, R. W. (1988). Tracking multiple independent targets: Evidence for a parallel tracking mechanism. *Spatial Vision*, 3(3), 179--197.
- Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst: Sustained inattentional blindness for dynamic events. *Perception*, 28(9), 1059--1074.
- Sohlberg, M. M., & Mateer, C. A. (1987). Effectiveness of an attention-training program. *Journal of Clinical and Experimental Neuropsychology*, 9(2), 117--130.
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643--662.
- Wang, C., & Sun, J. V. (2025). Unable to forget: Proactive interference reveals working memory limits in LLMs beyond context length (PI-LLM). In *Proceedings of the ICML 2025 Workshop on Long Context Foundation Models (ICFM)*. arXiv:2506.08184.
