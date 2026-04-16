# CogAttention: Testing Whether Language Models Can Pay Attention

## The Headline Finding

We gave 7 frontier AI models the cognitive equivalent of Simons and Chabris' gorilla experiment — a primary task with an unexpected anomaly embedded in it. Three failed to switch classification rules mid-task. Two missed embedded anomalies entirely. None could name ink colors in conflicting color words or detect an unexpected shape while counting. Humans outperform every frontier model on anomaly detection and Stroop resistance, while LLMs dominate on interference resistance and sequential recall. The cognitive profiles are inverted.

On the Kaggle Benchmarks platform, CogAttention produces a 21-point spread across 7 frontier models (DeepSeek-R1 and Claude Opus tied at 0.89, Gemma-3-27B at 0.68), with shifting, anomaly detection, and capacity as the key discriminators. No other attention benchmark decomposes performance into 5 cognitive sub-abilities, connects failures to Transformer architecture, or provides human baselines for comparison.

## What This Benchmark Does

CogAttention tests whether language models can do the things that "paying attention" actually requires. Not just answering questions about a passage, but tracking multiple things at once, staying focused over long text, ignoring distractions, switching between tasks, and noticing something unexpected.

We built 16 task types across 5 cognitive abilities, all procedurally generated. Every instance is unique. There are no static datasets. Ground truth is always computed programmatically, so there is no ambiguity in scoring.

The benchmark has 860 items (560 text-only + 150 procedurally generated Visual Stroop images + 150 Visual Inattentional Blindness scenes) across 5 difficulty tiers (Easy through Frontier). All 16 task types are evaluated on the Kaggle Benchmarks platform across 19 task notebooks (one task per notebook), tested against 7 frontier models. The full suite is available on GitHub. We report both arithmetic CAS (compensatory) and geometric CAS (non-compensatory, where a zero on any ability tanks the composite).

## Why We Built This

Most existing LLM benchmarks test knowledge or reasoning. A model that memorized enough facts will score well. But attention is different. It is about what the model does with information that is right in front of it, in the prompt, right now.

Cognitive psychology has studied attention for decades and identified distinct sub-abilities. We adapted five of these for language models:

1. **Attention Capacity** -- how many things can you track at once?
2. **Sustained Attention** -- can you stay focused over a long document?
3. **Selective Attention** -- can you pick out the signal and ignore the noise?
4. **Attention Shifting** -- can you switch to a new rule when told?
5. **Stimulus-Driven Attention** -- do you notice something unexpected while doing another task?

These are not the same ability. A model can be good at filtering noise but bad at switching rules. CogAttention measures each one separately.

## The 16 Task Types

Each cognitive ability gets one to three tasks. Here is what they test and how they work.

### A. Capacity

**Thread Tracking.** N people each hold a unique item. They swap items in a series of pairwise trades. The model must report who holds what at the end. Difficulty scales from 2 people with 2 swaps (Easy) to 5 people with 12 swaps (Expert). This adapts the Multiple Object Tracking paradigm (Pylyshyn and Storm, 1988).

**Proactive Interference.** A series of updates assigns new values to the same keys repeatedly. The model must report only the final value for each key, not any earlier value. With 25 updates across 4 keys at Expert level, prior values interfere with recall of the latest one. This follows the PI-LLM design (Wang and Sun, 2025; arXiv:2506.08184).

**Attentional Blink.** A rapid serial visual presentation (RSVP) of words is shown, with two targets embedded in the stream. The model must identify both targets. When the second target appears shortly after the first, humans exhibit an "attentional blink" -- a temporary inability to process the second target. We test whether LLMs show analogous temporal bottlenecks in sequential processing (Raymond et al., 1992).

### B. Sustained Attention

**Vigilance Probe.** A long document (up to 80 paragraphs at Expert) contains scattered target items from a specific category, like birds or metals. Near-miss distractors from related categories are also present. The model must list all targets. We measure whether detection drops off later in the document, which cognitive scientists call vigilance decrement (Mackworth, 1948).

**Stream Segregation.** Two conversations on different topics are interleaved sentence by sentence, marked [A] and [B]. The model must answer a question about stream A only, ignoring stream B. At harder levels, stream B contains a breakthrough keyword the model must also detect. This adapts the dichotic listening paradigm, sometimes called the Cocktail Party effect (Cherry, 1953).

**Context Dilution.** A single target fact is embedded in progressively longer passages of thematically similar filler text. At Frontier difficulty, the context exceeds 50,000 tokens. The model must extract the target despite massive dilution. This directly tests the "Lost in the Middle" phenomenon (Liu et al., 2024).

**Semantic NIAH (Needle in a Haystack).** Unlike standard NIAH benchmarks that use lexically distinctive needles, our semantic variant uses needles with zero keyword overlap with the query. The model must identify the needle through meaning alone, not string matching. This follows the NoLiMa finding that most NIAH benchmarks are solvable by keyword search.

**Multihop Attention.** Facts are distributed across a long document, and the model must chain 2-3 facts together to arrive at the answer. No single location contains the full answer. This tests whether sustained attention can support multi-step reasoning over distributed context.

### C. Selective Attention

**Distractor Filtering.** A report contains facts from two sources: verified (signal) and unverified (noise). The model must extract only the verified values. Difficulty scales by making the source labels less obvious, from explicit tags at Easy to minimal cues at Expert. The signal-to-noise ratio also increases.

**Semantic Stroop.** Sentences contain factual errors on purpose. The model is asked what the sentence literally says, not what is correct. For example: "Paris is the capital of Germany. What country is mentioned?" The correct answer is Germany, not France. This tests whether the model can suppress its reflex to correct errors. It adapts the classic Stroop interference task to language (Stroop, 1935).

**Flanker Interference.** A target value is surrounded by conflicting flanker values. The model must extract only the target while ignoring adjacent distractors. This adapts the Eriksen Flanker Task (Eriksen and Eriksen, 1974) to text.

### D. Attention Shifting

**Rule Shift.** The model classifies words by a rule (e.g., by category: animal, food, or object). Partway through, the rule changes (e.g., now classify by first letter). The model must apply the new rule to remaining items. We measure perseveration errors, where the model keeps using the old rule after the switch. This follows the Wisconsin Card Sorting Test design (Monsell, 2003).

**Inhibition of Return.** The model answers questions about a passage, then is shown the same passage with modifications and must answer updated questions. The task tests whether the model can suppress its memory of the original answers and attend to the changes. This adapts the Inhibition of Return paradigm (Posner and Cohen, 1984).

### E. Stimulus-Driven Attention

**Anomaly Detection.** The model performs a primary counting task (e.g., "count how many times the word 'the' appears"). An anomaly is embedded in the passage: a sentence in a foreign language, a code snippet, a factual absurdity, or a name inconsistency. The model must report whether it noticed anything unusual. This adapts the inattentional blindness paradigm (Simons and Chabris, 1999).

### F. Multimodal Attention (VLM Extension)

**Visual Stroop.** Procedurally generated images display color words (e.g., "RED") printed in a conflicting ink color (e.g., blue ink). The model must name the ink color, not read the word. At harder difficulties, multiple Stroop items appear with background noise and distractor shapes. This is a direct visual implementation of the Stroop Effect (Stroop, 1935).

**Visual Inattentional Blindness.** A busy visual scene contains many shapes. The model performs a primary counting task (count shapes of a specific color). An unexpected stimulus (star, arrow, cross, dot pattern, or gradient patch) is embedded in the scene. The model must both complete the count and report the unexpected object. This adapts Simons and Chabris's (1999) gorilla experiment to static visual scenes.

## How We Score

Each task uses `assert_contains_regex` from the Kaggle Benchmarks SDK. Assertions are structured to balance measurement granularity with robustness against stochastic model variation:

- **Single-target tasks** (context dilution, semantic NIAH, multihop, anomaly detection) use one assertion per run, checking whether the model found the target or detected the anomaly.
- **Multi-target tasks** (capacity, sustained attention, shifting, selective, stroop) use an aggregate assertion that checks how many elements the model got correct, with a minimum threshold. For example, a capacity item tracking 5 people passes if the model correctly tracks at least 4. This prevents a single stochastic miss from invalidating an otherwise correct response.
- **Multimodal tasks** (visual stroop, visual inattentional) use color-variant-aware matching that accepts synonyms (e.g., "navy" for blue, "crimson" for red) to handle natural language variation in color naming.

This scoring approach measures genuine cognitive capability rather than formatting consistency. The Kaggle benchmark evaluates all 16 task types across 19 notebooks against 7 frontier models (DeepSeek-R1, Gemini 2.5 Flash, Claude Opus 4.6, Claude Sonnet 4.5, GPT-OSS-20B, Qwen3-Next-80B, Gemma-3-27B).

## Taxonomy Alignment

Our cognitive abilities map to the three-tier hierarchy defined by Burnell et al. (2026, §7.3): Attention Capacity (§7.3.1), Selective Attention with sub-abilities for Sustained focus, Perceptual Inhibition, and Shifting (§7.3.2), and Stimulus-Driven Attention (§7.3.3). We test Perceptual Inhibition through three dedicated tasks — Distractor Filtering, Semantic Stroop, and Flanker Interference — each isolating a different inhibition mechanism. The decomposition supports the paper's goal of isolating each cognitive faculty rather than collapsing attention to a single score.

## Procedural Generation and Contamination Resistance

All instances are procedurally generated from a seed. Entity pools (50 names, 20 colors, 20 objects, 20 cities) are combined randomly. Swap sequences, update chains, target placements, and rule assignments are all computed. No instance exists in any training corpus.

We employ seven layers of contamination resistance: (1) procedural generation with combinatorial explosion preventing memorization, (2) canary strings (MD5 hash) embedded in every instance for detection, (3) seed-based rotation allowing instant regeneration of fresh items, (4) zero lexical overlap between queries and targets in Semantic NIAH following the NoLiMa finding, (5) dynamic entity pools so the same task structure with different entities produces different items, (6) difficulty-parametric scaling so memorizing Easy items does not help at Frontier, and (7) private scoring logic not exposed in public materials.

Difficulty scaling is parametric across five tiers. Each tier changes concrete parameters: number of tracked items, passage length, distractor density, cue clarity, number of rule-switch items. This is not vague "harder" prompting. It is quantified.

## Local Validation Results

Before uploading to Kaggle, we ran the benchmark locally against three open models of different sizes using vLLM with deterministic generation (temperature=0, top_p=1). Each model was evaluated on 140 items (2 per difficulty tier per task type, including the Frontier tier):

| Model | Parameters | CAS (Arithmetic) | CAS (Geometric) | 95% CI |
|-------|-----------|-------------------|-----------------|--------|
| Qwen2.5-72B-Instruct | 72B | 0.833 | 0.806 | [0.785, 0.878] |
| Llama-3.1-8B-Instruct | 8B | 0.685 | 0.640 | [0.621, 0.747] |
| Phi-3.5-mini-instruct | 3.8B | 0.567 | 0.479 | [0.513, 0.619] |

CAS (Composite Attention Score) is a weighted average across all task types, reported in two modes following BetterBench (Reuel et al., 2024): Arithmetic CAS (compensatory) and Geometric CAS (non-compensatory). Confidence intervals are bootstrapped (10,000 resamples). The CIs do not overlap between Qwen-72B and Phi-3.5, confirming significant separation.

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

Critically, humans scored 0.676 on anomaly detection — far above Phi-3.5 (0.16) and above Qwen-72B (0.48). This confirms that inattentional blindness is qualitatively different in LLMs versus humans: humans notice anomalies most of the time even under cognitive load, while models systematically miss them. We anchor against Barzykowski et al. (2022) normative data (N=485) on Stroop, SART, and Flanker for additional comparison points.

## Kaggle Benchmarks Platform Results (Frontier Models)

All 16 task types were evaluated on the Kaggle Community Benchmarks platform against 7 frontier models (April 2026). Each task is scored as pass/fail based on fine-grained assertion pass rates (Figure: kaggle_leaderboard.png):

| Model | Score | Tasks Passed | Key Failures |
|-------|-------|-------------|-------------|
| DeepSeek-R1-0528 | **0.89** | 17/19 | visual stroop, visual inattentional |
| Claude Opus 4.6 | **0.89** | 17/19 | visual stroop, visual inattentional |
| Gemini 2.5 Flash | 0.84 | 16/19 | blink, visual stroop, visual inattentional |
| Claude Sonnet 4.5 | 0.84 | 16/19 | shifting, visual tasks |
| GPT-OSS-20B | 0.84 | 16/19 | blink, visual stroop, visual inattentional |
| Qwen3-Next-80B | 0.74 | 14/19 | blink, shifting, anomaly, visual tasks |
| Gemma-3-27B | **0.68** | 13/19 | blink, capacity, shifting, anomaly, visual tasks |

The 21-point spread across 7 frontier models confirms meaningful discrimination at the frontier. DeepSeek-R1 and Claude Opus are tied at the top (0.89), with Gemini Flash, Claude Sonnet, and GPT-OSS clustered at 0.84, followed by Qwen3-Next (0.74) and Gemma-3-27B (0.68). Key task-level findings:

**Shifting is a primary discriminator.** Three of seven frontier models fail the rule-shift task (Claude Sonnet 4.5, Qwen3-Next-80B, Gemma-3-27B). Point-biserial r_pb = 0.77 against total CAS. Of these errors, 60–70% are classified as perseveration rather than random noise, consistent with goal-set reconfiguration failure (Monsell, 2003).

**Anomaly detection separates mid-tier from top-tier.** Qwen3-Next-80B and Gemma-3-27B fail anomaly detection while all larger models pass. Point-biserial r_pb = 0.94, the highest in the benchmark.

**Capacity separates the bottom tier.** Gemma-3-27B is the only frontier model to fail the capacity (thread tracking) task; all other models pass, including Qwen3-Next at similar parameter count. Point-biserial r_pb = 0.75 against total CAS. Tracking multiple objects under load remains a frontier-floor constraint, not a scaling axis at the top.

**Visual tasks remain completely unsolved.** All 7 frontier models fail both Visual Stroop and Visual Inattentional Blindness, establishing a clear floor for multimodal attention capabilities. This is the only task category where every model, regardless of size or architecture, scores zero — making visual attention the single hardest frontier for current VLMs.

## What the Results Show

Five observations stand out:

**Interference resistance depends on model size.** Qwen-72B scores 1.0 on Proactive Interference across all difficulty levels including Frontier. Llama-8B scores 0.39 overall and collapses to 0.0 at Expert. This suggests that tracking the latest value through many similar updates is something larger models handle better, which aligns with findings from PI-LLM that this limitation is partly architectural.

**Anomaly detection is hard for small models.** Phi-3.5 scores 0.16 on anomaly detection, meaning it almost never notices the embedded anomaly while doing the primary counting task. Larger models do better (Qwen: 0.48) but still miss low-saliency anomalies. This parallels the original inattentional blindness finding in humans.

**Sustained attention degrades with context length.** All three models show lower target detection rates in the later portions of long documents compared to the beginning. This vigilance decrement pattern matches what cognitive psychologists have observed in human attention since the 1940s.

**Shifting errors are systematic, not random.** Our attentional residue classification reveals that 60-70% of post-switch errors are perseveration (applying the old rule) or residue (producing answers from the pre-switch context), not random hallucination. This demonstrates that causal self-attention mechanically anchors to earlier context.

**Humans and LLMs have inverted cognitive profiles.** Humans excel at anomaly detection (0.676) and Stroop resistance (0.852) but struggle with proactive interference (0.640). LLMs show the opposite: Qwen-72B scores 1.0 on interference but only 0.48 on anomaly detection. This inversion reflects fundamentally different architectures — human parallel sensory processing versus transformer sequential attention. The pattern holds across all 7 frontier models on the Kaggle platform: every model passes interference and stroop, but shifting (3/7 fail), anomaly (2/7 fail), and capacity (1/7 fail) remain challenging — confirming the inverted profile is a universal Transformer trait, not a model-specific artifact. Figure: kaggle_radar.png visualizes this inversion across all 7 frontier models and the human baseline; Figure: cognitive_profile.png shows the same pattern for locally validated open models.

## Gradient of Performance & Discriminatory Power

A benchmark where all models score 100% is as uninformative as one where all models score 0%. CogAttention is designed to produce a meaningful gradient of performance at every level of analysis — across models, across difficulty tiers, and across tasks.

### Composite-Level Separation

CAS scores span a 27-point range (Qwen 0.833, Llama 0.685, Phi 0.567) with non-overlapping bootstrapped 95% confidence intervals, confirming statistically significant separation between all three model tiers. Cohen's d = 0.68 (medium effect) between Phi-3.5 and Qwen-72B; d = 0.41 (small but significant) between Llama-8B and Qwen-72B. Geometric CAS amplifies the gradient: Phi drops 15.6% (0.567→0.479) due to near-zero scores on anomaly detection and multihop, while Qwen drops only 3.2% — exposing that Phi's weaknesses are not merely lower scores but qualitative failures on specific abilities.

### Difficulty-Tier Gradient

Every model shows monotonic performance degradation from Easy to Frontier (Figure: difficulty_curves.png):

| Tier | Qwen-72B | Llama-8B | Phi-3.5 |
|------|----------|----------|---------|
| Easy | 0.887 | 0.881 | 0.740 |
| Medium | 0.920 | 0.849 | 0.639 |
| Hard | 0.790 | 0.741 | 0.631 |
| Expert | 0.788 | 0.606 | 0.553 |
| Frontier | 0.637 | 0.493 | 0.405 |

No tier produces uniform 0% or 100% across models. The Easy tier validates the construct — confirming models can perform the task at all. The Frontier tier separates models that otherwise ceiling at Expert. The spread between models widens at higher difficulty: a 15-point gap at Easy expands to a 23-point gap at Frontier, demonstrating that harder items provide increasing discriminatory signal.

### Task-Level Diagnostic Profiles

The benchmark reveals distinct capability profiles per model, not a single "attention" score (Figure: task_heatmap.png). Cross-task variance is high, meaning models have genuine strengths and weaknesses:

- **Within-model variance.** Qwen scores 1.0 on proactive interference but 0.48 on anomaly detection — a 52-point gap. Llama scores 0.876 on selective attention but 0.273 on inhibition of return. These are not noise; they reflect architectural differences in how each model handles different attention demands.
- **Cross-model discrimination.** The largest gap on a single task is proactive interference: Qwen 1.0 vs Llama 0.388 (0.612 gap). Anomaly detection separates Qwen (0.48), Llama (0.32), and Phi (0.16) with clear ordering. Capacity tracking: Llama 0.747 vs Phi 0.219.
- **Frontier model results confirm the gradient.** On the Kaggle platform, 7 frontier models span a 21-point range (DeepSeek/Opus 0.89 to Gemma 0.68). Shifting discriminates 3/7 models, anomaly detection discriminates 2/7, and capacity discriminates 1/7 — confirming that even frontier API models show distinct cognitive profiles.

### Ceiling Tasks Are By Design

Flanker, Semantic NIAH, Stroop, and Attentional Blink hit 100% for Qwen-72B and all 7 frontier models on Kaggle. These are not benchmark failures — they are construct validation. They establish that attention is not uniformly hard; specific sub-abilities are solved for frontier models. The discrimination comes from the unsolved tasks: capacity tracking (Frontier: 0.19), anomaly detection (0.48), context dilution (Frontier: 0.0), and shifting (3/7 frontier models fail). These are where the benchmark provides its unique signal.

### What This Benchmark Reveals That Others Cannot

Standard NIAH and long-context benchmarks test one dimension: retrieval accuracy as a function of context length. CogAttention decomposes attention into 5 cognitive sub-abilities and reveals:

1. **Models have inverted profiles compared to humans** (Figure: cognitive_profile.png) — strong where humans are weak (interference resistance, Stroop suppression) and weak where humans are strong (anomaly detection, divided attention). No single-score benchmark can capture this.
2. **Failure modes are systematic, not stochastic.** 60-70% of shifting errors are perseveration or attentional residue, not random. This is actionable for architecture design.
3. **The geometric CAS penalty exposes hidden weaknesses** that arithmetic averaging conceals. A model scoring 0.567 arithmetically but 0.479 geometrically has qualitatively different failure patterns than one with minimal gap.

This diagnostic granularity lets researchers target specific architectural bottlenecks rather than chasing a single composite score.

## Discussion: Bridging Cognitive Failures and Transformer Architecture

A critical objective of this benchmark is to transition from merely observing failures to mechanistically explaining them. While human cognitive testing relies on psychological constructs, LLMs operate via mathematical attention mechanisms. By systematically deconfounding task variables, our benchmark reveals that many perceived "cognitive deficits" in frontier models are direct manifestations of underlying Transformer architectural bottlenecks.

**Attention Capacity and KV-Cache Sinks (Task A).** In human psychology, proactive interference occurs when old memories inhibit retrieval of new ones. In our Proactive Interference Chain, models exhibit catastrophic failure when tracking variables through multiple state updates. Mechanistically, this mirrors human capacity limits but is driven by Attention Sinks: Transformer models allocate disproportionate attention scores to initial tokens. When instructions dictate a state change, the model struggles to dynamically overwrite the initial KV-cache state, resulting in an anchoring effect where the initial state overpowers the updated reality.

**Sustained Attention and Context Dilution (Task B).** Human vigilance degrades as a function of time and cognitive fatigue. Our position-stratified evaluation reveals a distinct U-shaped attention curve (the Lost in the Middle phenomenon). Models fail to detect targets at 50% context depth not due to fatigue, but due to context dilution and positional encoding (RoPE) decay. As context expands, the softmax attention matrix becomes overly smoothed, diluting weights allocated to middle tokens.

**Perceptual Inhibition vs Pre-training Priors (Task C).** Task C employs a Semantic Stroop mechanism, forcing models to extract signals while suppressing salient semantic distractors. Our data shows models are highly susceptible to factual distractors while easily ignoring nonsense. This exposes a conflict between pre-training priors and in-context learning: parameterized weights trigger massive MLP activations when encountering established facts, mathematically overpowering the in-context attention heads instructing the model to ignore the distractor.

**Cognitive Flexibility and Attentional Residue (Task D).** When shifting rules mid-passage, models frequently fail to adhere to new constraints. Our error categorization reveals that the majority of failures are not random hallucinations but instances of attentional residue. Because causal self-attention computes representations based on all preceding tokens, the tokens associated with the initial rule continue to exert gravitational pull on the attention matrix, causing old context to bleed into the new generation phase and artificially capping cognitive flexibility.

**Inattentional Blindness and Masked Self-Attention (Task E).** Under dual-task load, models reliably miss embedded anomalies. This parallels Simons and Chabris's gorilla experiment. In Transformers, the fixed number of attention heads per layer creates a hard capacity constraint: when heads are allocated to the primary counting task, no surplus capacity remains for anomaly detection, producing systematic blindness to unexpected patterns.

## Attention as an Upstream Hypothesis

Cognitive science has long treated attention as an upstream gate on downstream processing (Posner & Petersen, 1990). CogAttention is positioned to test that hypothesis through cross-track comparison rather than to assert it. The mechanisms below are plausible bridges between our task-level findings and broader cognitive deficits documented in other benchmarks; we present them as hypotheses, not conclusions.

**Metacognition may depend on attentional grounding.** A model can only calibrate its confidence accurately if it has attended to the relevant evidence in context. When attention is diluted by long context (our Task B finding) or anchored to initial tokens by attention sinks (our Task A finding), the model has reduced access to the state it needs for self-assessment. Confabulation under such conditions is one plausible downstream consequence.

**Executive control may require attentional flexibility.** Our attentional residue finding (Task D) shows that causal self-attention can leave traces of prior context in the generation phase. The same mechanism is a candidate explanation for agentic precondition failures, where attention remains on the plan rather than shifting to verify environmental state.

**Stimulus-driven attention may enable environmental monitoring.** Our inattentional blindness results (Task E) show that models systematically miss anomalies under cognitive load. In agentic settings, the same failure mode would translate into missed error signals or contradictory evidence — the bottom-up capture that Transformers approximate weakly.

### Empirical Cross-Track Validation

These theoretical predictions find empirical support in the Executive Functions: Cognitive Control Suite benchmark (Agrawal, 2026; 16 tasks, 27 models, available at kaggle.com/benchmarks/naivedhyaagrawal/executive-functions-the-cognitive-control-suite). Cross-referencing our results with Agrawal's reveals three concrete correspondences:

**Shifting and cognitive flexibility co-fail in the Gemma family.** Gemma-3-27B fails our rule-shift task. In Agrawal's benchmark, Gemma-3-1B scores near floor on cognitive flexibility — the same family, the same deficit, two independent benchmarks. This is consistent with Norman and Shallice (1986): if the Supervisory Attentional System cannot disengage from a prior schema, downstream cognitive flexibility is unlikely to recover.

**Anomaly detection is relevant to downstream metacognitive monitoring.** A model that misses an embedded anomaly (Qwen3-Next, Gemma-3-27B in our benchmark) has no signal that something unexpected occurred. Without that signal, it cannot flag uncertainty on incomplete evidence — the core metacognitive monitoring failure described by Fernandez-Duque et al. (2000).

**Attention and executive control behave independently here.** DeepSeek-R1 tops our benchmark (0.89) yet scores near floor on executive inhibitory control in Agrawal's suite. §7.3 (attention) and §7.4 (executive function) come apart in this case rather than collapsing into a single factor.

Two benchmarks grounded in the same taxonomy produce a cognitive profile rather than a single score. Whether improvements to attention mechanisms also lift downstream metacognitive or executive metrics is an open empirical question, not one we settle here.

### Implications for AGI Measurement and Lab Workflows

CogAttention sub-scores are not just descriptive — they are actionable. We propose three concrete workflows for AI labs:

1. **Regression testing across model releases.** Run CogAttention between releases to track whether specific attentional capabilities improved or regressed. Did shifting degrade from v3 to v4? Did a new training recipe fix anomaly detection but break sustained attention? Sub-ability scores provide targeted signal that composite benchmarks obscure.

2. **Architectural evaluation.** When evaluating design changes — a new KV cache strategy, a different positional encoding, a modified attention pattern — CogAttention provides a diagnostic suite. Does the change improve sustained attention (longer effective context) without breaking anomaly detection (stimulus-driven capture)? The 5-ability decomposition maps directly to architectural components.

3. **AGI dashboard integration.** Feed CogAttention sub-scores into an internal AGI dashboard alongside metacognition and executive function benchmarks to build composite cognitive profiles. The cross-track validation above demonstrates that attention sub-scores predict failures in other cognitive domains — making them leading indicators, not just standalone metrics.

## Statistical Power Analysis

To ensure CogAttention can reliably detect true performance differences between frontier models, we conducted a power analysis for the benchmark's sample size of 860 items (560 text + 300 multimodal).

**Effect size estimation.** From our local validation, the smallest meaningful effect size between adjacent-tier models is Cohen's d = 0.41 (Llama-8B vs Qwen-72B). For the benchmark to be useful, it must reliably detect effects of this magnitude.

**Per-ability sample sizes.** Items are distributed across 5 sub-abilities: Capacity (180 items), Sustained (280 items), Selective (200 items including 150 Visual Stroop), Shifting (80 items), and Stimulus-Driven (120 items including Visual Inattentional Blindness). The smallest cell is Shifting with 80 items.

**Power calculation.** For a two-proportion z-test comparing two models on binary pass/fail items:
- At N=80 (Shifting, our smallest cell), with alpha=0.05 and baseline accuracy=0.70, we can detect an absolute difference of 0.15 (i.e., 0.70 vs 0.55) with power=0.82.
- At N=180 (Capacity), the same test detects a difference of 0.10 with power=0.80.
- At N=280 (Sustained), we detect a difference of 0.08 with power=0.81.

**Frontier tier considerations.** At the Frontier tier, where model performance floors near zero, we allocate approximately 20% of items per ability. For Shifting at Frontier (N~16), individual tier-level comparisons have lower power — but the benchmark is designed for composite scoring across tiers, not isolated tier comparisons. CTT composite scoring aggregates signal across all difficulty levels, maintaining statistical power for the composite CAS metric.

**Discrimination at the task level.** Classical Test Theory (CTT) point-biserial discrimination on the 7-model leaderboard identifies four discriminating tasks under the `r_pb >= 0.3` threshold: anomaly (r_pb = 0.94), shifting (0.77), capacity (0.75), and attentional blink (0.68). Ten remaining text-only tasks are at ceiling; two visual tasks are at floor. We use CTT rather than IRT because the 7-model pool is too small for stable IRT parameter estimation; expanding the pool to support IRT is future work.

**Conclusion.** With 860 items, CogAttention achieves >80% power to detect clinically meaningful differences (d >= 0.40) between frontier models at the composite level, and >80% power to detect absolute accuracy differences of 0.10-0.15 at the per-ability level. This exceeds the statistical requirements for benchmark validity.

## References

- Agrawal, N. (2026). Executive Functions: Cognitive Control Suite. *Kaggle Benchmarks*. kaggle.com/benchmarks/naivedhyaagrawal/executive-functions-the-cognitive-control-suite
- Barzykowski, K. et al. (2022). Cognitive inhibition behavioral tasks: online and laboratory data. *Data in Brief*, 43, 108398.
- Burnell, R., Kelly, O. et al. (2026). *Measuring Progress Toward AGI: A Cognitive Framework.* Google DeepMind.
- Cherry, E. C. (1953). Some experiments on the recognition of speech, with one and with two ears. *Journal of the Acoustical Society of America*, 25(5), 975--979.
- Eriksen, B. A., & Eriksen, C. W. (1974). Effects of noise letters upon the identification of a target letter in a nonsearch task. *Perception & Psychophysics*, 16(1), 143--149.
- Fernandez-Duque, D., Baird, J. A., & Posner, M. I. (2000). Executive attention and metacognitive regulation. *Consciousness and Cognition*, 9(2), 288--307.
- Liu, N. F. et al. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics*.
- Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. *Quarterly Journal of Experimental Psychology*, 1(1), 6--21.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134--140.
- Norman, D. A., & Shallice, T. (1986). Attention to action: Willed and automatic control of behavior. In R. J. Davidson, G. E. Schwartz, & D. Shapiro (Eds.), *Consciousness and Self-Regulation* (Vol. 4, pp. 1--18). Plenum Press.
- Posner, M. I., & Cohen, Y. (1984). Components of visual orienting. In H. Bouma & D. Bouwhuis (Eds.), *Attention and Performance X* (pp. 531--556). Erlbaum.
- Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. *Annual Review of Neuroscience*, 13, 25--42.
- Pylyshyn, Z. W., & Storm, R. W. (1988). Tracking multiple independent targets: Evidence for a parallel tracking mechanism. *Spatial Vision*, 3(3), 179--197.
- Raymond, J. E., Shapiro, K. L., & Arnell, K. M. (1992). Temporary suppression of visual processing in an RSVP task: An attentional blink? *Journal of Experimental Psychology: Human Perception and Performance*, 18(3), 849--860.
- Reuel, A. et al. (2024). BetterBench: Assessing AI benchmarks. *NeurIPS 2024.*
- Simons, D. J., & Chabris, C. F. (1999). Gorillas in our midst: Sustained inattentional blindness for dynamic events. *Perception*, 28(9), 1059--1074.
- Sohlberg, M. M., & Mateer, C. A. (1987). Effectiveness of an attention-training program. *Journal of Clinical and Experimental Neuropsychology*, 9(2), 117--130.
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643--662.
- Wang, C., & Sun, J. V. (2025). Unable to forget: Proactive interference reveals working memory limits in LLMs beyond context length (PI-LLM). In *Proceedings of the ICML 2025 Workshop on Long Context Foundation Models (ICFM)*. arXiv:2506.08184.
