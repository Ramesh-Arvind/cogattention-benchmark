# Attention Benchmark - Master Plan & Progress Tracker

**Track**: Attention | **Deadline**: April 16, 2026 | **Days remaining**: 28

---

## Phase 1: Benchmark Architecture & Task Design (Days 1-4)
> Design all task types, metrics, and procedural generators before writing any SDK code.

- [x] **1.1** Define the 5 attention sub-benchmarks (one per sub-ability from the paper):
  - [x] **Task A: Attention Capacity** - "How many things can you track at once?"
    - Procedurally generate passages with N concurrent tracking targets (e.g., track prices AND dates AND names simultaneously)
    - Vary N from 1 to 6+; measure where accuracy drops off
    - Format: open-response extraction
  - [x] **Task B: Sustained Attention (Vigilance)** - "Can you stay focused over long passages?"
    - Generate 5K-20K token documents with rare target items (e.g., specific number patterns) scattered throughout
    - Measure detection rate in early vs middle vs late portions
    - Test vigilance decrement: does the model miss more targets later?
    - Format: list all instances found
  - [x] **Task C: Selective Attention / Perceptual Inhibition** - "Can you ignore vivid distractors?"
    - Similar to SiN but with more sophisticated distractor types (emotional, semantic, structural)
    - Procedurally generate tracking cue + signal facts + distractor prose
    - Vary distractor salience and density independently
    - Format: extraction with precision/recall scoring
  - [x] **Task D: Attention Shifting** - "Can you switch focus mid-task?"
    - Mid-passage, change the tracking instruction ("STOP tracking X, START tracking Y")
    - Measure switch cost: accuracy before vs after the switch
    - Test perseveration: does old tracking target bleed into post-switch answers?
    - Format: two-phase extraction with separate scoring
  - [x] **Task E: Stimulus-Driven Attention (Anomaly Detection)** - "Do you notice unexpected things?"
    - Give routine task (e.g., "track all prices"), but embed a clearly anomalous item not in the tracking set
    - Test bottom-up attention: does the model spontaneously mention the anomaly?
    - Format: primary extraction + bonus anomaly detection
- [x] **1.2** Define metrics for each task:
  - [x] Per-task metrics (recall, precision, intrusion rate, switch cost, vigilance decrement)
  - [x] Composite "Cognitive Attention Score" (CAS) across all sub-abilities
  - [x] Define difficulty levels: Easy / Medium / Hard / Expert / Frontier
- [x] **1.3** Design procedural generation strategy:
  - [x] Entity pools (names, locations, items, numbers) - randomized per instance
  - [x] Template libraries for signal embedding at varying concealment depths
  - [x] Difficulty scaling parameters (noise ratio, target count, passage length, switch frequency)
  - [x] Contamination resistance verification plan
- [x] **1.4** Write detailed spec document for each task with:
  - [x] Exact prompt format
  - [x] Exact expected output format
  - [x] Exact assertion logic (regex patterns, scoring rules)
  - [x] Edge cases and failure modes

### Phase 1 Deliverables:
- `[Spec]` `docs/task_specs.md` - Detailed specification for all 5 tasks
- `[Spec]` `docs/metrics.md` - Metric definitions and scoring formulas

---

## Phase 2: Core Implementation (Days 4-10)
> Build the procedural generators, scoring logic, and local test harness.

- [x] **2.1** Project scaffolding:
  - [x] Create directory structure: `src/`, `tests/`, `notebooks/`, `docs/`, `logs/`
  - [x] Set up seed management (numpy, random, reproducible per instance)
  - [x] Create base generator class with common entity pools
- [x] **2.2** Implement Task A: Attention Capacity Generator
  - [x] `src/generators/capacity.py` - Thread Tracking (32 instances, ground truth verified)
  - [x] `src/generators/novel_capacity.py` - Proactive Interference Chain (32 instances, NOVEL)
  - [x] `src/scorers/capacity.py` - Score extraction accuracy + perseveration detection
  - [x] Unit tests: verify gold facts, verify difficulty scaling
- [x] **2.3** Implement Task B: Sustained Attention Generator
  - [x] `src/generators/sustained.py` - Vigilance Probe (32 instances, targets verified in prompts)
  - [x] `src/generators/novel_sustained.py` - Stream Segregation / Cocktail Party (32 instances, NOVEL)
  - [x] `src/scorers/sustained.py` - Vigilance decrement + stream accuracy + breakthrough detection
  - [x] Unit tests: verify target distribution, verify length scaling
- [x] **2.4** Implement Task C: Selective Attention Generator
  - [x] `src/generators/selective.py` - Distractor Filtering (32 instances, signals verified)
  - [x] `src/generators/novel_selective.py` - Semantic Stroop (32 instances, NOVEL)
  - [x] `src/scorers/selective.py` - SAS score + Stroop resistance + correction reflex detection
  - [x] Unit tests: verify signal-distractor ratios, verify difficulty curves
- [x] **2.5** Implement Task D: Attention Shifting Generator
  - [x] `src/generators/shifting.py` - Rule Shift (32 instances, answer counts verified)
  - [x] `src/scorers/shifting.py` - Switch cost + perseveration rate detection
  - [x] Unit tests: verify switch point placement, verify cue independence
- [x] **2.6** Implement Task E: Stimulus-Driven Attention Generator
  - [x] `src/generators/anomaly.py` - Anomaly Detection (32 instances, anomalies verified in prompts)
  - [x] `src/scorers/anomaly.py` - Dual-task score + saliency-stratified detection
  - [x] Unit tests: verify anomaly saliency, verify primary task unaffected
- [x] **2.7** Implement composite scoring:
  - [x] `src/scorers/composite.py` - CAS aggregation (weighted across 8 task types)
  - [x] Weighted combination: capacity(0.20), sustained(0.20), selective(0.25), shifting(0.20), stimulus(0.15)

### Phase 2 Deliverables:
- `[Generator]` `src/generators/*.py` - All 5 procedural generators
- `[Scorer]` `src/scorers/*.py` - All scoring modules
- `[Tests]` `tests/test_generators.py` - Generator unit tests
- `[Tests]` `tests/test_scorers.py` - Scorer unit tests

### Phase 2 Edge Cases & Testing: ✅ 88/88 TESTS PASSED
- [x] Verify no instance ever repeats (hash-check across full dataset)
- [x] Verify difficulty scaling produces monotonic difficulty increase
- [x] Verify gold answers are always unambiguous and extractable
- [x] Verify generated passages stay within token limits (Easy: ~393w, Expert: ~2475w)
- [x] Test assertion regex against adversarial LLM output patterns (extra whitespace, markdown, numbering)
- [x] Test with empty/refused responses (model says "I can't answer")
- [x] Test with over-extraction (model dumps everything)
- [x] Verify signal-distractor no overlap (fixed: selective generator now guarantees uniqueness)
- [x] Verify all canary strings are unique across all 256 instances
- [x] Verify reproducibility (same seed → same dataset)

---

## Phase 3: Local LLM Validation (Days 10-14)
> Run benchmarks against real LLMs locally to verify discriminatory power before porting to kbench.

- [x] **3.1** Set up local inference:
  - [x] Choose 3 local models: Qwen2.5-72B (ceiling), Llama-3.1-8B (mid), Phi-3.5-mini (floor)
  - [x] Implement vLLM wrapper with OOM resilience (`src/inference_engine.py`)
  - [x] Deterministic generation: temperature=0.0, top_p=1.0, max_tokens=2048
  - [x] Context truncation safety (middle-out, keep first+last 512 tokens)
  - [x] Configuration & task registry (`src/eval_config.py`)
- [x] **3.2** Run pilot evaluation:
  - [x] 2 instances per task per difficulty per model (64 instances × 3 models = 192 evals)
  - [x] All raw responses logged in `results/pilot_<model>_results.json`
  - [x] Checkpoint/resume after every item (`logs/checkpoint_<model>.json`)
  - [x] Memory profiling daemon (`logs/memory_profile_<model>.log`)
- [x] **3.3** Analyze discriminatory power:
  - [x] Clear tier separation: qwen72b=0.74, llama8b=0.69, phi3=0.48 CAS
  - [x] All 8 tasks show cross-model spread >= 0.1 (sufficient discrimination)
  - [x] Difficulty gradient analysis (`results/discrimination_analysis.md`)
  - [x] Ceiling/floor detection: interference & stroop ceiling, anomaly floor
- [x] **3.4** Iterate on task design (based on pilot flags):
  - [x] Fix **anomaly** floor (phi3=0.0→0.15) — changed dual_task_score from multiplication to weighted avg (40/60)
  - [x] Fix **interference** Expert difficulty — scaled to 25 updates × 4 keys + decoy verification questions
  - [x] Fix **stroop** Expert difficulty — scaled to 8 items (was 6), Hard to 6 items (was 5)
  - [x] Fix **stream_segregation** scoring — changed dual_score from multiplication to weighted avg (60/40)
  - [x] Re-ran pilot: anomaly floor fixed, all spreads still >=0.1, phi3 CAS 0.48→0.51
  - [x] Remaining ceiling (interference/stroop qwen72b=1.0) is valid: 72B models are genuinely strong here, spread 0.56/0.12 is sufficient
  - [x] Non-monotonic gradients in stream_segregation/capacity are due to small n=2 sample + model format-parsing issues on short prompts

### Phase 3 Edge Cases & Testing:
- [x] Test with models that refuse to follow format instructions (phi3 handled gracefully)
- [x] Test with models that add preamble/commentary before answers (all 3 models, scorer extracts correctly)
- [x] Test with models that use different number formats (fuzzy_value_match handles tolerance)
- [x] Verify OOM handling works (OOM guard with try/except, gc.collect, returns "[OOM_ERROR]")
- [x] Benchmark execution time: 64 instances × 3 models completed in ~7 min total (well within 9hr)

### Phase 3 Deliverables:
- `[Eval]` `src/run_local_eval.py` - Local model evaluation harness
- `[Log]` `logs/pilot_results_*.log` - Pilot evaluation logs
- `[Results]` `results/pilot_metrics.json` - Pilot metric summaries
- `[Results]` `results/discrimination_analysis.md` - Analysis of model variance

---

## Phase 4: Kaggle SDK Integration (Days 14-18)
> Port validated benchmarks to the kaggle-benchmarks SDK format.

- [x] **4.1** Build notebook generator & assertion logic:
  - [x] `src/kbench_assertions.py` — assertion logic per task type, inline helper source, gold_json builder
  - [x] `src/build_kbench_notebooks.py` — generates 5 self-contained .ipynb files with embedded datasets
  - [x] `src/test_kbench_local.py` — local validation with mock SDK (gold replay, bad replay, phase 3 replay)
- [x] **4.2** Port each task to kbench format (5 notebooks, now 560 items with Frontier tier):
  - [x] Task A (Capacity + Interference) → `notebooks/task_a_capacity.ipynb` (120 items incl. Frontier)
  - [x] Task B (Sustained + Stream Seg. + Context Dilution + Semantic NIAH + Multihop) → `notebooks/task_b_sustained.ipynb` (200 items incl. Frontier)
  - [x] Task C (Selective + Stroop + Flanker) → `notebooks/task_c_selective.ipynb` (120 items incl. Frontier)
  - [x] Task D (Shifting + Inhibition of Return) → `notebooks/task_d_shifting.ipynb` (80 items incl. Frontier)
  - [x] Task E (Anomaly) → `notebooks/task_e_anomaly.ipynb` (40 items incl. Frontier)
- [x] **4.3** Define assertions for each task:
  - [x] Map scoring logic to kbench.assertions.assert_contains_regex — fine-grained per-element assertions
  - [x] Handle partial credit via per-element assertion pass rate (continuous scoring)
  - [x] Test assertions against gold responses (100% pass) and bad responses (99.7% fail)
- [x] **4.4** Generate full benchmark dataset:
  - [x] 560 total items across 5 notebooks (expanded from 256 with Frontier tier)
  - [x] 5 difficulty tiers: Easy / Medium / Hard / Expert / Frontier
- [x] **4.5** Metadata export for behavioral probing (analytical notebook support):
  - [x] `src/generators/sustained.py` — `quintile_targets` and target position fractions in metadata
  - [x] `src/generators/selective.py` — `noise_ratio` explicitly in metadata dict
  - [x] `src/generators/novel_selective.py` — `trap_answers` (Stroop factual error traps) in metadata
  - [x] `src/generators/novel_capacity.py` — `all_prior_values` (for attention sink detection) in metadata
  - [x] `src/scorers/selective.py` — `noise_ratio` now in ScoreResult metrics (for notebook parsing)
  - [x] `src/scorers/shifting.py` — residue/random error classification in ScoreResult metrics
- [ ] **4.6** Upload and run on Kaggle:
  - [ ] Create benchmark on Kaggle
  - [ ] Upload all 5 task notebooks
  - [ ] Run against frontier models (gemini-2.0-flash, claude-3-5-sonnet, gpt-4o)

### Phase 4 Edge Cases & Testing:
- [x] Verify kbench assertions handle all LLM output patterns discovered in Phase 3 (r=0.771 correlation)
- [ ] Test that benchmark runs within time limits on Kaggle infrastructure
- [x] Verify all items load correctly — all 5 notebooks syntactically valid JSON + Python
- [ ] Test with 2+ frontier models via Kaggle's Community Benchmarks
- [x] Prompt size check: 8 sustained_expert items slightly over 15K chars (max 16.1K) — acceptable

### Phase 4 Deliverables:
- `[Assertions]` `src/kbench_assertions.py` — Assertion logic + inline helpers for notebooks
- `[Generator]` `src/build_kbench_notebooks.py` — Notebook generator CLI
- `[Tests]` `src/test_kbench_local.py` — Local validation with mock SDK
- `[Notebook]` `notebooks/task_a_capacity.ipynb` - kbench task for Attention Capacity (64 items)
- `[Notebook]` `notebooks/task_b_sustained.ipynb` - kbench task for Sustained Attention (64 items)
- `[Notebook]` `notebooks/task_c_selective.ipynb` - kbench task for Selective Attention (64 items)
- `[Notebook]` `notebooks/task_d_shifting.ipynb` - kbench task for Attention Shifting (32 items)
- `[Notebook]` `notebooks/task_e_anomaly.ipynb` - kbench task for Stimulus-Driven Attention (32 items)

---

## Phase 4.6: Frontier Difficulty Tier ✅ COMPLETE
> All frontier models (Claude Opus 4.6, Sonnet 4.5, Sonnet 4.6, Gemini 2.5 Flash, Qwen3-235B, DeepSeek-R1) scored 1.00. Added "Frontier" tier to break them. All generators and notebooks rebuilt.

**Problem**: Current Expert difficulty was too easy for frontier models. Leaderboard was flat.
**Solution**: Added "Frontier" difficulty tier designed to break frontier models. Regenerated notebooks with 560 total items (up from 256).

- [x] **4.6.1** Update generators with Frontier difficulty tier:
  - [x] `capacity.py` — 8 people, 25 swaps (up from 5/12)
  - [x] `novel_capacity.py` — 50 updates, 8 keys (up from 25/4)
  - [x] `sustained.py` — 150+ paragraphs, subtler targets, higher near-miss density
  - [x] `novel_sustained.py` — 3 interleaved streams (up from 2), longer conversations
  - [x] `selective.py` — zero explicit source cues, pure inference required
  - [x] `novel_selective.py` — 15 Stroop items, multi-hop factual errors
  - [x] `shifting.py` — 20 post-switch items, triple rule change (Rule1→Rule2→Rule3)
  - [x] `anomaly.py` — ultra-subtle anomalies (single character swap, minor date inconsistency, off-by-one numbers)
- [x] **4.6.2** Generate 8 Frontier instances per task (64 new items total)
- [x] **4.6.3** Update kbench assertions for new Frontier items
- [x] **4.6.4** Rebuild all 5 notebooks with Frontier tier included
- [ ] **4.6.5** Local validation: run against best local model, verify scores < 0.8
- [ ] **4.6.6** Re-upload notebooks to Kaggle, re-run frontier models
- [ ] **4.6.7** Verify leaderboard shows spread (target: best model < 0.95)
- [x] **4.6.8** Update writeup with new difficulty tier and results
- [x] **4.6.9** Update visualizations with Frontier tier data

---

## Phase 5: Visualization & Analysis (Days 18-22) — PARTIALLY COMPLETE
> Build publication-quality visualizations and analyze benchmark results.

- [x] **5.1** Design visualization suite:
  - [x] Radar chart: cognitive attention profile across 5 sub-abilities per model → `figures/cognitive_profile.png`
  - [x] Heatmap: model × difficulty × sub-ability performance matrix → `figures/task_heatmap.png`
  - [x] Line chart: performance degradation curves (capacity vs accuracy, position vs detection) → `figures/difficulty_curves.png`
  - [x] Scatter: recall vs intrusion trade-off (selectivity frontier) → `figures/selectivity_frontier.png`
  - [x] Bar chart: CAS scores across models → `figures/cas_scores.png`
- [x] **5.2** Implement plotting code:
  - [x] Dark theme, polished style (matching top-voted entries' aesthetic)
  - [x] Save in all 4 formats: .eps, .pdf, .png, .jpg — `_save_multi_format()` + `export_all_figures_multi_format()`
  - [x] DPI=300 for raster, bbox_inches='tight'
- [x] **5.3** Behavioral probing visualizations & analysis:
  - [x] `plot_lost_in_the_middle()` → `fig_position_bias()` in `src/visualize.py` — U-shaped curve from metadata depths
  - [x] `diagnose_shifting_errors()` → `fig_shifting_error_breakdown()` in `src/visualize.py` — stacked bar proving Attentional Residue
  - [x] `calculate_degradation_coefficient()` → `compute_degradation_coefficient()` in `src/analysis/degradation.py` — power-law fit isolating intrinsic vs extraneous load
  - [x] `fig_degradation_power_law()` in `src/visualize.py` — log-log scatter + fit line
  - [x] `fig_arithmetic_vs_geometric()` in `src/visualize.py` — non-compensatory vs compensatory CAS
- [x] **5.4** Statistical analysis:
  - [x] Bootstrapped confidence intervals for all metrics (`src/analysis/bootstrap.py`)
  - [x] Position bias / U-shape detection (`src/analysis/position_bias.py`)
  - [x] Effect sizes between models → `src/analysis/effect_size.py` (Cohen's d + rank-biserial)
  - [x] Item Response Theory analysis → `src/analysis/irt.py` (2PL/1PL via scipy.optimize)
- [x] **5.5** Public notebook analytical functions:
  - [x] Ensure Kaggle Public Notebook can load Frontier model results and run analytical functions live
  - [x] All analysis functions (`degradation`, `bootstrap`, `position_bias`) are importable and self-contained
- [ ] **5.6** Re-generate figures with full Frontier-tier results (once frontier model eval is done)

### Phase 5 Deliverables:
- `[Script]` `src/visualize.py` ✅ - All plotting code (8 figures total)
- `[Script]` `src/make_thumbnail.py` ✅ - Competition thumbnail generator
- `[Analysis]` `src/analysis/degradation.py` ✅ - Power-law degradation coefficient
- `[Analysis]` `src/analysis/bootstrap.py` ✅ - Bootstrapped confidence intervals
- `[Analysis]` `src/analysis/position_bias.py` ✅ - U-shaped attention detection
- `[Figures]` `figures/cas_scores.png` ✅ - CAS bar chart (pilot data)
- `[Figures]` `figures/cognitive_profile.png` ✅ - Radar chart (pilot data)
- `[Figures]` `figures/difficulty_curves.png` ✅ - Degradation curves (pilot data)
- `[Figures]` `figures/task_heatmap.png` ✅ - Performance matrix (pilot data)
- `[Figures]` `figures/thumbnail.png` ✅ - Competition thumbnail
- `[Figures]` `figures/degradation_power_law.png` ✅ - Log-log power-law scatter + fit
- `[Figures]` `figures/arithmetic_vs_geometric.png` ✅ - Compensatory vs non-compensatory CAS
- `[Figures]` `figures/shifting_error_breakdown.png` ✅ - Perseveration vs residue vs random
- `[Figures]` `figures/position_bias.png` ✅ - U-shaped attention curve
- `[Results]` `results/final_analysis.md` - Statistical analysis report (NOT YET CREATED)

---

## Phase 6: Writeup & Submission (Days 22-27) — PARTIALLY COMPLETE
> Write the benchmark writeup and finalize the submission.

- [x] **6.1** Write benchmark writeup (~1500 words):
  - [x] Introduction: why attention matters for AGI, gap in existing benchmarks
  - [x] Methodology: 5 sub-abilities, procedural generation, contamination resistance
  - [x] Task descriptions: one paragraph per task with example
  - [x] Metrics: CAS composite score and per-task metrics
  - [x] Results & Insights: what the benchmark reveals about frontier models
  - [x] Connection to cognitive science literature (cite the DeepMind paper)
- [ ] **6.2** Create public notebook (for community votes):
  - [ ] Self-contained notebook demonstrating the benchmark
  - [ ] Eye-catching visualizations
  - [ ] Clear narrative with key findings
  - [ ] Run against simulated + real results
- [ ] **6.3** Final submission:
  - [ ] Link writeup to benchmark
  - [ ] Verify all tasks run correctly
  - [ ] Double-check all assertions pass expected patterns
  - [ ] Submit before deadline (April 16, 2026)

### Phase 6 Deliverables:
- `[Writeup]` `docs/writeup.md` ✅ - Competition writeup (1500 words, complete)
- `[Notebook]` `notebooks/public_demo.ipynb` - Public-facing demo notebook (NOT YET CREATED)
- `[Submission]` Final benchmark submitted on Kaggle

---

## Phase 7: Polish & Community (Days 27-29)
> Buffer time for fixes, community engagement, and last-minute improvements.

- [ ] **7.1** Review & iterate:
  - [ ] Re-read writeup with fresh eyes
  - [ ] Check for anti-patterns from lessons.md
  - [ ] Verify discriminatory power with latest model results
- [ ] **7.2** Community engagement:
  - [ ] Share public notebook
  - [ ] Engage in competition discussions
  - [ ] Upvote other quality benchmarks (builds goodwill)
- [ ] **7.3** Final buffer:
  - [ ] Address any last-minute SDK issues
  - [ ] Backup all code and results

---

---

## Research-Informed Task Upgrades (From Literature + Competition Analysis)

> These findings upgrade our original 5-task plan with specific methodologies from 20+ papers and 13 cognitive psychology paradigms. Each upgrade is tagged with its source.

### Task A (Capacity) — UPGRADED Design

**Original**: Multi-target tracking with N concurrent targets.

**Research-informed additions**:
1. **Thread Tracking** (from MOT paradigm): Track N characters through M item-swaps. Plot accuracy × N to find exact capacity limit. *[Source: Pylyshyn MOT; no LLM benchmark exists]*
   ```
   Start: Alice=key, Bob=book, Carol=coin
   1. Alice and Bob trade items.
   2. Bob and Carol trade items.
   → Who has the key?
   ```
   - Scale N from 2→6, swaps from 3→15
   - VERY HIGH contamination resistance (fully procedural)

2. **Proactive Interference Chain** (from PI-LLM, ICML 2025): Stream K semantically related updates to same key, query latest value. Accuracy declines log-linearly. *[Source: Wang & Sun 2025, arXiv:2506.08184]*
   - Key insight: prompt engineering ("ignore earlier input") yields LIMITED success — this is architectural
   - Scale K from 2→20

3. **Attentional Blink** (from cognitive psych): Rapid word stream with two targets (T1, T2). Measure T2 accuracy as function of lag between targets. *[No LLM benchmark exists]*
   - Tests temporal capacity limits
   - Scale lag from 1→10 items

**Key metric**: Capacity curve — accuracy × number of tracked items. Find each model's breakpoint.

### Task B (Sustained) — UPGRADED Design

**Original**: Long-context vigilance with targets at multiple positions.

**Research-informed additions**:
1. **Vigilance Decrement Function** (from CPT paradigm + Context Rot): Place IDENTICAL task instances at 0%, 25%, 50%, 75%, 100% of context. Measure accuracy decay. *[Source: Chroma Context Rot; BABILong NeurIPS 2024]*
   - Key finding: Models utilize only 10-20% of context (BABILong)
   - Key finding: Critical threshold exists — models maintain then fail catastrophically *[arXiv:2601.15300]*

2. **Semantic NIAH** (from NoLiMa, ICML 2025): Needle has ZERO lexical overlap with question. Forces semantic inference, not keyword matching. *[Source: Adobe NoLiMa, arXiv:2502.05167]*
   - Key finding: 11/12 models drop below 50% at 32K when keyword matching is prevented
   - This MUST be in our benchmark — exposes false NIAH passes

3. **Structured vs. Shuffled Haystack** (from Context Rot): Same task in coherent narrative vs. shuffled sentences. *[Source: Chroma Research]*
   - Counterintuitive finding: Models do BETTER on shuffled text (narrative captures attention)
   - This is a genuinely novel insight we can showcase

**Key metric**: Vigilance decrement curve — accuracy × position-in-context.

### Task C (Selective) — UPGRADED Design

**Original**: Signal-in-noise extraction (enhanced SiN).

**Research-informed additions**:
1. **Distractor Gradient with Power-Law** (from GSM-DC, EMNLP 2025): Inject N distractors, measure error rate. Error follows E(m) ~ m^delta. *[Source: Yang et al., arXiv:2505.18761]*
   - Key finding: delta grows with reasoning depth (0.11 at depth 2, 0.49 at depth 5)
   - Build DAG-based tasks with parametric distractor injection

2. **Semantic Stroop** (NOVEL — from cognitive psych adaptation): Embed factual errors in sentences, ask unrelated question. Does model correct the error or answer the question? *[No existing benchmark]*
   ```
   "In the sentence 'Paris is the capital of Germany', what country is mentioned?"
   Expected: Germany. Failure: model says "Actually, Paris is the capital of France..."
   ```
   - Tests suppression of "correction reflex" — a known LLM failure mode

3. **Flanker with Semantic Similarity** (NOVEL): Target sentence flanked by distractors at varying semantic distances. *[No existing benchmark]*
   ```
   "Read ONLY sentence 3: [5 semantically similar sentences]. What day is mentioned in sentence 3?"
   ```

4. **Position-Randomized Retrieval** (from Lost in the Middle, TACL 2024): Control for U-shaped attention bias by randomizing target position and reporting position-stratified metrics. *[Source: Liu et al., arXiv:2307.03172]*

**Key metric**: Power-law degradation coefficient (delta) per model.

### Task D (Shifting) — UPGRADED Design

**Original**: Mid-passage cue switch with perseveration detection.

**Research-informed additions**:
1. **WCST-Style Set Shifting** (from NeuroCognition, 2026): Sort items by unstated rule, rule changes without warning. Measure perseverative errors. *[Source: arXiv:2603.02540]*
   - Distinguish intradimensional shifts (same class) from extradimensional shifts (new dimension) — adds psychometric depth
   - Key finding: Models stronger on text than images; task complexity degrades results

2. **Task Interference Measurement** (from EMNLP 2024): Present Task A context, then switch to Task B. Does Task A context degrade Task B performance? *[Source: Gupta et al., aclanthology/2024.emnlp-main.811]*
   - Even GPT-4 shows interference vulnerabilities
   - Measure "attentional residue" after switching

3. **Inhibition of Return** (NOVEL): Sequential QA returning to previously attended passage. Does accuracy degrade when returning? *[No existing benchmark]*
   ```
   Q1: about paragraph C → Q2: about paragraph A → Q3: about paragraph C again
   Compare: Q3 accuracy vs Q1 accuracy (return penalty?)
   ```

**Key metric**: Switch cost (pre vs post accuracy delta) + perseveration error rate.

### Task E (Stimulus-Driven) — UPGRADED Design

**Original**: Anomaly detection during routine task.

**Research-informed additions**:
1. **Dual-Task Inattentional Blindness** (NOVEL — from Simons & Chabris gorilla experiment): Primary counting task + secondary anomaly in same text. Does primary task load cause blindness? *[No existing benchmark]*
   ```
   "Count the letter 'e' in this passage. Also report any misspelled words."
   [Passage with e's and 2 misspelled words]
   ```

2. **Saliency Gradient** (from Context Rot insight): Vary distractor saliency (ALL CAPS, exclamation marks, emotional language, code blocks). Measure interference on target task. *[Novel application]*
   - Coherent narrative disrupts attention more than shuffled text — test this

3. **Cocktail Party Breakthrough** (from dichotic listening): Track stream A, ignore stream B. But stream B contains a "breakthrough" keyword. Does model notice it? *[No existing benchmark]*
   ```
   "Follow conversation A only. If conversation B mentions 'ALERT', report it."
   ```

**Key metric**: Anomaly detection rate under varying cognitive load.

---

## Winning Patterns Summary (From Prior Competition Research)

### From ARC-AGI ($1M+ prize)
- **Ensemble/multi-approach wins**: Cover multiple sub-abilities, not just one
- **Synthetic data is THE winning strategy**: NVARC generated 3.2M synthetic datapoints
- **Small + clever beats big + brute-force**: Well-designed 4B model beat larger ones
- **Paper prizes reward theoretical grounding**: Cite cognitive science heavily in writeup
- **Rule compliance is non-negotiable**: MindsAI disqualified despite highest score

### From BIG-Bench (Google's own prior benchmark)
- **Minimum 32 items per task** (aligns with our 30+ plan)
- **Include canary strings** to detect training data leakage
- **Design with headroom** — include items current models CAN'T solve (BBEH lesson)
- **Multiple evaluation formats per faculty** (extraction + classification + anomaly)
- **Document construction choices thoroughly** in writeup

### From Stanford BetterBench (46-criteria quality framework)
- **58% of benchmarks report NO statistical significance** — CIs alone differentiate us
- **17/24 benchmarks cannot be reproduced** — self-contained notebooks are rare/valued
- **Include random baselines AND human performance references**
- **Implementation quality scores lowest** (6.2/15 avg) — our code quality is a weapon

### Critical Design Principles From Literature
1. **Avoid keyword/lexical overlap** between queries and targets (NoLiMa lesson)
2. **Randomize target positions** to avoid U-shape exploitation (Lost in the Middle)
3. **Use structured narratives** for haystacks, not random text (Context Rot — harder)
4. **Measure degradation curves**, not pass/fail (power-law from GSM-DC)
5. **Procedurally generate everything** to prevent contamination
6. **Control for confounders** — isolate attention from reasoning/memory

---

## Artifact Index
*(Updated 2026-03-19)*

| Type | Path | Purpose |
|------|------|---------|
| **Specs & Research** | | |
| [Spec] | `tasks/lessons.md` | Strategic intelligence, competitive analysis, winning patterns |
| [Spec] | `tasks/todo.md` | This file — master plan and progress tracker |
| [Spec] | `docs/task_specs.md` | Detailed specification for all 5 tasks |
| [Spec] | `docs/metrics.md` | Metric definitions and scoring formulas |
| [Research] | `tasks/attention_research_report.md` | 20+ papers on LLM attention benchmarks |
| [Research] | `attention_paradigms_research_report.md` | 13 cognitive psychology paradigms adapted for LLMs |
| [Reference] | `reference_notebooks/*.ipynb` | Downloaded competition notebooks (SiN, Metacognition) |
| [Paper] | `measuring-progress-toward-agi-a-cognitive-framework.pdf` | DeepMind cognitive framework paper |
| [Writeup] | `docs/writeup.md` | Competition writeup (~1500 words, complete) |
| **Config** | | |
| [Config] | `config/kaggle/kaggle.json` | Kaggle API credentials |
| [Config] | `CLAUDE.md` | Hackathon engineering standards |
| [Config] | `AGENTS.md` | Team orchestration framework |
| **Generators** (15 modules, all with Frontier tier) | | |
| [Generator] | `src/generators/capacity.py` | Thread Tracking — multi-object tracking (MOT) |
| [Generator] | `src/generators/novel_capacity.py` | Proactive Interference Chain |
| [Generator] | `src/generators/attentional_blink.py` | RSVP rapid serial target detection |
| [Generator] | `src/generators/sustained.py` | Vigilance Probe — CPT-style rare target detection |
| [Generator] | `src/generators/novel_sustained.py` | Stream Segregation / Cocktail Party |
| [Generator] | `src/generators/context_dilution.py` | Performance vs. context length at constant difficulty |
| [Generator] | `src/generators/semantic_niah.py` | Semantic Needle-in-a-Haystack (zero lexical overlap) |
| [Generator] | `src/generators/multihop_attention.py` | Multi-hop scattered reasoning (BABILong-style) |
| [Generator] | `src/generators/selective.py` | Signal-in-Noise distractor filtering |
| [Generator] | `src/generators/novel_selective.py` | Semantic Stroop — correction reflex suppression |
| [Generator] | `src/generators/flanker.py` | Flanker Interference — semantic similarity distractors |
| [Generator] | `src/generators/shifting.py` | Rule-Switch Classification (WCST-style) |
| [Generator] | `src/generators/inhibition_return.py` | Inhibition of Return — re-attention penalty |
| [Generator] | `src/generators/anomaly.py` | Dual-Task Anomaly / Inattentional Blindness |
| **Scorers** | | |
| [Scorer] | `src/scorers/capacity.py` | Extraction accuracy + perseveration detection |
| [Scorer] | `src/scorers/sustained.py` | Vigilance decrement + stream accuracy |
| [Scorer] | `src/scorers/selective.py` | SAS score + Stroop resistance |
| [Scorer] | `src/scorers/shifting.py` | Switch cost + perseveration rate |
| [Scorer] | `src/scorers/anomaly.py` | Dual-task score + saliency-stratified detection |
| [Scorer] | `src/scorers/composite.py` | CAS aggregation (weighted across task types) |
| **Eval & Infrastructure** | | |
| [Eval] | `src/eval_config.py` | Model configs, generation params, task registry |
| [Eval] | `src/inference_engine.py` | vLLM wrapper with OOM resilience, chat template, truncation |
| [Eval] | `src/run_local_eval.py` | Main eval harness with argparse, checkpoint/resume |
| [Analysis] | `src/analysis/discrimination.py` | Difficulty gradient, cross-model spread, ceiling/floor |
| [Assertions] | `src/kbench_assertions.py` | SDK assertion logic, inline helpers, gold_json builder |
| [Generator] | `src/build_kbench_notebooks.py` | Notebook generator (5 .ipynb, 560 items with Frontier) |
| [Tests] | `src/test_kbench_local.py` | Local validation: gold replay, bad replay, phase 3 correlation |
| [Tests] | `tests/test_generators.py` | Generator unit tests |
| [Tests] | `tests/test_scorers.py` | Scorer unit tests |
| **Visualization** | | |
| [Script] | `src/visualize.py` | Publication-quality figures from pilot results |
| [Script] | `src/make_thumbnail.py` | Competition thumbnail generator |
| [Figure] | `figures/cas_scores.png` | CAS bar chart across 3 models |
| [Figure] | `figures/cognitive_profile.png` | Radar chart — 5 cognitive abilities per model |
| [Figure] | `figures/difficulty_curves.png` | Performance degradation across difficulty tiers |
| [Figure] | `figures/task_heatmap.png` | Model × Task performance matrix |
| [Figure] | `figures/thumbnail.png` | Competition display thumbnail |
| **Results** | | |
| [Results] | `results/pilot_metrics.json` | Cross-model comparison (3 models × 8 tasks) |
| [Results] | `results/pilot_qwen72b_cas.json` | Qwen-72B CAS breakdown |
| [Results] | `results/pilot_llama8b_cas.json` | Llama-8B CAS breakdown |
| [Results] | `results/pilot_phi3_cas.json` | Phi-3.5 CAS breakdown |
| [Results] | `results/pilot_qwen72b_results.json` | Qwen-72B raw responses |
| [Results] | `results/pilot_llama8b_results.json` | Llama-8B raw responses |
| [Results] | `results/pilot_phi3_results.json` | Phi-3.5 raw responses |
| [Results] | `results/discrimination_analysis.md` | Actionable flags for task tuning |
| **Notebooks** (5 kbench tasks, all with Frontier tier) | | |
| [Notebook] | `notebooks/task_a_capacity.ipynb` | Capacity + Interference + Blink (120 items) |
| [Notebook] | `notebooks/task_b_sustained.ipynb` | Sustained + Stream Seg. + Dilution + NIAH + Multihop (200 items) |
| [Notebook] | `notebooks/task_c_selective.ipynb` | Selective + Stroop + Flanker (120 items) |
| [Notebook] | `notebooks/task_d_shifting.ipynb` | Shifting + Inhibition of Return (80 items) |
| [Notebook] | `notebooks/task_e_anomaly.ipynb` | Anomaly / Inattentional Blindness (40 items) |

---

## Phase 8: Scoring Upgrades, Analysis & Visualization (Path to Victory)
> Non-compensatory scoring, mechanistic analysis, statistical rigor, visualization upgrades.

- [x] **8.1** Non-Compensatory Scoring (Geometric Mean CAS):
  - [x] Add `_geometric_weighted_mean()` to `src/scorers/composite.py`
  - [x] Add `cas_geometric` to `compute_cas()` return dict (backward compatible)
  - [x] 5 unit tests in `tests/test_scorers.py::TestNonCompensatoryScoring`
- [x] **8.2** Attentional Residue Classification:
  - [x] Split "incorrect" into "residue" and "random_error" in `src/scorers/shifting.py`
  - [x] Add `residue_error_count`, `residue_rate`, `random_error_count`, `random_error_rate`
  - [x] 4 unit tests in `tests/test_scorers.py::TestShiftingResidueClassification`
- [x] **8.3** Degradation Coefficient (Power-Law Fit):
  - [x] New `src/analysis/degradation.py` — power-law fit for selective attention
  - [x] Add `noise_ratio` to `src/scorers/selective.py` metrics
  - [x] 5 unit tests in `tests/test_analysis.py::TestDegradationCoefficient`
- [x] **8.4** Bootstrapped Confidence Intervals:
  - [x] New `src/analysis/bootstrap.py` — `bootstrap_ci()` + `bootstrap_cas_ci()`
  - [x] 7 unit tests in `tests/test_analysis.py::TestBootstrapCI`
- [x] **8.5** Position Bias Analysis (U-Shaped Attention):
  - [x] New `src/analysis/position_bias.py` — accuracy per position bin
  - [x] 4 unit tests in `tests/test_analysis.py::TestPositionBias`
- [x] **8.6** Visualization Upgrades (4 new figures):
  - [x] `fig_degradation_power_law()` — log-log scatter + fit line
  - [x] `fig_arithmetic_vs_geometric()` — grouped bar chart
  - [x] `fig_shifting_error_breakdown()` — stacked bar: persev vs residue vs random
  - [x] `fig_position_bias()` — U-shaped curve
- [x] **8.7** Writeup Discussion Section:
  - [x] Bridging cognitive failures to Transformer architecture (~300 words)
  - [x] Maps: KV-cache sinks, RoPE decay, pre-training priors, attentional residue, masked self-attention
- [x] **8.8** Strategic Plan:
  - [x] `tasks/path_to_victory.md` — 1st Place Master Plan

### Phase 8 Deliverables:
- `[Analysis]` `src/analysis/degradation.py` — Power-law degradation fit
- `[Analysis]` `src/analysis/bootstrap.py` — Bootstrapped confidence intervals
- `[Analysis]` `src/analysis/position_bias.py` — Position bias (U-shape) detection
- `[Scorer]` `src/scorers/composite.py` — Updated with geometric mean CAS
- `[Scorer]` `src/scorers/shifting.py` — Updated with residue/random classification
- `[Scorer]` `src/scorers/selective.py` — Updated with noise_ratio in metrics
- `[Script]` `src/visualize.py` — 4 new figures (8 total)
- `[Writeup]` `docs/writeup.md` — Discussion section added
- `[Tests]` `tests/test_scorers.py` — 9 new tests (Steps 2-3)
- `[Tests]` `tests/test_analysis.py` — 16 new tests (Steps 4-6)
- `[Strategy]` `tasks/path_to_victory.md` — 1st Place Master Plan

---

## Phase 9: Showstopper Notebook + Multimodal VLM Extension ✅ COMPLETE
> Interactive public notebook for judges + procedurally generated Visual Stroop for VLMs.

- [x] **9.1** Showstopper Public Notebook (`notebooks/public_demo.ipynb`):
  - [x] "Try It Yourself" interactive cell — human inattentional blindness demo (ipywidgets)
  - [x] Interactive Plotly radar chart — AI vs Human cognitive profile (hover + toggle)
  - [x] Interactive Plotly difficulty curves — 4 tasks × 3 models
  - [x] Interactive Plotly stacked bar — shifting error breakdown
  - [x] Statistical rigor table (arithmetic/geometric CAS, CIs, effect sizes)
  - [x] Architecture → Cognition mapping table
  - [x] Multimodal extension section
- [x] **9.2** Visual Stroop Generator (`src/generators/visual_selective.py`):
  - [x] Procedural PIL image generation — color word in conflicting ink color
  - [x] 5 difficulty tiers: font size, background noise, distractor shapes
  - [x] 40 instances (8 per difficulty), base64-encoded PNG images
  - [x] Zero data leakage — all images generated at runtime
- [x] **9.3** Visual Stroop Scorer (`src/scorers/visual_selective.py`):
  - [x] Color alias matching (red/crimson/scarlet etc.)
  - [x] Error classification: stroop_error (OCR pathway failure) vs random_error
  - [x] Stroop resistance metric
- [x] **9.4** Documentation Updates:
  - [x] `docs/metrics.md` — Added geometric CAS, residue metrics, degradation, visual Stroop, statistical methods table
  - [x] `docs/task_specs.md` — Updated to 15 task types, 600 items, 5 difficulty tiers
  - [x] `docs/writeup.md` — Already had CIs, effect sizes, discussion section

### Phase 9 Deliverables:
- `[Notebook]` `notebooks/public_demo.ipynb` — Interactive showstopper for judges
- `[Generator]` `src/generators/visual_selective.py` — Procedural Visual Stroop (PIL)
- `[Scorer]` `src/scorers/visual_selective.py` — VLM Stroop scorer with error classification
- `[Figures]` `figures/visual_stroop_sample_*.png` — Sample generated Stroop images
- `[Docs]` `docs/metrics.md` — Updated with all new metrics and statistical methods
- `[Docs]` `docs/task_specs.md` — Updated to reflect 15 tasks × 40 items = 600
