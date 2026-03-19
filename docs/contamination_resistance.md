# Contamination Resistance Strategy

## Why This Matters

The DeepMind companion paper warns: "many of the high-quality benchmarks that do exist are fully public, so they are susceptible to data contamination and may not provide generalizable signal." Our benchmark is designed from the ground up to be contamination-proof.

## Seven Layers of Contamination Resistance

### 1. Procedural Generation (Primary Defense)

Every benchmark item is generated algorithmically from a seed. No item exists in any training corpus because no item existed before generation time.

- **Thread Tracking**: Random names from a pool of 50, random items from a pool of 20, random swap sequences — `O(50^N × 20^N × N!)` possible instances per difficulty level
- **Proactive Interference**: Random key names, random value domains (cities, names, numbers), random update sequences — combinatorial explosion prevents memorization
- **Sustained Attention**: Random category assignment, random target/near-miss selection, random paragraph ordering, random filler text
- **All 15 generators** follow this pattern: entity pools × configuration parameters × random orderings

### 2. Canary Strings (Detection)

Every instance includes a unique MD5 canary hash:
```
<!-- CogAttention-v1-a3f2b1c9d4e7 -->
```

If this string appears in any model's training data, we can detect contamination. The canary is embedded in the prompt but does not affect the task.

### 3. Seed-Based Reproducibility with Rotation

- All generation uses `random.Random(seed + instance_idx * prime1 + difficulty_idx * prime2)`
- The seed can be rotated to produce entirely new item sets while maintaining the same difficulty structure
- A single parameter change regenerates 600 fresh items

### 4. No Lexical Overlap Between Query and Target

Following the NoLiMa finding (ICML 2025) that keyword matching inflates NIAH scores, our Semantic NIAH task has **zero lexical overlap** between the question and the needle. The model must use semantic inference, not string matching.

### 5. Dynamic Entity Pools

Entity names, values, and categories are drawn from pools at generation time. The same task structure with different entities produces a different item that tests the same cognitive ability. Memorizing "Alice holds the key after 3 swaps" doesn't help when the next instance uses "Viktor holds a compass after 7 swaps."

### 6. Difficulty-Parametric Scaling

Difficulty is controlled by continuous parameters (number of tracked objects, context length, distractor density, cue explicitness), not by changing the task content. This means even if a model has seen Easy items, it cannot generalize to Expert/Frontier items — the cognitive load is qualitatively different.

### 7. Private Scoring Logic

The scoring logic (assertion patterns, regex matchers, error classification rules) is not embedded in the public demo notebook. Only the task notebooks submitted to Kaggle contain the assertion code, and these run server-side.

## Verification Protocol

1. **Canary check**: After evaluation, search model outputs for canary substring matches
2. **Cross-seed validation**: Generate a second item set with seed=9999, evaluate same models — scores should be statistically equivalent (within bootstrap CI) if no contamination
3. **Novel item injection**: Insert 10% items with deliberately unusual entity combinations not found in any training corpus
4. **Positional control**: Randomize target positions across items to prevent position-based shortcuts

## What We Do NOT Do (and Why)

- We do NOT use items from existing cognitive psychology test batteries (WAIS, CPT-II, WCST) — these appear in textbooks and training data
- We do NOT use fixed prompt templates that could appear in benchmark collections
- We do NOT rely on knowledge retrieval — all answers are derivable from the prompt alone
