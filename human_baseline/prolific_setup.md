# Prolific Study Setup Guide -- CogAttention Human Baseline

## Study Title

**Human Baseline for CogAttention: A Cognitive Attention Benchmark**

## Study Description (shown to participants on Prolific)

> We are conducting a study to measure human performance on a set of attention
> and information-processing tasks. You will read short passages and answer
> questions that test different aspects of attention: tracking multiple items,
> finding targets in long text, filtering relevant information from distractors,
> adapting to rule changes, and detecting unusual content.
>
> This study is part of a research project that compares human cognitive
> attention abilities with those of large language models. Your responses will
> help establish a human performance baseline.
>
> **Estimated time:** 25--35 minutes
> **Payment:** GBP 6.00 (approximately USD 7.50, equivalent to ~$12--14/hr)

---

## Platform Configuration

| Parameter | Value |
|---|---|
| **Platform** | Prolific (prolific.com) |
| **Survey host** | Qualtrics (linked from Prolific via redirect URL) |
| **Completion code** | Auto-generated; embedded at end of Qualtrics survey |
| **Device** | Desktop only (no mobile -- passages are too long) |

---

## Screening Criteria

| Filter | Setting |
|---|---|
| **Age** | 18--65 |
| **Minimum education** | Upper secondary (high school diploma / A-levels equivalent) |
| **Fluent languages** | English (native or fluent) |
| **Approval rate** | >= 95% on Prolific |
| **Country of residence** | UK, US, Canada, Australia, Ireland, New Zealand |
| **Previous participation** | Exclude anyone who participated in pilot studies for this project |

---

## Payment and Timing

| Item | Value |
|---|---|
| **Base payment** | GBP 6.00 per participant |
| **Estimated completion time** | 25--35 minutes |
| **Effective hourly rate** | ~GBP 10.30--12.00/hr (~USD 12.00--14.40/hr) |
| **Bonus** | None (flat rate) |
| **Prolific fee** | ~33% service charge on top of base payment |
| **Total cost per participant** | ~GBP 8.00 including fees |

Prolific's minimum recommended rate is GBP 6.00/hr. Our rate exceeds this to
ensure ethical compensation and reduce dropout.

---

## Sample Size

| Parameter | Value |
|---|---|
| **Target N** | 50 participants |
| **Design** | Within-subjects: all participants complete the same 30 items |
| **Justification** | 50 participants gives stable per-item accuracy estimates (SE < 0.07 for proportions near 0.5). For 30 items x 50 participants = 1,500 data points total. |
| **Overshoot** | Recruit 55 to account for exclusions (attention checks, incomplete) |

---

## Study Design

### Items

- 30 total items: 6 per cognitive ability cluster
- Clusters: Capacity, Sustained, Selective, Shifting, Anomaly Detection
- Difficulty levels: Easy (2 items), Medium (2 items), Hard (2 items) per cluster
- Expert and Frontier difficulties are excluded (designed for LLMs, not humans)

### Item Order

- Fixed order within each cluster, randomized cluster order across participants
  (Qualtrics block randomization)
- This controls for fatigue effects while keeping within-cluster ordering consistent

### Attention Checks

- 2 embedded attention check items (e.g., "Please type the word 'attention' in
  the box below")
- Participants failing either attention check are excluded from analysis

### Exclusion Criteria

1. Failed 1+ attention check items
2. Completed in < 8 minutes (indicates random responding)
3. Completed in > 60 minutes (indicates significant distraction/interruption)
4. Prolific return/timeout

---

## Qualtrics Survey Structure

1. **Welcome page** -- consent form (see `consent_form.md`)
2. **Demographics** -- age range, education level (for analysis, not screening)
3. **Instructions** -- general overview + practice example (see `instructions.md`)
4. **Block 1-5** -- one per ability cluster, randomized order
   - Each block starts with a brief task-specific instruction
   - 6 items per block
5. **Attention check** -- 1 item after block 2, 1 item after block 4
6. **Debrief** -- optional free-text feedback, completion code display

---

## Data Collection

- Qualtrics exports: CSV with one row per participant, one column per item
- Prolific exports: participant IDs, completion status, time taken
- Merge on Prolific participant ID (embedded as URL parameter in Qualtrics)

---

## Timeline

| Step | Target Date |
|---|---|
| Finalize survey items | Day 0 |
| Qualtrics survey built and tested | Day 1 |
| Pilot run (5 participants) | Day 2 |
| Review pilot data, adjust timing estimate if needed | Day 2 |
| Full launch (50 participants) | Day 3 |
| Data collection closes | Day 4 |
| Score and analyze | Day 5 |

---

## Ethical Considerations

- No deception is involved
- Tasks involve only reading and answering questions about fictional text
- No sensitive personal data collected beyond Prolific demographics
- Data stored pseudonymously (Prolific IDs only, no names)
- See `consent_form.md` for full informed consent document
