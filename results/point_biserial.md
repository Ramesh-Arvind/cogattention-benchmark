# Point-Biserial Discrimination (CTT)

## Task-level (7-model leaderboard)

`r_pb` is Pearson correlation between task pass/fail across models and model total CAS. `p` is item difficulty (fraction passing). Classical threshold: `r_pb >= 0.3` → discriminating.

| Task | N | p (difficulty) | r_pb | Discriminating |
|------|---|----------------|------|----------------|
| cogattention_anomaly1 | 7 | 0.714 | 0.88 | Yes |
| cogattention_blink | 7 | 0.143 | 0.561 | Yes |
| cogattention_blinks | 7 | 1.0 | n/a (no variance) | No |
| cogattention_capacity | 7 | 0.857 | 0.73 | Yes |
| cogattention_context_dilution | 7 | 1.0 | n/a (no variance) | No |
| cogattention_flanker | 6 | 1.0 | n/a (no variance) | No |
| cogattention_flankers | 7 | 1.0 | n/a (no variance) | No |
| cogattention_inhibition_return | 7 | 1.0 | n/a (no variance) | No |
| cogattention_inhibition_return1 | 7 | 1.0 | n/a (no variance) | No |
| cogattention_interference | 7 | 1.0 | n/a (no variance) | No |
| cogattention_multihop | 7 | 1.0 | n/a (no variance) | No |
| cogattention_selective | 7 | 1.0 | n/a (no variance) | No |
| cogattention_semantic_niah | 7 | 1.0 | n/a (no variance) | No |
| cogattention_shifting | 7 | 0.571 | 0.863 | Yes |
| cogattention_stream_segregation | 7 | 1.0 | n/a (no variance) | No |
| cogattention_stroop | 7 | 1.0 | n/a (no variance) | No |
| cogattention_sustained | 7 | 1.0 | n/a (no variance) | No |
| cogattention_visual_inattentional | 7 | 0.0 | n/a (no variance) | No |
| cogattention_visual_stroop | 7 | 0.0 | n/a (no variance) | No |

## Item-level (pilot, N=3 models — INTERNAL ONLY, do not cite)

**Do not cite these numbers in the main writeup.** N=3 is too thin for item-level point-biserial to be defensible; a reviewer will (correctly) flag it. This table is kept only to surface items with zero variance for internal task-quality review. Treat as exploratory; expand model pool before external use.

| Task type | Items w/ variance | Mean r_pb | Median r_pb | % discriminating | Mean p |
|-----------|-------------------|-----------|-------------|-----------------|--------|
| anomaly | 7 | 0.25 | 0.016 | 0.429 | 0.381 |
| capacity | 9 | 0.57 | 0.858 | 0.667 | 0.481 |
| dilution | 3 | 0.577 | 0.858 | 0.667 | 0.667 |
| interference | 8 | 0.334 | 0.016 | 0.375 | 0.625 |
| multihop | 7 | 0.863 | 0.858 | 1.0 | 0.572 |
| stroop | 1 | 0.858 | 0.858 | 1.0 | 0.667 |
