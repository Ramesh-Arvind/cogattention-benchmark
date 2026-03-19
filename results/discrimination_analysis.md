# Discrimination Analysis Report

## 1. Difficulty Gradient (per model)

### Model: `llama8b`

| Task | Easy | Medium | Hard | Expert | Monotonic | Gradient |
|------|------|--------|------|--------|-----------|----------|
| anomaly | 1.0 | 0.8 | 0.0 | 0.2 | **NO** | 0.8 |
| capacity | 0.0 | 1.0 | 1.0 | 1.0 | **NO** | -1.0 |
| interference | 1.0 | 0.25 | 0.5 | 0.0 | **NO** | 1.0 |
| selective | 1.0 | 1.0 | 0.5 | 0.9334 | **NO** | 0.0666 |
| shifting | 0.9166 | 0.625 | 0.6 | 0.6 | Yes | 0.3166 |
| stream_segregation | 1.0 | 1.0 | 1.0 | 1.0 | Yes | 0.0 |
| stroop | 1.0 | 0.875 | 0.8334 | 0.8125 | Yes | 0.1875 |
| sustained | 1.0 | 1.0 | 0.75 | 0.4583 | Yes | 0.5417 |

### Model: `phi3`

| Task | Easy | Medium | Hard | Expert | Monotonic | Gradient |
|------|------|--------|------|--------|-----------|----------|
| anomaly | 0.2 | 0.0 | 0.2 | 0.2 | **NO** | 0.0 |
| capacity | 0.0 | 0.1666 | 0.75 | 0.6 | **NO** | -0.6 |
| interference | 1.0 | 0.75 | 0.1666 | 0.5 | **NO** | 0.5 |
| selective | 0.8334 | 0.3958 | 0.3 | 0.6667 | **NO** | 0.1667 |
| shifting | 0.9166 | 0.625 | 0.75 | 0.45 | **NO** | 0.4666 |
| stream_segregation | 1.0 | 0.5 | 0.5 | 0.5 | Yes | 0.5 |
| stroop | 1.0 | 1.0 | 0.6666 | 0.875 | **NO** | 0.125 |
| sustained | 0.9 | 0.6875 | 0.55 | 0.5834 | **NO** | 0.3166 |

### Model: `qwen72b`

| Task | Easy | Medium | Hard | Expert | Monotonic | Gradient |
|------|------|--------|------|--------|-----------|----------|
| anomaly | 0.5 | 0.6 | 0.2 | 0.4 | **NO** | 0.1 |
| capacity | 1.0 | 0.5 | 0.125 | 1.0 | **NO** | 0.0 |
| interference | 1.0 | 1.0 | 1.0 | 1.0 | Yes | 0.0 |
| selective | 0.5 | 1.0 | 0.4 | 0.8334 | **NO** | -0.3334 |
| shifting | 0.9166 | 0.875 | 1.0 | 0.85 | **NO** | 0.0666 |
| stream_segregation | 1.0 | 1.0 | 1.0 | 1.0 | Yes | 0.0 |
| stroop | 1.0 | 1.0 | 1.0 | 1.0 | Yes | 0.0 |
| sustained | 1.0 | 1.0 | 0.7 | 0.9584 | **NO** | 0.0416 |

## 2. Cross-Model Spread

| Task | Spread | Sufficient (>=0.1) | Model Scores |
|------|--------|---------------------|--------------|
| anomaly | 0.35 | Yes | llama8b=0.5, phi3=0.15, qwen72b=0.425 |
| capacity | 0.3708 | Yes | llama8b=0.75, phi3=0.3792, qwen72b=0.6562 |
| interference | 0.5625 | Yes | llama8b=0.4375, phi3=0.6042, qwen72b=1.0 |
| selective | 0.3093 | Yes | llama8b=0.8583, phi3=0.549, qwen72b=0.6833 |
| shifting | 0.225 | Yes | llama8b=0.6854, phi3=0.6854, qwen72b=0.9104 |
| stream_segregation | 0.375 | Yes | llama8b=1.0, phi3=0.625, qwen72b=1.0 |
| stroop | 0.1198 | Yes | llama8b=0.8802, phi3=0.8854, qwen72b=1.0 |
| sustained | 0.2344 | Yes | llama8b=0.8021, phi3=0.6802, qwen72b=0.9146 |

## 3. Ceiling / Floor Detection

| Task | Best Model Mean | Worst Model Mean | Ceiling (>0.95) | Floor (<0.05) |
|------|----------------|-----------------|-----------------|---------------|
| anomaly | 0.5 | 0.15 | No | No |
| capacity | 0.75 | 0.3792 | No | No |
| interference | 1.0 | 0.4375 | **YES** | No |
| selective | 0.8583 | 0.549 | No | No |
| shifting | 0.9104 | 0.6854 | No | No |
| stream_segregation | 1.0 | 0.625 | **YES** | No |
| stroop | 1.0 | 0.8802 | **YES** | No |
| sustained | 0.9146 | 0.6802 | No | No |

## 4. Actionable Flags

- **interference**: Ceiling hit (best=1.0) — consider harder instances.
- **stream_segregation**: Ceiling hit (best=1.0) — consider harder instances.
- **stroop**: Ceiling hit (best=1.0) — consider harder instances.
- **capacity** (llama8b): Non-monotonic difficulty with flat gradient (-1.0).
- **anomaly** (phi3): Non-monotonic difficulty with flat gradient (0.0).
- **capacity** (phi3): Non-monotonic difficulty with flat gradient (-0.6).
- **capacity** (qwen72b): Non-monotonic difficulty with flat gradient (0.0).
- **selective** (qwen72b): Non-monotonic difficulty with flat gradient (-0.3334).
- **sustained** (qwen72b): Non-monotonic difficulty with flat gradient (0.0416).
