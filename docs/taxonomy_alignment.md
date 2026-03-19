# Taxonomy Alignment: CogAttention → DeepMind Cognitive Framework

## DeepMind's Attention Hierarchy (Paper §7.3)

The companion paper defines Attention as "the ability to focus cognitive resources on specific aspects of perceptual stimuli, thoughts, or task demands" with this exact hierarchy:

```
Attention
├── Attention Capacity
│   └── Amount of information a system can focus on simultaneously
│       (potentially different across modalities)
│
├── Selective Attention / Attentional Control (top-down, goal-driven)
│   ├── Sustained Attention
│   │   └── Maintaining focus on goal-relevant information over time
│   ├── Perceptual Inhibition
│   │   └── Ignoring distracting or goal-irrelevant perceptual information
│   └── Attention Shifting
│       └── Actively shifting attention from one location/information to another
│
└── Stimulus-Driven Attention (bottom-up)
    └── Attention directed toward new stimuli or environmental changes
```

## Our Mapping

| DeepMind Sub-Ability | Our Task Types | Paradigm Source |
|---------------------|---------------|----------------|
| **Attention Capacity** | Thread Tracking, Proactive Interference, Attentional Blink | MOT (Pylyshyn, 1988), PI-LLM (Wang & Sun, 2025) |
| **Sustained Attention** | Vigilance Probe, Stream Segregation, Context Dilution, Semantic NIAH, Multi-hop Attention | CPT (Mackworth, 1948), Dichotic Listening (Cherry, 1953) |
| **Perceptual Inhibition** | Distractor Filtering, Semantic Stroop, Flanker Interference | SiN, Stroop (1935), Eriksen Flanker |
| **Attention Shifting** | Rule Shift (WCST-style), Inhibition of Return | WCST (Monsell, 2003), IOR |
| **Stimulus-Driven Attention** | Anomaly Detection (Inattentional Blindness) | Simons & Chabris (1999) |

### Key Alignment Decisions

1. **Perceptual Inhibition is explicitly separated from Selective Attention.** The paper distinguishes these as sibling sub-abilities under Attentional Control. We test Perceptual Inhibition through three dedicated tasks:
   - **Distractor Filtering** — ignore noise values mixed with signal values
   - **Semantic Stroop** — suppress correction reflex when encountering factual errors
   - **Flanker Interference** — identify target among semantically similar distractors

2. **Attention Capacity includes temporal capacity.** The Attentional Blink task tests capacity limits in the temporal domain (rapid sequential presentation), complementing the spatial capacity tested by Thread Tracking.

3. **Sustained Attention includes context-length effects.** Context Dilution and Semantic NIAH test whether attention is maintained over increasing distances, which is the LLM-native manifestation of vigilance decrement.

4. **Stimulus-Driven Attention uses dual-task paradigm.** The paper's "bottom-up" attention is tested through inattentional blindness under cognitive load — the model must notice anomalies while performing a primary task.

## Format Variety (Paper Criterion 5)

The paper requires "multiple tasks with a variety of structures and formats":

| Format | Tasks Using It |
|--------|---------------|
| **Open extraction** | Thread Tracking, Vigilance Probe, Distractor Filtering |
| **Numbered classification** | Rule Shift, Stroop, Flanker |
| **Dual-response** | Anomaly Detection (count + detect), Stream Segregation (answer + breakthrough) |
| **List enumeration** | Sustained Attention (list all targets) |
| **Key-value extraction** | Proactive Interference (report final values) |
| **Binary + explanation** | Anomaly Detection (yes/no + describe anomaly) |
| **Multi-phase** | Rule Shift (pre-switch + post-switch), Inhibition of Return (initial + shift + return) |
| **Visual (multimodal)** | Visual Stroop (image input, color naming) |

## Difficulty Gradient (Paper Criterion 4)

The paper requires tasks "easy for humans and hard for AI" alongside tasks "that test the limits of human capabilities":

| Tier | Human Expected | AI Observed | Design |
|------|---------------|-------------|--------|
| Easy | ~95-100% | 85-100% | Minimal cognitive load, explicit cues |
| Medium | ~85-95% | 60-90% | Moderate load, standard cues |
| Hard | ~70-85% | 40-75% | High load, subtle cues |
| Expert | ~50-70% | 20-60% | Near human capacity limits |
| Frontier | ~30-50% | 0-40% | Designed to break frontier AI and challenge expert humans |

The Easy tier deliberately includes items where human accuracy approaches 100% — testing whether models fail on things humans find trivial (per Chollet et al., 2025). The Frontier tier pushes beyond typical human performance, testing the absolute limits of attention.

## Three Sources of Uncertainty (Paper §9)

The paper identifies three sources of uncertainty we address:

1. **Task quality** — IRT item fit statistics verify items are well-calibrated (discrimination > 0.5, no DIF)
2. **Construct validity** — Each task isolates a specific attention sub-ability; attentional residue classification demonstrates we measure attention shifting, not general reasoning
3. **Stochasticity** — Bootstrapped CIs (10,000 resamples) quantify sampling uncertainty; deterministic generation (temperature=0) eliminates LLM response variability
