# ADAPT architecture

## Core idea

Everything converges on one structured object, the **decision frame**
(`src/adapt/schema.py`): options, criteria (soft preferences with weights),
constraints (hard limits), context facts and known unknowns. Every value carries
its provenance: `stated`, `assumed`, `inferred`, `retrieved` (must cite a source) or `computed`.

Domain generality comes from this shared schema, not from model size. A job offer
and a laptop purchase are both "options scored against criteria under constraints".

```
text / structured input
        │
        ▼
  extraction ──────────────► DecisionFrame ◄──── retrieval (evidence + citations)
  (pretrained baseline;            │
   from-scratch transformer        ▼
   as research track)        decision engine
                         ┌─────────┴──────────┐
                         │ 1. hard-constraint filter (deterministic)
                         │ 2. scoring: weighted baseline → learned preference model
                         │ 3. consequence estimates (Python math / small regressors)
                         │ 4. sensitivity + value-of-information
                         │ 5. policy: recommend | clarify | abstain
                         └─────────┬──────────┘
                                   ▼
                       response (ranking, trade-offs, assumptions,
                       citations, versions) ──► feedback store (reviewed, offline)
```

## Where neural networks are and are not used

| Component | Approach | Why |
|---|---|---|
| Constraint checking, arithmetic, budgets | Plain Python | Must be exact; a network adds error and nothing else |
| Baseline scoring | Weighted sum over normalised criteria | Transparent; every learned model must beat it |
| Preference learning | Small PyTorch pairwise model (Bradley–Terry / RankNet style) | Learns how much users care about each criterion from their choices |
| Consequence prediction | Regression/classification per domain, only where real data exists | Without data, it's deterministic math or "unknown" |
| Text → frame extraction | Pretrained extractor (main path) vs. from-scratch transformer (research) | Hardest part; a tiny model trained from scratch can't be on the critical path |
| Clarify/abstain policy | Rules over sensitivity + missing data first, learned later if it helps | Must be explainable and testable |

## Clarifying questions

ADAPT asks the question whose answer is most likely to **change the ranking**
(value of information), not every question about a missing field. Sensitivity
analysis also produces statements like "B wins unless cost weight exceeds 0.4".

## What "correct" means

Usually no option is objectively best across users. The system is scored on
being **consistent with the user's own stated preferences and limits**, never
violating hard limits, and abstaining or asking when the evidence doesn't
support a recommendation.
