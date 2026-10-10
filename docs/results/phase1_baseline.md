# Phase 1 — baseline engine results

Measured 2026-10-10. Engine `baseline-0.1.0`, dataset `synthetic-v1`, seed 0,
600 scenarios × 9 domains = 5,400. Reproduce with `python scripts/evaluate_baseline.py`.

**These are synthetic results.** They test the engine's mechanics against known
ground truth. They are not evidence of real-world decision quality.

## Headline (test splits)

| Split | n | Decision acc. | Top-1 (complete) | NDCG | NDCG random | Missed questions | Extra questions | Violations |
|---|---|---|---|---|---|---|---|---|
| test_seen | 657 | 0.892 | 0.925 | 0.985 | 0.807 | 0.236 | 0.019 | 0 |
| test_unseen_domain | 1,200 | 0.906 | 0.904 | 0.976 | 0.805 | 0.162 | 0.009 | 0 |

The baseline has no learned parameters, so seen vs. unseen domains shouldn't differ
for it; the gap is sampling noise. The split matters for phase 2's learned models.

## By scenario kind (all splits)

| Kind | n | Decision acc. | Notes |
|---|---|---|---|
| complete | 2,431 | 0.894 | Errors come from the hidden nonlinear value shapes the linear baseline can't see |
| infeasible | 647 | 1.000 | Always abstains when hard limits exclude everything |
| missing_weight | 827 | 1.000 | Asks for priorities, or recommends only a dominating option |
| missing_value | 1,078 | 0.789 | **Weakest area.** Misses 39% of questions that mattered |
| no_options | 417 | 1.000 | Always asks for options |

Hard-limit violations: **0 of 5,400**. Latency: ~0.27 ms per decision on CPU.

## Known weaknesses (targets for phase 2)

1. **Missing values (missed 39% of needed questions).** The engine assumes a missing value
   lies within the range of the other options. The ground truth allows anywhere in the
   domain's real range, so the engine is overconfident when the hidden value could be extreme.
2. **Nonlinear preferences (top-1 0.894 on complete scenarios).** Users' true preferences have
   diminishing returns and thresholds; a weighted linear sum misses them. This is what the learned
   preference model should improve on.
3. **Normalisation is relative to the options shown.** Adding or removing an option can reorder
   the others. Known limitation of min-max scoring.

## Metric note

NDCG uses each option's utility *above the worst feasible option* as its gain. With raw utilities,
a random ranking already scored 0.929, which made the metric uninformative. The random-ranking
NDCG is now reported next to every NDCG.
