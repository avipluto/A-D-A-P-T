# Phase 2a — learned preference model vs. baseline

Measured 2026-10-10 on `synthetic-v1` (seed 0, 5,400 scenarios). Engine policy
`baseline-0.1.0` for all rows; only the scorer changes. Reproduce:

```powershell
D:\adapt-env\Scripts\python.exe scripts/train_preference.py --seed 0            # and --no-attr, seeds 0-2
D:\adapt-env\Scripts\python.exe scripts/evaluate.py --model D:\adapt-data\artifacts\models\pref-v1-s0\model.pt
```

**Synthetic results only.** The hidden preference shapes were written by us, so this
shows the model can learn shapes of the kind we planted, not that it matches real people.

## Model

`score = Σ_c w_c · g(n, cv, k, attribute)`, where g is a 2-layer MLP (hidden 32) with a sigmoid output.
1,706 parameters. Bradley–Terry pairwise loss on 12,028 pairs from 2,107 train scenarios.
Attribute names are randomly hidden 30% of the time during training. Early stopping on val pair accuracy.

| | with attribute names | generic (no names) |
|---|---|---|
| Val pair accuracy (seeds 0/1/2) | 0.959 / 0.960 / 0.962 | 0.947 / 0.945 / 0.945 |
| Train time per run (RTX 2050) | 9–10 s | 7–10 s |
| Peak GPU memory | ~34 MiB | ~34 MiB |
| Inference (CPU, incl. policy) | ~0.72 ms/decision | ~0.72 ms/decision |
| Baseline inference | 0.26 ms/decision | |

## Test results (3 seeds each; ranges shown)

| Split | Scorer | Decision acc. | Top-1 (complete) | NDCG | Missed questions | Violations |
|---|---|---|---|---|---|---|
| test_seen | baseline | 0.892 | 0.925 | 0.985 | 0.236 | 0 |
| test_seen | learned + names | 0.900–0.901 | **0.953–0.958** | 0.992 | 0.245–0.250 | 0 |
| test_seen | learned, generic | 0.892–0.895 | 0.939–0.944 | 0.988 | 0.250–0.264 | 0 |
| test_unseen_domain | baseline | 0.906 | 0.904 | 0.976 | 0.162 | 0 |
| test_unseen_domain | learned + names | 0.907–0.910 | **0.916–0.923** | 0.980 | 0.186 | 0 |
| test_unseen_domain | learned, generic | 0.907–0.909 | 0.920–0.922 | 0.981 | 0.177–0.183 | 0 |

Random-ranking NDCG: 0.807 (seen), 0.805 (unseen).

### Is the top-1 gain real? (seed 0, paired bootstrap, 5,000 resamples)

| Model | Split | n | Wins / losses vs. baseline | Δ top-1 | 95% CI |
|---|---|---|---|---|---|
| learned + names | test_seen | 214 | 11 / 4 | +0.033 | [+0.000, +0.070] |
| learned + names | test_unseen_domain | 561 | 17 / 8 | +0.016 | [+0.000, +0.034] |
| learned, generic | test_seen | 214 | 7 / 3 | +0.019 | [−0.009, +0.047] |
| learned, generic | test_unseen_domain | 561 | 16 / 7 | +0.016 | [+0.000, +0.032] |

## Findings

1. **The learned model improves top-1 consistently but modestly.** It wins more scenarios than it
   loses in every split and seed, but the confidence intervals reach zero. The seen-domain test set
   is small (214 complete scenarios); a bigger benchmark is needed for a firm claim.
2. **Attribute names help on familiar domains** (+3.3 points vs. +1.9 generic) **but not on unseen ones**,
   where both versions gain about +1.6. This is as expected: most unseen attribute names map to "unknown",
   so the model falls back on the generic shape, and that shape does transfer.
3. **Missed questions got slightly worse** (0.236 → ~0.247 seen, 0.162 → 0.186 unseen). The learned g never
   reaches exactly 0 or 1 (sigmoid output), so its intervals for missing values are narrower than the
   baseline's, and the policy is more confident than it should be. Phase 2b addresses this.
4. **No hard-limit violations** with any scorer: the policy filters options before any model sees them.
5. **Early stopping bug found and fixed.** Patience 60 stopped the generic model at epoch 2–4, effectively
   untrained, because validation accuracy dips early before improving. Patience is now 400 over 2,000 epochs.

## Decision

Keep the baseline as the default scorer for now. The learned scorer is available via
`decide(frame, scorer=LearnedScorer.from_checkpoint(...))`. Promote it only after phase 2b fixes the
missed-question regression and a larger benchmark confirms the top-1 gain.
