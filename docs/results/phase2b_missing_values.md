# Phase 2b — missing values, sensitivity, and a larger benchmark

Measured 2026-10-10. Policy `policy-0.2.0` vs. the phase 2a policy (`baseline-0.1.0`, commit `fcd0918`,
run from a temporary checkout on the same data). Model checkpoints are the phase 2a ones (`pref-v1-s0`,
`pref-v1-noattr-s0`), unchanged.

**Synthetic results only.** They show the mechanics work against known ground truth, not real-world quality.

## What changed

1. **Missing values: try plausible values instead of assuming a range.** For each missing value a criterion
   or hard limit needs, the engine tries plausible fills, re-runs the full decision for each combination, and
   recommends only if the same option wins every time. Otherwise it asks about exactly the values that changed
   the winner. Plausible range: `plausible_range` if given, else the known values stretched by
   `extension × spread` each way (floored at 0 for non-negative quantities).
2. **Hard limits are never assumed for the recommended option.** At `extension = 0` the first version recommended
   an option whose hidden value broke a hard limit (1 violation on val). The policy now asks instead.
3. **Exact weight-flip analysis.** Both scorers are additive, so the weight share at which another option wins is
   solved in closed form. Recommendations within 0.10 of a flip are marked `fragile`.
4. **Larger benchmark:** seed 1, 2,000 scenarios per domain (18,000 total): 2,079 test_seen and 4,000
   test_unseen_domain scenarios, vs. 657 / 1,200 before.

## Choosing `extension` (validation split only, baseline scorer)

| extension | decision acc. | missed questions | extra questions | violations |
|---|---|---|---|---|
| 0 | 0.900 | 0.167 | 0.011 | 0 (was 1 before the hard-limit rule) |
| 0.5 | 0.928 | 0.080 | 0.019 | 0 |
| 1 | 0.935 | 0.067 | 0.022 | 0 |
| **2 (chosen)** | **0.937** | **0.060** | **0.022** | 0 |
| 4 | 0.937 | 0.060 | 0.033 | 0 |
| real ranges given | 0.935 | 0.067 | 0.019 | 0 |

Giving the engine the *true* attribute ranges didn't beat the heuristic. The remaining misses come from the
scorer's preference shapes being wrong, not from bad range guesses.

## Test results (large benchmark, seed 1)

| Split | Policy + scorer | Decision acc. | Top-1 | Missed q | Extra q | Err when issued | Violations |
|---|---|---|---|---|---|---|---|
| test_seen | old + baseline | 0.900 | 0.896 | 0.179 | 0.024 | 0.161 | 0 |
| test_seen | old + learned | 0.910 | 0.944 | 0.222 | 0.008 | 0.140 | 0 |
| test_seen | **new + baseline** | 0.929 | 0.896 | 0.097 | 0.062 | 0.124 | 0 |
| test_seen | **new + learned** | **0.962** | **0.944** | **0.059** | 0.075 | **0.070** | 0 |
| test_unseen_domain | old + baseline | 0.913 | 0.892 | 0.122 | 0.015 | 0.138 | 0 |
| test_unseen_domain | old + learned | 0.918 | 0.923 | 0.157 | 0.007 | 0.128 | 0 |
| test_unseen_domain | **new + baseline** | 0.930 | 0.892 | 0.066 | 0.032 | 0.117 | 0 |
| test_unseen_domain | **new + learned** | **0.952** | **0.923** | **0.040** | 0.037 | **0.082** | 0 |

Generic learned model (no attribute names), new policy: 0.952 / 0.948 decision accuracy (seen / unseen).

### Significance (paired bootstrap, 5,000 resamples)

| Comparison | Split | n | Δ | 95% CI |
|---|---|---|---|---|
| Top-1, learned vs. baseline | test_seen | 675 | +0.047 | [+0.028, +0.068] |
| Top-1, learned vs. baseline | test_unseen_domain | 1,791 | +0.031 | [+0.018, +0.042] |
| Decision acc., new vs. old policy (baseline scorer) | test_seen | 2,079 | +0.028 | [+0.021, +0.036] |
| Decision acc., new vs. old policy (baseline scorer) | test_unseen_domain | 4,000 | +0.017 | [+0.013, +0.021] |
| Decision acc., new + learned vs. old + baseline | test_seen | 2,079 | +0.061 | [+0.050, +0.073] |
| Decision acc., new + learned vs. old + baseline | test_unseen_domain | 4,000 | +0.038 | [+0.031, +0.046] |

On the larger set, the phase 2a top-1 gain is now clearly real.

### Missing-value scenarios only

| Split | Version | Accuracy | Missed q | Extra q |
|---|---|---|---|---|
| test_seen (n=763) | old baseline | 0.820 | 0.356 | 0.066 |
| | new baseline | 0.898 | 0.192 | 0.170 |
| | new learned | 0.945 | 0.118 | 0.205 |
| test_unseen_domain (n=806) | old baseline | 0.809 | 0.379 | 0.077 |
| | new baseline | 0.892 | 0.205 | 0.171 |
| | new learned | 0.931 | 0.125 | 0.198 |

**Trade-off:** missed questions roughly halved, at the cost of more unnecessary ones (0.07 → ~0.17–0.20 in these
scenarios). For a decision tool, an extra question is much cheaper than a wrong recommendation, but it's a real cost.

### Does "fragile" flag risky recommendations?

| Split | Scorer | Fragile rate | Error if fragile | Error if robust |
|---|---|---|---|---|
| test_seen | baseline | 0.123 | 0.329 | 0.096 |
| test_seen | learned | 0.136 | 0.212 | 0.047 |
| test_unseen_domain | baseline | 0.155 | 0.351 | 0.073 |
| test_unseen_domain | learned | 0.152 | 0.292 | 0.044 |

Yes: fragile recommendations are wrong **3–6× as often** as robust ones. This is a usable uncertainty signal,
though it is not a calibrated probability and must not be shown as one.

## Decision

- Default `extension = 2.0` (chosen on val).
- The learned scorer `pref-v1-s0` is now the **recommended** scorer: it beats the baseline on top-1 with CIs above
  zero on both test splits, and with the new policy it also misses fewer questions. The baseline stays the
  code default (no torch needed) and the fallback.
