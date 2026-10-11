# Phase 6 — controlled learning from feedback

Built 2026-10-11. Nothing learns automatically; every step is an explicit command (see README).

## Pipeline

1. **Review** (`scripts/review_feedback.py list | accept | reject`): feedback starts as `pending`. Only
   `accepted` rows can be exported.
2. **Export** (`review_feedback.py export`): accepted feedback → `real-preferences-<sha>.jsonl` plus a manifest
   (count, train / held-out split, the feedback ids used, the rules). Training signal:
   - the chosen option beats every other option that met all hard limits;
   - weight 1.0 if the outcome was rated 4–5, 0.5 if 3; **skipped if 1–2** (regret doesn't say which option was
     right); with no outcome yet, satisfaction 4–5 counts 0.5, otherwise skipped;
   - choices of an option that broke a hard limit are not used;
   - decisions split 80/20 by a hash of their id, so held-out real decisions are never trained on.
3. **Retrain** (`train_preference.py --real-data ...`): synthetic pairs plus weighted real pairs
   (default ×5 per real pair), saved as a new candidate. Production is untouched.
4. **Gate** (`scripts/gate.py`): candidate vs. production through the same policy on the frozen benchmark
   (`synthetic-v1`, seed 1, 2,000 per domain, test splits only: 2,079 + 4,000 scenarios). Must pass all of:
   - zero hard-limit violations;
   - decision accuracy not worse: 95% paired-bootstrap lower bound of the difference ≥ −0.5 points, per split;
   - missed questions not worse: upper bound ≤ +1 point, per split;
   - agreement with held-out real choices not worse (lower bound ≥ −2 points), once there are ≥ 30 of them.
5. **Promote** (`gate.py --promote --reason ...`): only with a passing report for that exact checkpoint. Writes
   `production.json` (read by the API at start-up) and appends to `history.jsonl`. Rollback = promote the
   previous checkpoint.

## Test of the gate itself

| Candidate | Gate | Why |
|---|---|---|
| Baseline vs. itself (unit test) | pass | identical |
| Inverted scorer, ranks upside down (unit test) | **fail** | accuracy check |
| `pref-v2-s3` (same data, different seed) vs. `pref-v1-s0`, full benchmark | pass | see below |

`pref-v2-s3` vs. production:

| Check | Result |
|---|---|
| Hard-limit violations | 0 |
| Accuracy, test_seen | +0.00 pts, CI [−0.38, +0.38] |
| Accuracy, test_unseen_domain | −0.10 pts, CI [−0.35, +0.13] |
| Missed questions, test_seen | −0.16 pts, CI [−0.78, +0.47] |
| Missed questions, test_unseen_domain | +0.46 pts, CI [+0.09, +0.92]: slightly worse, within tolerance |
| Real choices | none reviewed yet |

**Not promoted:** a candidate that only matches production gives no reason to switch. The gate decides whether
a switch is *safe*; whether it is *worth it* stays a human call.

## Status

The pipeline is complete and tested, but **there is no real feedback yet**. Until reviewed real decisions exist,
retraining only reshuffles synthetic data. The next real step is using the app on real decisions.
