"""Regression gate: may a candidate scorer replace the production one?

Both scorers run through the SAME policy on the SAME frozen benchmark (synthetic test
splits only: never train or val) and, when there are enough, held-out real decisions.
Checks, all of which must pass:

1. hard-limit violations: the candidate has zero;
2. decision accuracy: not worse. The lower end of the 95% paired-bootstrap interval of
   (candidate − production) must be ≥ −tolerance, on each test split;
3. missed questions: not worse. The upper end of the interval of the difference must be
   ≤ tolerance;
4. real held-out decisions: agreement with what people chose is not worse, if there are at
   least `min_real` of them (fewer are reported but can't block or justify anything).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from adapt.engine import decide
from adapt.evaluation.metrics import evaluate
from adapt.lifecycle.feedback_data import RealExample
from adapt.synthetic import DATASET_VERSION, generate

TEST_SPLITS = ("test_seen", "test_unseen_domain")


@dataclass
class Tolerances:
    accuracy: float = 0.005  # may lose at most half a point at the CI's lower end
    missed_questions: float = 0.01
    real_agreement: float = 0.02
    min_real: int = 30


def _paired_ci(a: np.ndarray, b: np.ndarray, seed: int = 0, n: int = 4000) -> tuple[float, float, float]:
    """Mean of (b − a) and its 95% bootstrap interval, resampling scenarios."""
    d = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_gate(candidate, production, *, n_per_domain: int = 2000, seed: int = 1,
             real_heldout: list[RealExample] | None = None, tol: Tolerances | None = None) -> dict:
    """candidate / production: Scorer objects (with .version)."""
    tol = tol or Tolerances()
    scenarios = [s for s in generate(n_per_domain, seed) if s.truth.split in TEST_SPLITS]
    rows_p, _ = evaluate(scenarios, partial(decide, scorer=production))
    rows_c, _ = evaluate(scenarios, partial(decide, scorer=candidate))
    checks: list[dict] = []

    v = int(rows_c.violation.sum())
    checks.append({"check": "hard-limit violations", "passed": v == 0, "detail": f"candidate {v}"})

    metrics: dict[str, dict] = {}
    for split in TEST_SPLITS:
        mp, mc = rows_p.split == split, rows_c.split == split
        acc = _paired_ci(rows_p[mp].decision_ok.to_numpy(), rows_c[mc].decision_ok.to_numpy())
        must_ask_p = rows_p[mp & (rows_p.preferred == "clarify")]
        must_ask_c = rows_c[mc & (rows_c.preferred == "clarify")]
        missed = _paired_ci(must_ask_p.missed_question.to_numpy(), must_ask_c.missed_question.to_numpy())
        metrics[split] = {
            "n": int(mp.sum()),
            "accuracy_production": float(rows_p[mp].decision_ok.mean()),
            "accuracy_candidate": float(rows_c[mc].decision_ok.mean()),
            "accuracy_delta": acc,
            "missed_q_production": float(must_ask_p.missed_question.mean()),
            "missed_q_candidate": float(must_ask_c.missed_question.mean()),
            "missed_q_delta": missed,
        }
        checks.append({"check": f"accuracy not worse ({split})", "passed": acc[1] >= -tol.accuracy,
                       "detail": f"delta {acc[0]:+.4f}, 95% CI [{acc[1]:+.4f}, {acc[2]:+.4f}], tolerance -{tol.accuracy}"})
        checks.append({"check": f"missed questions not worse ({split})", "passed": missed[2] <= tol.missed_questions,
                       "detail": f"delta {missed[0]:+.4f}, 95% CI [{missed[1]:+.4f}, {missed[2]:+.4f}], tolerance +{tol.missed_questions}"})

    real = [e for e in (real_heldout or []) if e.split == "heldout"]
    if real:
        agree_p = np.array([decide(e.frame, scorer=production).recommended == e.chosen for e in real])
        agree_c = np.array([decide(e.frame, scorer=candidate).recommended == e.chosen for e in real])
        metrics["real_heldout"] = {"n": len(real), "agreement_production": float(agree_p.mean()),
                                   "agreement_candidate": float(agree_c.mean())}
        if len(real) >= tol.min_real:
            d = _paired_ci(agree_p, agree_c)
            checks.append({"check": "agreement with real choices not worse", "passed": d[1] >= -tol.real_agreement,
                           "detail": f"n={len(real)}, delta {d[0]:+.4f}, 95% CI [{d[1]:+.4f}, {d[2]:+.4f}]"})
        else:
            checks.append({"check": "agreement with real choices", "passed": True,
                           "detail": f"only {len(real)} held-out real decisions (< {tol.min_real}); reported, not gated"})
    else:
        checks.append({"check": "agreement with real choices", "passed": True,
                       "detail": "no reviewed real decisions yet; synthetic benchmark only"})

    return {
        "candidate_version": candidate.version, "production_version": production.version,
        "benchmark": {"dataset_version": DATASET_VERSION, "seed": seed, "n_per_domain": n_per_domain,
                      "splits": list(TEST_SPLITS)},
        "metrics": metrics, "checks": checks, "passed": all(c["passed"] for c in checks),
    }
