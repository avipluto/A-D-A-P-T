"""Evaluation of any decide() function against synthetic ground truth.

Per scenario we record whether the outcome type was acceptable, whether a
recommendation picked a true best option, ranking quality (NDCG) on complete
scenarios, and any hard-limit violations checked against the TRUE values
(including values hidden from the engine).
"""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from adapt.engine import DecisionResult
from adapt.schema import DecisionFrame
from adapt.synthetic.generator import Scenario, passes

Decider = Callable[[DecisionFrame], DecisionResult]


def relative_gains(utilities: dict[str, float]) -> dict[str, float]:
    """Utility above the worst feasible option. Raw utilities are all similar in size,
    which lets even a random ranking score ~0.93 NDCG; shifting fixes that."""
    floor = min(utilities.values(), default=0.0)
    return {k: u - floor for k, u in utilities.items()}


def random_ndcg(gains: dict[str, float], trials: int = 50, seed: int = 0) -> float | None:
    """Expected NDCG of a random ranking: the floor any ranking should clear."""
    import random

    rng = random.Random(seed)
    ids = list(gains)
    vals = []
    for _ in range(trials):
        rng.shuffle(ids)
        vals.append(ndcg(ids, gains))
    return None if vals[0] is None else sum(vals) / trials


def ndcg(ranked_ids: list[str], gains: dict[str, float]) -> float | None:
    ranked = [gains[i] for i in ranked_ids if i in gains]
    ideal = sorted(gains.values(), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg <= 0 or len(ideal) < 2:
        return None
    return sum(g / math.log2(i + 2) for i, g in enumerate(ranked)) / idcg


def violates(s: Scenario, option_id: str) -> bool:
    vals = s.truth.true_values[option_id]
    return not all(passes(vals[c.attribute], c) for c in s.frame.constraints)


def score_one(s: Scenario, decide: Decider) -> dict:
    r = decide(s.frame)
    t = s.truth
    outcome = r.outcome.value
    rec = r.recommended
    rec_correct = rec in t.best_ids if rec else None
    gains = relative_gains({k: u for k, u in t.utilities.items() if u is not None})
    is_complete = t.kind == "complete"
    return {
        "scenario_id": t.scenario_id,
        "domain": t.domain,
        "kind": t.kind,
        "split": t.split,
        "outcome": outcome,
        "preferred": t.preferred,
        "outcome_ok": outcome in t.acceptable,
        "decision_ok": outcome in t.acceptable and rec_correct is not False,
        "recommended": rec is not None,
        "rec_correct": rec_correct,
        "violation": bool(rec and violates(s, rec)),
        "missed_question": t.acceptable == ["clarify"] and outcome == "recommend",
        "unnecessary_question": t.preferred == "recommend" and outcome == "clarify",
        "ndcg": ndcg([x.option_id for x in r.ranking], gains) if is_complete else None,
        "ndcg_random": random_ndcg(gains) if is_complete else None,
    }


def summarise(rows: pd.DataFrame) -> dict:
    recs = rows[rows.recommended]
    complete = rows[rows.kind == "complete"]
    infeasible = rows[rows.kind == "infeasible"]
    prefer_rec = rows[rows.preferred == "recommend"]
    must_ask = rows[rows.preferred == "clarify"]
    return {
        "n": int(len(rows)),
        "decision_accuracy": float(rows.decision_ok.mean()),
        "outcome_accuracy": float(rows.outcome_ok.mean()),
        "top1_complete": float(complete.rec_correct.eq(True).mean()) if len(complete) else None,
        "ndcg_complete": float(complete.ndcg.dropna().mean()) if complete.ndcg.notna().any() else None,
        "ndcg_random": float(complete.ndcg_random.dropna().mean()) if complete.ndcg_random.notna().any() else None,
        "coverage": float(rows.recommended.mean()),
        "error_rate_issued": float((~recs.decision_ok).mean()) if len(recs) else None,
        "missed_question_rate": float(must_ask.missed_question.mean()) if len(must_ask) else None,
        "unnecessary_question_rate": float(prefer_rec.unnecessary_question.mean()) if len(prefer_rec) else None,
        "abstain_recall_infeasible": float((infeasible.outcome == "abstain").mean()) if len(infeasible) else None,
        "constraint_violations": int(rows.violation.sum()),
    }


def evaluate(scenarios: list[Scenario], decide: Decider) -> tuple[pd.DataFrame, dict]:
    rows = pd.DataFrame([score_one(s, decide) for s in scenarios])
    report = {
        "overall": summarise(rows),
        "by_split": {k: summarise(g) for k, g in rows.groupby("split")},
        "by_kind": {k: summarise(g) for k, g in rows.groupby("kind")},
        "by_domain": {k: summarise(g) for k, g in rows.groupby("domain")},
    }
    return rows, report
