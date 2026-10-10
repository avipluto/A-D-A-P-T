"""Field-level and end-to-end scores for an extractor.

Field level (micro precision / recall / F1 over all scenarios):
  options      — option labels found
  values       — (option, attribute, value) triples
  criteria     — (attribute, direction) pairs; priority accuracy over the matched ones
  constraints  — (attribute, operator, value) triples

Attribute names are matched loosely: an extracted "commute_time" counts for gold
"commute_minutes" because both are names the text could use. Values must match to
within 1% (or be the same number written in thousands, e.g. $120k as 120 or 120000).

End to end: decide() on the extracted frame vs. on the gold frame. Same outcome,
and the same recommended option, is what actually matters to a user.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from adapt.engine import decide
from adapt.extraction.schema import Extraction, to_frame
from adapt.extraction.textgen import PHRASES, gold_frame, level, render
from adapt.schema import DecisionFrame
from adapt.synthetic.generator import Scenario

Extractor = Callable[[str], Extraction]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def aliases(attr: str) -> set[str]:
    return {_norm(attr), *(_norm(p) for p in PHRASES.get(attr, ()))}


def attr_match(x: str, gold: str) -> bool:
    """Same attribute if the names share an alias, or the extracted name contains one of
    the gold names as whole words (e.g. "commute time minutes" ⊇ "commute")."""
    gx, gg = aliases(x), aliases(gold)
    if gx & gg:
        return True
    padded = f" {_norm(x)} "
    return any(f" {a} " in padded for a in gg)


def value_match(x, g) -> bool:
    if isinstance(g, bool) or isinstance(x, bool):
        return isinstance(x, bool) and isinstance(g, bool) and x == g
    try:
        x, g = float(x), float(g)
    except (TypeError, ValueError):
        return False
    tol = 0.01 * max(1.0, abs(g))
    return abs(x - g) <= tol or abs(x - g * 1000) <= tol * 1000 or abs(x * 1000 - g) <= tol


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int, n_pred: int, n_gold: int) -> None:
        self.tp += tp
        self.fp += n_pred - tp
        self.fn += n_gold - tp

    def prf(self) -> dict[str, float | None]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else None
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else None
        f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        return {"precision": p, "recall": r, "f1": f}


def _greedy(pred: list, gold: list, ok: Callable) -> list[tuple[int, int]]:
    used, pairs = set(), []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j not in used and ok(p, g):
                used.add(j)
                pairs.append((i, j))
                break
    return pairs


@dataclass
class ScenarioScore:
    counts: dict[str, Counts] = field(default_factory=dict)
    priority_hits: int = 0
    priority_total: int = 0
    option_map: dict[str, str] = field(default_factory=dict)  # extracted id → gold id
    # gold attribute → extracted names used for it. More than one name for the same thing
    # (e.g. "monthly_cost" and "monthly_cost_usd") breaks the decision even if each field "matches".
    names: dict[str, set[str]] = field(default_factory=dict)

    @property
    def consistent(self) -> bool:
        return all(len(v) == 1 for v in self.names.values())


def score_fields(x: DecisionFrame, gold: DecisionFrame) -> ScenarioScore:
    sc = ScenarioScore(counts={k: Counts() for k in ("options", "values", "criteria", "constraints")})

    opt_pairs = _greedy(x.options, gold.options, lambda a, b: _norm(a.label) == _norm(b.label))
    sc.counts["options"].add(len(opt_pairs), len(x.options), len(gold.options))
    sc.option_map = {x.options[i].id: gold.options[j].id for i, j in opt_pairs}

    pred_vals = [(o.id, a, v.value) for o in x.options for a, v in o.attributes.items()]
    gold_vals = [(o.id, a, v.value) for o in gold.options for a, v in o.attributes.items()]
    hits = _greedy(
        pred_vals, gold_vals,
        lambda p, g: sc.option_map.get(p[0]) == g[0] and attr_match(p[1], g[1]) and value_match(p[2], g[2]),
    )
    sc.counts["values"].add(len(hits), len(pred_vals), len(gold_vals))
    for i, j in hits:
        sc.names.setdefault(gold_vals[j][1], set()).add(pred_vals[i][1])

    crit = _greedy(x.criteria, gold.criteria, lambda p, g: attr_match(p.attribute, g.attribute) and p.direction == g.direction)
    sc.counts["criteria"].add(len(crit), len(x.criteria), len(gold.criteria))
    for i, j in crit:
        sc.names.setdefault(gold.criteria[j].attribute, set()).add(x.criteria[i].attribute)
        g = gold.criteria[j]
        if g.weight is not None:
            sc.priority_total += 1
            p = x.criteria[i]
            sc.priority_hits += p.weight is not None and _level_of(p.weight) == _level_of(g.weight)

    cons = _greedy(
        x.constraints, gold.constraints,
        lambda p, g: attr_match(p.attribute, g.attribute) and p.operator == g.operator and value_match(p.value, g.value),
    )
    sc.counts["constraints"].add(len(cons), len(x.constraints), len(gold.constraints))
    for i, j in cons:
        sc.names.setdefault(gold.constraints[j].attribute, set()).add(x.constraints[i].attribute)
    return sc


def _level_of(weight: float) -> str:
    return {3.0: "high", 2.0: "medium", 1.0: "low"}.get(weight, level(weight))


def evaluate_extractor(scenarios: list[Scenario], extractor: Extractor, text_seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rows, totals = [], {k: Counts() for k in ("options", "values", "criteria", "constraints")}
    pri_hit = pri_tot = 0
    for s in scenarios:
        text = render(s, text_seed)
        gold = gold_frame(s)
        t0 = time.perf_counter()
        try:
            x = to_frame(extractor(text), s.frame.situation)
            failed = False
        except Exception:  # an extractor that crashes or returns invalid output scores zero
            x = DecisionFrame(situation=s.frame.situation)
            failed = True
        ms = (time.perf_counter() - t0) * 1000

        sc = score_fields(x, gold)
        for k, c in sc.counts.items():
            totals[k].add(c.tp, c.tp + c.fp, c.tp + c.fn)
        pri_hit += sc.priority_hits
        pri_tot += sc.priority_total

        rg, rx = decide(gold), decide(x)
        rx_rec = sc.option_map.get(rx.recommended) if rx.recommended else None
        rows.append({
            "scenario_id": s.truth.scenario_id,
            "domain": s.truth.domain,
            "kind": s.truth.kind,
            "split": s.truth.split,
            "failed": failed,
            "ms": ms,
            "consistent_names": sc.consistent,
            "perfect": (
                all(c.fp == 0 and c.fn == 0 for c in sc.counts.values())
                and sc.priority_hits == sc.priority_total
                and sc.consistent
            ),
            "outcome_agree": rx.outcome == rg.outcome,
            "decision_agree": rx.outcome == rg.outcome and rx_rec == rg.recommended,
            # the worst failure: extracted frame leads to recommending something the gold frame wouldn't
            "wrong_recommendation": rx.recommended is not None and rx_rec != rg.recommended,
        })

    df = pd.DataFrame(rows)
    report = {
        "n": len(df),
        **{k: c.prf() for k, c in totals.items()},
        "priority_accuracy": pri_hit / pri_tot if pri_tot else None,
        "naming_consistency": float(df.consistent_names.mean()),
        "perfect_rate": float(df.perfect.mean()),
        "outcome_agreement": float(df.outcome_agree.mean()),
        "decision_agreement": float(df.decision_agree.mean()),
        "wrong_recommendation_rate": float(df.wrong_recommendation.mean()),
        "failure_rate": float(df.failed.mean()),
        "ms_per_text": float(df.ms.mean()),
    }
    return df, report
