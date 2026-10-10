"""Synthetic scenario generator with known ground truth.

Every scenario is a DecisionFrame (what the engine sees) plus a ScenarioTruth (what
it doesn't): full attribute values, hidden value shapes, true utilities, the true
best option(s), and which outcomes are acceptable.

Scenario kinds:
- complete:       everything given → must recommend the true best
- missing_weight: priorities withheld → must ask, unless one option dominates
- missing_value:  one value withheld → must ask if that value could change the
                  true best, otherwise recommending is preferred
- infeasible:     hard limits exclude every option → must abstain
- no_options:     nothing to compare → must ask

Splits are by scenario *family* (domain, kind, criteria set) so the same family never
appears in two splits. Two whole domains are held out as test_unseen_domain.
"""

from __future__ import annotations

import hashlib
from typing import Iterator

import numpy as np
from pydantic import BaseModel

from adapt.schema import (
    Constraint,
    Criterion,
    DecisionFrame,
    Operator,
    Option,
    Provenance,
    Scalar,
    Value,
)
from adapt.synthetic.domains import DOMAINS, MIN, UNSEEN_DOMAINS, Attr, Domain, Flag

DATASET_VERSION = "synthetic-v1"
KINDS = ("complete", "missing_weight", "missing_value", "infeasible", "no_options")
KIND_PROBS = (0.45, 0.15, 0.2, 0.12, 0.08)
TIE_TOLERANCE = 0.01  # true utilities this close to the best also count as correct


class ScenarioTruth(BaseModel):
    scenario_id: str
    dataset_version: str = DATASET_VERSION
    domain: str
    kind: str
    family: str
    split: str
    true_values: dict[str, dict[str, Scalar]]
    true_weights: dict[str, float]
    shapes: dict[str, str]
    utilities: dict[str, float | None]  # None = violates a hard limit
    best_ids: list[str]
    acceptable: list[str]
    preferred: str


class Scenario(BaseModel):
    frame: DecisionFrame
    truth: ScenarioTruth


def _stated(x: Scalar, unit: str | None = None) -> Value:
    return Value(value=x, unit=unit, provenance=Provenance.STATED)


def _sample(rng: np.random.Generator, a: Attr) -> float | int:
    x = rng.uniform(a.lo, a.hi)
    return int(round(x)) if a.integer else round(float(x), 2)


def passes(x: Scalar, c: Constraint) -> bool:
    return {
        Operator.LE: lambda: x <= c.value,
        Operator.GE: lambda: x >= c.value,
        Operator.EQ: lambda: x == c.value,
    }[c.operator]()


def true_utilities(
    values: dict[str, dict[str, Scalar]],
    attrs: list[Attr],
    weights: dict[str, float],
    constraints: list[Constraint],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for oid, vals in values.items():
        if not all(passes(vals[c.attribute], c) for c in constraints):
            out[oid] = None
        else:
            out[oid] = sum(weights[a.name] * a.desirability(float(vals[a.name])) for a in attrs)
    return out


def best_ids(utilities: dict[str, float | None]) -> list[str]:
    feasible = {k: u for k, u in utilities.items() if u is not None}
    if not feasible:
        return []
    top = max(feasible.values())
    return sorted(k for k, u in feasible.items() if u >= top - TIE_TOLERANCE)


def _constraint(
    rng: np.random.Generator,
    values: dict[str, dict[str, Scalar]],
    attrs: list[Attr],
    flags: tuple[Flag, ...],
    infeasible: bool,
) -> Constraint:
    """A hard limit that leaves at least one option feasible, or none if infeasible."""
    ids = list(values)
    pool: list[Attr | Flag] = [*attrs, *flags]
    target = pool[rng.integers(len(pool))]

    if isinstance(target, Flag):
        if infeasible:
            for oid in ids:
                values[oid][target.name] = not target.good
        elif not any(values[o][target.name] == target.good for o in ids):
            values[ids[rng.integers(len(ids))]][target.name] = target.good
        return Constraint(attribute=target.name, operator=Operator.EQ, value=target.good)

    xs = sorted(float(values[o][target.name]) for o in ids)
    margin = (target.hi - target.lo) * 0.05
    if target.direction is MIN:  # "at most"
        threshold = xs[0] - margin if infeasible else xs[rng.integers(len(xs))]
        op = Operator.LE
    else:  # "at least"
        threshold = xs[-1] + margin if infeasible else xs[rng.integers(len(xs))]
        op = Operator.GE
    threshold = int(round(threshold)) if target.integer and not infeasible else round(threshold, 2)
    return Constraint(attribute=target.name, operator=op, value=threshold, unit=target.unit)


def _split(domain: str, family: str) -> str:
    if domain in UNSEEN_DOMAINS:
        return "test_unseen_domain"
    bucket = int(hashlib.sha256(family.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test_seen"


def make_scenario(rng: np.random.Generator, domain: Domain, kind: str, index: int) -> Scenario:
    k = int(rng.integers(2, min(4, len(domain.attrs)) + 1))
    attrs = [domain.attrs[i] for i in sorted(rng.choice(len(domain.attrs), size=k, replace=False))]
    weights = {a.name: round(float(w), 2) for a, w in zip(attrs, rng.dirichlet(np.full(k, 1.5)))}
    family = f"{domain.name}/{kind}/{'+'.join(a.name for a in attrs)}"

    n_opts = int(rng.integers(2, 7))
    ids = [chr(ord("a") + i) for i in range(n_opts)]
    values: dict[str, dict[str, Scalar]] = {
        oid: {
            **{a.name: _sample(rng, a) for a in attrs},
            **{f.name: bool(rng.random() < 0.7) == f.good for f in domain.flags},
        }
        for oid in ids
    }

    constraints: list[Constraint] = []
    if kind == "infeasible" or (kind != "no_options" and rng.random() < 0.5):
        constraints.append(_constraint(rng, values, attrs, domain.flags, kind == "infeasible"))

    units = {a.name: a.unit for a in attrs}
    options = [
        Option(
            id=oid,
            label=f"{domain.noun.title()} {oid.upper()}",
            attributes={name: _stated(x, units.get(name)) for name, x in values[oid].items()},
        )
        for oid in ids
    ]
    utilities = true_utilities(values, attrs, weights, constraints)
    best = best_ids(utilities)
    acceptable, preferred = ["recommend"], "recommend"

    if kind == "no_options":
        options, acceptable, preferred = [], ["clarify"], "clarify"
    elif kind == "infeasible":
        acceptable, preferred = ["abstain"], "abstain"
    elif kind == "missing_weight":
        dominant = _dominates(best, values, attrs, utilities)
        acceptable = ["recommend", "clarify"] if dominant else ["clarify"]
        preferred = "recommend" if dominant else "clarify"
    elif kind == "missing_value":
        needed = [a.name for a in attrs] + [c.attribute for c in constraints]
        hidden_attr = needed[rng.integers(len(needed))]
        hidden_opt = ids[rng.integers(len(ids))]
        del options[ids.index(hidden_opt)].attributes[hidden_attr]
        matters = _value_matters(hidden_opt, hidden_attr, values, attrs, weights, constraints, domain)
        acceptable = ["clarify"] if matters else ["recommend", "clarify"]
        preferred = "clarify" if matters else "recommend"

    frame = DecisionFrame(
        situation=f"Choosing between {n_opts} {domain.noun} options.",
        domain_hint=domain.name,
        options=options,
        criteria=[
            Criterion(
                attribute=a.name,
                direction=a.direction,
                weight=None if kind == "missing_weight" else weights[a.name],
            )
            for a in attrs
        ],
        constraints=constraints,
    )
    truth = ScenarioTruth(
        scenario_id=f"{domain.name}-{index:05d}",
        domain=domain.name,
        kind=kind,
        family=family,
        split=_split(domain.name, family),
        true_values=values,
        true_weights=weights,
        shapes={a.name: a.shape for a in attrs},
        utilities=utilities,
        best_ids=best,
        acceptable=acceptable,
        preferred=preferred,
    )
    return Scenario(frame=frame, truth=truth)


def _dominates(
    best: list[str], values: dict[str, dict[str, Scalar]], attrs: list[Attr], utilities: dict
) -> bool:
    """True if a true-best option is at least as good as every feasible option on every
    criterion, i.e. it wins whatever the user's priorities are."""
    feasible = [o for o, u in utilities.items() if u is not None]
    for b in best:
        ok = True
        for o in feasible:
            if o == b:
                continue
            signs = [
                (float(values[b][a.name]) - float(values[o][a.name])) * (-1 if a.direction is MIN else 1)
                for a in attrs
            ]
            if min(signs) < 0 or max(signs) <= 0:
                ok = False
                break
        if ok:
            return True
    return False


def _value_matters(
    oid: str,
    attr: str,
    values: dict[str, dict[str, Scalar]],
    attrs: list[Attr],
    weights: dict[str, float],
    constraints: list[Constraint],
    domain: Domain,
) -> bool:
    """Would the true best change depending on the hidden value?"""
    spec = next((a for a in domain.attrs if a.name == attr), None)
    fills: list[Scalar] = [True, False] if spec is None else [spec.lo, (spec.lo + spec.hi) / 2, spec.hi]
    outcomes = set()
    for x in fills:
        trial = {k: dict(v) for k, v in values.items()}
        trial[oid][attr] = x
        outcomes.add(tuple(best_ids(true_utilities(trial, attrs, weights, constraints))))
    return len(outcomes) > 1


def generate(n_per_domain: int = 600, seed: int = 0) -> Iterator[Scenario]:
    rng = np.random.default_rng(seed)
    for domain in DOMAINS:
        for i in range(n_per_domain):
            kind = KINDS[rng.choice(len(KINDS), p=KIND_PROBS)]
            yield make_scenario(rng, domain, kind, i)
