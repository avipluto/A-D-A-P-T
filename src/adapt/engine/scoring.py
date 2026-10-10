"""Transparent baseline scoring: weighted sum of min-max normalised criteria.

Scores are relative to the options being compared and are NOT probabilities.
A missing value produces a score interval: the low end assumes the option is the
worst observed on that criterion, the high end assumes it is the best.

Assumption: a missing value lies within the range of the known values. A value far
outside it would rescale everyone's normalised score; the baseline doesn't model that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adapt.schema import Criterion, Direction, Option


@dataclass
class Scored:
    option_id: str
    low: float
    high: float
    contributions: dict[str, tuple[float, float]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2


def criterion_values(options: list[Option], c: Criterion) -> dict[str, float]:
    """Numeric values for one criterion, keyed by option id. Missing, non-numeric
    or unit-mismatched values are left out."""
    units = [o.attributes[c.attribute].unit for o in options if c.attribute in o.attributes]
    ref_unit = next((u for u in units if u is not None), None)
    out: dict[str, float] = {}
    for o in options:
        v = o.attributes.get(c.attribute)
        if v is None or isinstance(v.value, str):
            continue
        if v.unit is not None and ref_unit is not None and v.unit != ref_unit:
            continue
        out[o.id] = float(v.value)
    return out


def score_options(
    options: list[Option], criteria: list[Criterion], weights: dict[str, float]
) -> list[Scored]:
    total = sum(weights[c.attribute] for c in criteria)
    if total <= 0:
        raise ValueError("at least one criterion needs a positive weight")

    scored = {o.id: Scored(o.id, 0.0, 0.0) for o in options}
    for c in criteria:
        w = weights[c.attribute] / total
        vals = criterion_values(options, c)
        lo, hi = (min(vals.values()), max(vals.values())) if vals else (0.0, 0.0)
        # With missing values but no spread among the known ones, a missing value could
        # land above or below them, so the known options are uncertain too.
        degenerate = hi == lo and len(vals) < len(options)
        for o in options:
            s = scored[o.id]
            if o.id in vals and degenerate:
                part = (0.0, w)
            elif o.id in vals:
                n = 0.5 if hi == lo else (vals[o.id] - lo) / (hi - lo)
                if c.direction is Direction.MINIMIZE:
                    n = 1.0 - n
                part = (w * n, w * n)
            else:
                part = (0.0, w)
                s.missing.append(c.attribute)
            s.contributions[c.attribute] = part
            s.low += part[0]
            s.high += part[1]
    return list(scored.values())


def dominant_option(options: list[Option], criteria: list[Criterion]) -> str | None:
    """The option at least as good as every other on every criterion and strictly
    better than each on at least one. Needs no weights. None if no such option or
    any value is missing."""
    table = {c.attribute: criterion_values(options, c) for c in criteria}
    if any(len(vals) != len(options) for vals in table.values()):
        return None

    def better_or_equal(a: str, b: str) -> tuple[bool, bool]:
        strict = False
        for c in criteria:
            x, y = table[c.attribute][a], table[c.attribute][b]
            if c.direction is Direction.MINIMIZE:
                x, y = -x, -y
            if x < y:
                return False, False
            strict |= x > y
        return True, strict

    for cand in options:
        if all(
            other.id == cand.id or all(better_or_equal(cand.id, other.id))
            for other in options
        ):
            return cand.id
    return None
