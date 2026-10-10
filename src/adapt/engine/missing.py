"""Does a missing value change the decision? Answered by trying plausible values.

For every missing (option, attribute) "slot" that a criterion or hard limit needs,
we pick plausible fill values, re-run the complete-information decision (hard
limits, then scoring) for each combination, and see which options win.

- One option wins under every combination → the decision is robust; recommend it.
- The winner changes → ask about exactly the slots whose value made the difference.

Plausible values come from the criterion's `plausible_range` when one is given
(e.g. from the user or retrieved evidence). Otherwise from the known values,
stretched by `extension × spread` on each side, never below 0 when every known
value is non-negative. This is an assumption and is reported as one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from adapt.engine.constraints import Status, check
from adapt.engine.scoring import Scorer
from adapt.schema import Constraint, Criterion, Option, Provenance, Scalar, Value

COMBINATION_BUDGET = 400  # beyond this, each slot is varied on its own (approximate)
_TIE = 1e-9


@dataclass(frozen=True)
class Slot:
    option_id: str
    attribute: str


@dataclass
class FillResult:
    winners: set[str]  # options that win under every combination tried (may be empty)
    deciding: list[Slot]  # slots whose value changes the winner
    ranges: dict[Slot, tuple[Scalar, Scalar]] = field(default_factory=dict)
    approximate: bool = False


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def missing_slots(options: list[Option], attributes: list[str]) -> list[Slot]:
    return [Slot(o.id, a) for o in options for a in attributes if a not in o.attributes]


def fill_values(
    slot: Slot,
    all_options: list[Option],
    criteria: list[Criterion],
    constraints: list[Constraint],
    extension: float,
) -> list[Scalar] | None:
    """Plausible values for a slot, or None if they can't be bounded (e.g. free text)."""
    a = slot.attribute
    known = [o.attributes[a].value for o in all_options if a in o.attributes]
    limits = [c.value for c in constraints if c.attribute == a]
    if any(isinstance(x, bool) for x in [*known, *limits]):
        return [True, False]
    nums = sorted({float(x) for x in known if _is_number(x)})
    crit = next((c for c in criteria if c.attribute == a), None)

    if crit is not None and crit.plausible_range is not None:
        lo_b, hi_b = crit.plausible_range
    elif nums:
        lo, hi = nums[0], nums[-1]
        spread = (hi - lo) or (abs(lo) or 1.0)
        lo_b, hi_b = lo - extension * spread, hi + extension * spread
        if lo >= 0:
            lo_b = max(lo_b, 0.0)
    else:
        return None

    inside = [x for x in nums if lo_b <= x <= hi_b]
    mids = [(x + y) / 2 for x, y in zip(inside, inside[1:])]
    thresholds = [float(v) for v in limits if _is_number(v) and lo_b <= float(v) <= hi_b]
    return sorted({lo_b, hi_b, *inside, *mids, *thresholds})


def _leaders(
    options: list[Option],
    criteria: list[Criterion],
    constraints: list[Constraint],
    weights: dict[str, float],
    scorer: Scorer,
) -> frozenset[str]:
    """Winners with complete information: every option within _TIE of the best score."""
    feasible = [
        o for o in options if all(check(o, c).status is Status.PASS for c in constraints)
    ]
    if not feasible:
        return frozenset()
    scored = scorer(feasible, criteria, weights)
    top = max(s.low for s in scored)
    return frozenset(s.option_id for s in scored if s.low >= top - _TIE)


def _filled(options: list[Option], assignment: dict[Slot, Scalar]) -> list[Option]:
    out = []
    for o in options:
        extra = {
            s.attribute: Value(value=v, provenance=Provenance.ASSUMED)
            for s, v in assignment.items()
            if s.option_id == o.id
        }
        out.append(o.model_copy(update={"attributes": {**o.attributes, **extra}}) if extra else o)
    return out


def robust_winners(
    candidates: list[Option],
    all_options: list[Option],
    criteria: list[Criterion],
    constraints: list[Constraint],
    weights: dict[str, float],
    scorer: Scorer,
    slots: list[Slot],
    extension: float,
) -> FillResult | None:
    """None if some slot can't be bounded; the caller should ask about it."""
    fills = {s: fill_values(s, all_options, criteria, constraints, extension) for s in slots}
    if any(v is None for v in fills.values()):
        return None
    grids: dict[Slot, list[Scalar]] = fills  # type: ignore[assignment]
    ranges = {s: (v[0], v[-1]) for s, v in grids.items()}

    size = 1
    for v in grids.values():
        size *= len(v)
    if size <= COMBINATION_BUDGET:
        combos = [dict(zip(slots, vals)) for vals in itertools.product(*(grids[s] for s in slots))]
        results = [(c, _leaders(_filled(candidates, c), criteria, constraints, weights, scorer)) for c in combos]
        winners = set.intersection(*(set(r) for _, r in results))
        deciding = [s for s in slots if _slot_matters(s, results)]
        return FillResult(winners, _ask_list(winners, deciding, slots), ranges)

    # Too many combinations: vary one slot at a time, others at the middle of their range.
    base = {s: v[len(v) // 2] for s, v in grids.items()}
    winners: set[str] | None = None
    deciding = []
    for s in slots:
        outcomes = [
            _leaders(_filled(candidates, {**base, s: x}), criteria, constraints, weights, scorer)
            for x in grids[s]
        ]
        common = set.intersection(*(set(o) for o in outcomes))
        if not common:
            deciding.append(s)
        winners = common if winners is None else winners & common
    winners = winners or set()
    return FillResult(winners, _ask_list(winners, deciding, slots), ranges, approximate=True)


def _ask_list(winners: set[str], deciding: list[Slot], slots: list[Slot]) -> list[Slot]:
    """If no single slot explains the change (they only matter together), ask about all."""
    if winners:
        return []
    return deciding or list(slots)


def _slot_matters(slot: Slot, results: list[tuple[dict[Slot, Scalar], frozenset[str]]]) -> bool:
    """True if, holding every other slot fixed, changing this one changes who wins."""
    groups: dict[tuple, list[frozenset[str]]] = {}
    for combo, leaders in results:
        key = tuple(sorted((s.option_id, s.attribute, repr(v)) for s, v in combo.items() if s != slot))
        groups.setdefault(key, []).append(leaders)
    return any(not set.intersection(*(set(x) for x in g)) for g in groups.values())
