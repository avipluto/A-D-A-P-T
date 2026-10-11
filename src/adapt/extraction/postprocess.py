"""Deterministic clean-up of any extractor's output. Optional and separately measured.

It never changes an extractor; it takes an Extraction plus the original text and
returns a cleaned copy and a log of what it changed. Every extractor is reported both
raw and post-processed, so an extractor's own quality is never mixed up with this step.

Two fixes, both domain-independent:

1. Hard limits need limit language. A limit is kept only if a sentence that states
   its value (or names its attribute, for yes/no limits) also uses ordinary English
   limit words ("can't", "must", "at most", "no more than", ...). This targets limits
   invented from preferences ("lower price is better" → price <= 2785).

2. One name per attribute. Two attribute names are merged when no option has both,
   their values are of the same kind and similar magnitude, and together they cover
   more options. This targets "stops" for some options and "layovers" for others.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from adapt.extraction.schema import Extraction, XConstraint, norm_unit

LIMIT_WORDS = re.compile(
    r"\b(can't|cannot|can not|must|mustn't|has to|have to|need(s)? to|needs?|required|at most|at least|"
    r"no more than|no less than|not more than|not less than|maximum|minimum|max|min|up to|"
    r"under|below|over|above|limit|only if|won't|wouldn't accept)\b",
    re.IGNORECASE,
)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class CleanLog:
    dropped_limits: list[str] = field(default_factory=list)
    merged_names: list[tuple[str, str]] = field(default_factory=list)  # (removed, kept)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _numbers(sentence: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUM.findall(sentence)}


def _same_number(x: float, nums: set[float]) -> bool:
    return any(abs(x - n) <= 1e-6 * max(1.0, abs(n)) or abs(x - n * 1000) < 1e-3 or abs(x * 1000 - n) < 1e-3 for n in nums)


def _limit_is_stated(c: XConstraint, sentences: list[str]) -> bool:
    if isinstance(c.value, bool) or not isinstance(c.value, (int, float)):
        words = [w for w in c.attribute.lower().replace("_", " ").split() if len(w) > 2]
        return any(LIMIT_WORDS.search(s) and any(w in s.lower() for w in words) for s in sentences)
    return any(LIMIT_WORDS.search(s) and _same_number(float(c.value), _numbers(s)) for s in sentences)


def _kind(v: object) -> str:
    return "bool" if isinstance(v, bool) else "num" if isinstance(v, (int, float)) else "text"


def _merge_plan(x: Extraction) -> dict[str, str]:
    """Map from names to remove → name to keep."""
    holders: dict[str, set[int]] = {}
    values: dict[str, list] = {}
    units: dict[str, set] = {}
    for i, o in enumerate(x.options):
        for k, v in o.attributes.items():
            holders.setdefault(k, set()).add(i)
            values.setdefault(k, []).append(v.value)
            units.setdefault(k, set()).add(norm_unit(v.unit))

    def compatible(a: str, b: str) -> bool:
        if holders[a] & holders[b]:
            return False
        ka, kb = {_kind(v) for v in values[a]}, {_kind(v) for v in values[b]}
        if ka != kb or len(ka) != 1:
            return False
        ua, ub = units[a] - {None}, units[b] - {None}
        if ua and ub and not (ua & ub):
            return False
        if ka == {"num"}:
            ma = statistics.median(abs(float(v)) for v in values[a]) or 1e-9
            mb = statistics.median(abs(float(v)) for v in values[b]) or 1e-9
            if max(ma, mb) / min(ma, mb) > 10:
                return False
        return True

    plan: dict[str, str] = {}
    names = sorted(holders, key=lambda n: -len(holders[n]))
    for small in list(reversed(names)):  # copy: names shrinks as we merge
        if small in plan or small not in holders:
            continue
        candidates = [big for big in names if big != small and big not in plan and compatible(small, big)
                      and len(holders[big]) >= len(holders[small])]
        if len(candidates) == 1:  # merge only when the match is unambiguous
            big = candidates[0]
            plan[small] = big
            holders[big] |= holders.pop(small)
            values[big] += values.pop(small)
            units[big] |= units.pop(small)
            names.remove(small)
    return plan


def clean(x: Extraction, text: str, limits: bool = True, names: bool = True) -> tuple[Extraction, CleanLog]:
    log = CleanLog()
    sentences = _sentences(text)
    constraints = list(x.constraints)
    if limits:
        kept = []
        for c in constraints:
            if _limit_is_stated(c, sentences):
                kept.append(c)
            else:
                log.dropped_limits.append(f"{c.attribute} {c.operator} {c.value}")
        constraints = kept

    options, criteria = list(x.options), list(x.criteria)
    if names:
        plan = _merge_plan(x)
        log.merged_names = list(plan.items())
        options = [
            o.model_copy(update={"attributes": {plan.get(k, k): v for k, v in o.attributes.items()}}) for o in options
        ]
        criteria = [c.model_copy(update={"attribute": plan.get(c.attribute, c.attribute)}) for c in criteria]
        constraints = [c.model_copy(update={"attribute": plan.get(c.attribute, c.attribute)}) for c in constraints]

        # A criterion or limit naming no option attribute ("growth prospects" when the options
        # say "growth score") goes to the ONE option attribute sharing a meaningful word with it.
        option_names = {k for o in options for k in o.attributes}
        def resolve(name: str) -> str:
            if name in option_names:
                return name
            words = {w for w in name.split("_") if len(w) > 3}
            hits = [n for n in option_names if words & {w for w in n.split("_") if len(w) > 3}]
            if len(hits) == 1:
                log.merged_names.append((name, hits[0]))
                return hits[0]
            return name
        criteria = [c.model_copy(update={"attribute": resolve(c.attribute)}) for c in criteria]
        constraints = [c.model_copy(update={"attribute": resolve(c.attribute)}) for c in constraints]

    return Extraction(options=options, criteria=criteria, constraints=constraints), log
