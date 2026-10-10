"""Deterministic hard-constraint checks.

A constraint either passes, fails, or is UNKNOWN because the value is missing or not
comparable. UNKNOWN is never silently treated as a pass: the policy decides whether
it matters enough to ask about.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum

from adapt.schema import Constraint, Operator, Option

_OPS = {
    Operator.LE: operator.le,
    Operator.GE: operator.ge,
    Operator.LT: operator.lt,
    Operator.GT: operator.gt,
    Operator.EQ: operator.eq,
    Operator.NE: operator.ne,
}
_ORDERED = {Operator.LE, Operator.GE, Operator.LT, Operator.GT}


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Check:
    constraint: Constraint
    status: Status
    reason: str = ""


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check(option: Option, c: Constraint) -> Check:
    v = option.attributes.get(c.attribute)
    if v is None:
        return Check(c, Status.UNKNOWN, f"{c.attribute} is missing")
    if c.unit and v.unit and c.unit != v.unit:
        return Check(c, Status.UNKNOWN, f"unit mismatch: {v.unit} vs {c.unit}")
    if c.operator in _ORDERED and not (_is_number(v.value) and _is_number(c.value)):
        return Check(c, Status.UNKNOWN, f"cannot order non-numeric values for {c.attribute}")
    ok = _OPS[c.operator](v.value, c.value)
    return Check(c, Status.PASS if ok else Status.FAIL)


def describe(c: Constraint) -> str:
    unit = f" {c.unit}" if c.unit else ""
    return f"{c.attribute} {c.operator.value} {c.value}{unit}"
