"""The extractor output contract, shared by every extractor (rules, Qwen, from-scratch).

Extractors return this small JSON-shaped structure; `to_frame` turns it into a
DecisionFrame. Everything extracted is marked INFERRED: a model read it from text,
so the UI should let the user confirm or correct it before relying on it.
"""

from __future__ import annotations

import re
from typing import Literal, Union

from pydantic import BaseModel, Field

from adapt.extraction.textgen import LEVEL_WEIGHT
from adapt.schema import (
    Constraint,
    Criterion,
    DecisionFrame,
    Direction,
    Operator,
    Option,
    Provenance,
    Value,
)

XScalar = Union[bool, int, float, str]

UNIT_ALIASES = {
    "₹": "INR", "rs": "INR", "inr": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "dollars": "USD",
    "min": "min", "mins": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hours": "h", "hours a week": "h",
    "%": "%", "percent": "%",
}


class XValue(BaseModel):
    value: XScalar
    unit: str | None = None


class XOption(BaseModel):
    label: str
    attributes: dict[str, XValue] = Field(default_factory=dict)


class XCriterion(BaseModel):
    attribute: str
    direction: Literal["maximize", "minimize"]
    priority: Literal["high", "medium", "low"] | None = None


class XConstraint(BaseModel):
    attribute: str
    operator: Literal["<=", ">=", "<", ">", "==", "!="]
    value: XScalar
    unit: str | None = None


class Extraction(BaseModel):
    options: list[XOption] = Field(default_factory=list)
    criteria: list[XCriterion] = Field(default_factory=list)
    constraints: list[XConstraint] = Field(default_factory=list)


def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def norm_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = unit.strip().lower()
    return UNIT_ALIASES.get(u, unit.strip()) or None


def to_frame(x: Extraction, situation: str, domain_hint: str | None = None) -> DecisionFrame:
    def val(v: XValue) -> Value:
        return Value(value=v.value, unit=norm_unit(v.unit), provenance=Provenance.INFERRED)

    seen: set[str] = set()
    options = []
    for i, o in enumerate(x.options):
        oid = chr(ord("a") + i) if i < 26 else f"o{i}"
        options.append(
            Option(
                id=oid,
                label=o.label,
                attributes={snake(k): val(v) for k, v in o.attributes.items()},
                provenance=Provenance.INFERRED,
            )
        )
    criteria = []
    for c in x.criteria:
        a = snake(c.attribute)
        if a in seen:  # one criterion per attribute
            continue
        seen.add(a)
        criteria.append(
            Criterion(
                attribute=a,
                direction=Direction(c.direction),
                weight=None if c.priority is None else LEVEL_WEIGHT[c.priority],
                provenance=Provenance.INFERRED,
            )
        )
    constraints = [
        # Units are dropped on limits: an extractor writing "₹" for the limit and "INR" for
        # the options would otherwise make every check UNKNOWN.
        Constraint(attribute=snake(c.attribute), operator=Operator(c.operator), value=c.value, provenance=Provenance.INFERRED)
        for c in x.constraints
    ]
    return DecisionFrame(
        situation=situation, domain_hint=domain_hint, options=options, criteria=criteria, constraints=constraints
    )
