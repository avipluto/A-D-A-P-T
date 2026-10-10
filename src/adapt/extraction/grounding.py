"""Drop extracted numbers that don't appear in the text.

A language model can produce plausible-looking values and limits that the person
never stated ("cost <= 100000" when no such limit was mentioned). Every numeric
option value and every numeric hard limit must match a number written in the text
(allowing "$120k" ↔ 120 or 120000). Yes/no values and text values are kept: they
can't be checked this way.

This is a cheap guard against unsupported claims, not a proof of correctness: a
number can appear in the text and still be attached to the wrong option.
"""

from __future__ import annotations

import re

from adapt.extraction.schema import Extraction

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def text_numbers(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUM.findall(text)}


def _grounded(x: object, nums: set[float]) -> bool:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return True
    x = float(x)
    return any(abs(x - n) <= 1e-6 * max(1.0, abs(n)) or abs(x - n * 1000) < 1e-3 or abs(x * 1000 - n) < 1e-3 for n in nums)


def ground(x: Extraction, text: str) -> tuple[Extraction, int]:
    """Copy of the extraction without ungrounded numbers, and how many were dropped."""
    nums = text_numbers(text)
    dropped = 0
    options = []
    for o in x.options:
        keep = {k: v for k, v in o.attributes.items() if _grounded(v.value, nums)}
        dropped += len(o.attributes) - len(keep)
        options.append(o.model_copy(update={"attributes": keep}))
    constraints = [c for c in x.constraints if _grounded(c.value, nums)]
    dropped += len(x.constraints) - len(constraints)
    return x.model_copy(update={"options": options, "constraints": constraints}), dropped
