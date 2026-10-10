"""Per-(option, criterion) features shared by training and inference.

Using the same function in both places means the model can't silently see
different inputs at training time than at decision time.

Features (all scale-free, so they mean the same thing in any domain or unit):
  n   — desirability position among the options: 0 = worst, 1 = best
  cv  — spread of the values (std / |mean|, clipped), i.e. how different the options are
  k   — number of options / 6
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from adapt.engine.scoring import criterion_values
from adapt.schema import Criterion, Direction, Option

FEATURE_DIM = 3
GRID = tuple(i / 10 for i in range(11))  # positions tried for a missing value


@dataclass
class CriterionFeatures:
    known: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    cv: float = 0.0
    k: float = 0.0

    def grid_rows(self) -> list[tuple[float, float, float]]:
        """Feature rows for a missing value placed at every grid position."""
        return [(n, self.cv, self.k) for n in GRID]


def criterion_features(options: list[Option], c: Criterion) -> CriterionFeatures:
    vals = criterion_values(options, c)
    out = CriterionFeatures(k=len(options) / 6)
    if vals:
        xs = list(vals.values())
        mean = statistics.fmean(xs)
        std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        out.cv = min(std / (abs(mean) + 1e-9), 3.0)
        lo, hi = min(xs), max(xs)
    degenerate = bool(vals) and lo == hi and len(vals) < len(options)
    for o in options:
        if o.id not in vals or degenerate:  # mirrors the baseline's interval rule
            out.missing.append(o.id)
            continue
        n = 0.5 if hi == lo else (vals[o.id] - lo) / (hi - lo)
        if c.direction is Direction.MINIMIZE:
            n = 1.0 - n
        out.known[o.id] = (n, out.cv, out.k)
    return out
