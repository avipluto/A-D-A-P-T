"""The additive learned scorer, independent of the maths library that runs the network.

    score(option) = Σ_c  ŵ_c · g(features_c, attribute_c)

This module builds the feature rows (one per option and criterion, or a grid of rows
for a missing value) and turns g's outputs into interval scores. Subclasses only
provide g: PyTorch for training and research (learning/model.py), NumPy for a light
deployment without PyTorch (inference/numpy_models.py). Both use this same code, so
they can't drift apart.
"""

from __future__ import annotations

from adapt.engine.scoring import Scored
from adapt.learning.features import GRID, criterion_features
from adapt.schema import Criterion, Option


class AdditiveScorer:
    version: str

    def attr_index(self, name: str) -> int:
        raise NotImplementedError

    def values(self, rows: list[tuple[float, float, float]], attrs: list[int]) -> list[float]:
        """g in (0, 1) for each feature row."""
        raise NotImplementedError

    def __call__(
        self, options: list[Option], criteria: list[Criterion], weights: dict[str, float]
    ) -> list[Scored]:
        total = sum(weights[c.attribute] for c in criteria)
        if total <= 0:
            raise ValueError("at least one criterion needs a positive weight")

        rows: list[tuple[float, float, float]] = []
        attrs: list[int] = []
        plan: list[tuple[str, str, int, int]] = []  # (option, attribute, first row, n rows)
        for c in criteria:
            f = criterion_features(options, c)
            idx = self.attr_index(c.attribute)
            for o in options:
                start = len(rows)
                new = [f.known[o.id]] if o.id in f.known else f.grid_rows()
                rows += new
                attrs += [idx] * len(new)
                plan.append((o.id, c.attribute, start, len(new)))

        g = self.values(rows, attrs)

        scored = {o.id: Scored(o.id, 0.0, 0.0) for o in options}
        for oid, attr, start, n in plan:
            w = weights[attr] / total
            vals = g[start : start + n]
            unit = (min(vals), max(vals))
            part = (w * unit[0], w * unit[1])
            s = scored[oid]
            s.unit[attr] = unit
            s.contributions[attr] = part
            s.low += part[0]
            s.high += part[1]
            if n == len(GRID):
                s.missing.append(attr)
        return list(scored.values())
