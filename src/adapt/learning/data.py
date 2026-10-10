"""Turn synthetic scenarios into pairwise training tensors.

Only options that pass every hard limit are used: the policy filters the others
out before any scorer runs, so the model never needs to rank them. Training uses
the TRUE values and weights from the scenario truth (including ones hidden from
the engine in that scenario), so missing_value and missing_weight scenarios still
provide complete comparisons to learn from.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adapt.learning.features import criterion_features
from adapt.schema import Criterion, Option, Provenance, Value
from adapt.synthetic.generator import Scenario

LEARNABLE_KINDS = {"complete", "missing_value", "missing_weight"}
MIN_GAP = 1e-6  # pairs with (almost) equal true utility carry no preference signal


@dataclass
class PairData:
    feats: torch.Tensor  # [R, FEATURE_DIM]
    attr: torch.Tensor  # [R] attribute index
    weight: torch.Tensor  # [R] normalised criterion weight
    row_option: torch.Tensor  # [R] global option index
    pairs: torch.Tensor  # [P, 2] (better, worse) option indices
    n_options: int
    n_scenarios: int

    def to(self, device: torch.device) -> PairData:
        return PairData(
            self.feats.to(device), self.attr.to(device), self.weight.to(device),
            self.row_option.to(device), self.pairs.to(device), self.n_options, self.n_scenarios,
        )


def true_options(s: Scenario) -> tuple[list[Option], list[Criterion], dict[str, float]]:
    """Feasible options rebuilt from the true values, plus criteria with true weights."""
    feasible = [o for o, u in s.truth.utilities.items() if u is not None]
    options = [
        Option(
            id=o,
            label=o,
            attributes={
                k: Value(value=v, provenance=Provenance.STATED)
                for k, v in s.truth.true_values[o].items()
            },
        )
        for o in feasible
    ]
    criteria = [c.model_copy(update={"weight": s.truth.true_weights[c.attribute]}) for c in s.frame.criteria]
    return options, criteria, s.truth.true_weights


def vocabulary(scenarios: list[Scenario]) -> list[str]:
    return sorted({c.attribute for s in scenarios for c in s.frame.criteria})


def build_pairs(scenarios: list[Scenario], attr_index) -> PairData:
    feats, attr, weight, row_option, pairs = [], [], [], [], []
    n_opt = n_scen = 0
    for s in scenarios:
        if s.truth.kind not in LEARNABLE_KINDS:
            continue
        options, criteria, weights = true_options(s)
        if len(options) < 2:
            continue
        total = sum(weights.values())
        if total <= 0:
            continue
        gid = {o.id: n_opt + i for i, o in enumerate(options)}
        for c in criteria:
            f = criterion_features(options, c)
            for o in options:
                feats.append(f.known[o.id])
                attr.append(attr_index(c.attribute))
                weight.append(weights[c.attribute] / total)
                row_option.append(gid[o.id])
        u = s.truth.utilities
        for a in options:
            for b in options:
                if u[a.id] - u[b.id] > MIN_GAP:  # type: ignore[operator]
                    pairs.append((gid[a.id], gid[b.id]))
        n_opt += len(options)
        n_scen += 1
    return PairData(
        torch.tensor(feats, dtype=torch.float32),
        torch.tensor(attr, dtype=torch.long),
        torch.tensor(weight, dtype=torch.float32),
        torch.tensor(row_option, dtype=torch.long),
        torch.tensor(pairs, dtype=torch.long).reshape(-1, 2),
        n_opt,
        n_scen,
    )
