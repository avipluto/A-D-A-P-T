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
    pair_weight: torch.Tensor | None = None  # [P]; None = every pair counts 1

    def to(self, device: torch.device) -> PairData:
        return PairData(
            self.feats.to(device), self.attr.to(device), self.weight.to(device),
            self.row_option.to(device), self.pairs.to(device), self.n_options, self.n_scenarios,
            None if self.pair_weight is None else self.pair_weight.to(device),
        )


def concat(a: PairData, b: PairData) -> PairData:
    """Both datasets in one, with b's option indices shifted after a's."""
    def w(d: PairData) -> torch.Tensor:
        return torch.ones(len(d.pairs)) if d.pair_weight is None else d.pair_weight

    return PairData(
        torch.cat([a.feats, b.feats]), torch.cat([a.attr, b.attr]), torch.cat([a.weight, b.weight]),
        torch.cat([a.row_option, b.row_option + a.n_options]), torch.cat([a.pairs, b.pairs + a.n_options]),
        a.n_options + b.n_options, a.n_scenarios + b.n_scenarios, torch.cat([w(a), w(b)]),
    )


def build_real_pairs(examples, attr_index, scale: float = 1.0) -> PairData:
    """Real decisions (lifecycle.feedback_data.RealExample): chosen option > each other
    option that met every hard limit, weighted by the example weight × scale. Decisions
    with unknown priorities are skipped: there is nothing to attribute the choice to."""
    from adapt.lifecycle.feedback_data import feasible

    feats, attr, weight, row_option, pairs, pw = [], [], [], [], [], []
    n_opt = n_scen = 0
    for ex in examples:
        f = ex.frame
        if any(c.weight is None for c in f.criteria) or not f.criteria:
            continue
        ok = set(feasible(f))
        options = [o for o in f.options if o.id in ok]
        if any(c.attribute not in o.attributes for o in options for c in f.criteria):
            continue  # incomplete values: features would be guesses
        total = sum(c.weight for c in f.criteria)  # type: ignore[misc]
        gid = {o.id: n_opt + i for i, o in enumerate(options)}
        for c in f.criteria:
            cf = criterion_features(options, c)
            for o in options:
                feats.append(cf.known[o.id] if o.id in cf.known else (0.5, cf.cv, cf.k))
                attr.append(attr_index(c.attribute))
                weight.append(c.weight / total)  # type: ignore[operator]
                row_option.append(gid[o.id])
        for o in options:
            if o.id != ex.chosen:
                pairs.append((gid[ex.chosen], gid[o.id]))
                pw.append(ex.weight * scale)
        n_opt += len(options)
        n_scen += 1
    return PairData(
        torch.tensor(feats, dtype=torch.float32).reshape(-1, 3), torch.tensor(attr, dtype=torch.long),
        torch.tensor(weight, dtype=torch.float32), torch.tensor(row_option, dtype=torch.long),
        torch.tensor(pairs, dtype=torch.long).reshape(-1, 2), n_opt, n_scen, torch.tensor(pw, dtype=torch.float32),
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
