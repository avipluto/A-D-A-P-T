"""How much would priorities have to change for a different option to win?

Both scorers are additive: score_o = Σ_c ŵ_c · u_oc, with u_oc independent of the
weights. Move one criterion's share of importance to t and scale the others
proportionally so the shares still sum to 1:

    score_o(t) = t · u_oc + (1 − t) · R_o,   R_o = Σ_{c'≠c} ŵ_c' / (1 − ŵ_c) · u_oc'

The leader L's lead over rival j is linear in t, so the exact t where j catches up is
t* = B / (B − A) with A = u_Lc − u_jc and B = R_L − R_j. No grid search needed.

Only meaningful with complete information (point scores).
"""

from __future__ import annotations

from pydantic import BaseModel

from adapt.engine.scoring import Scored

FRAGILE = 0.10  # a flip within this much of the current weight share counts as fragile
_EPS = 1e-9


class Flip(BaseModel):
    attribute: str
    current_weight: float  # normalised share of importance, 0..1
    flip_weight: float  # share at which the winner changes
    new_leader: str

    @property
    def distance(self) -> float:
        return abs(self.flip_weight - self.current_weight)


def weight_flips(scored: list[Scored], weights: dict[str, float], leader: str) -> list[Flip]:
    """Nearest flip in each direction for each criterion, closest first."""
    attrs = list(weights)
    if len(attrs) < 2 or len(scored) < 2:
        return []
    total = sum(weights.values())
    w = {a: weights[a] / total for a in attrs}
    u = {s.option_id: {a: s.unit[a][0] for a in attrs} for s in scored}

    def rest(o: str, a: str) -> float:
        others = [b for b in attrs if b != a]
        if w[a] < 1 - _EPS:
            return sum(w[b] / (1 - w[a]) * u[o][b] for b in others)
        return sum(u[o][b] for b in others) / len(others)  # all weight on a: split the rest evenly

    flips: list[Flip] = []
    for a in attrs:
        up: tuple[float, str] | None = None
        down: tuple[float, str] | None = None
        for j in u:
            if j == leader:
                continue
            A = u[leader][a] - u[j][a]
            B = rest(leader, a) - rest(j, a)
            if abs(B - A) < _EPS:
                continue  # the lead doesn't depend on this weight
            t = B / (B - A)
            if not 0.0 <= t <= 1.0:
                continue
            if t > w[a] + _EPS and (up is None or t < up[0]):
                up = (t, j)
            elif t < w[a] - _EPS and (down is None or t > down[0]):
                down = (t, j)
        for found in (up, down):
            if found:
                flips.append(Flip(attribute=a, current_weight=round(w[a], 4), flip_weight=round(found[0], 4), new_leader=found[1]))
    return sorted(flips, key=lambda f: f.distance)
