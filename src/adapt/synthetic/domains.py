"""Domain templates for synthetic scenarios.

Each attribute has a realistic range, a direction, and a hidden value *shape* that
the engine never sees. Shapes make the true utility nonlinear (diminishing returns,
thresholds), so a linear baseline can't score perfectly and learned models have
something real to improve on.

These templates test mechanics only. They are not evidence of real-world behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adapt.schema import Direction

MAX, MIN = Direction.MAXIMIZE, Direction.MINIMIZE

SHAPES = {
    "linear": lambda n: n,
    "concave": lambda n: math.sqrt(n),  # diminishing returns
    "convex": lambda n: n * n,  # only the extremes really matter
    "threshold": lambda n: 1 / (1 + math.exp(-10 * (n - 0.5))),  # good enough vs not
}


@dataclass(frozen=True)
class Attr:
    name: str
    lo: float
    hi: float
    direction: Direction
    shape: str = "linear"
    unit: str | None = None
    integer: bool = False

    def desirability(self, x: float) -> float:
        """Hidden true value of x in [0, 1], using the full domain range."""
        n = min(max((x - self.lo) / (self.hi - self.lo), 0.0), 1.0)
        if self.direction is MIN:
            n = 1.0 - n
        return SHAPES[self.shape](n)


@dataclass(frozen=True)
class Flag:
    """A yes/no attribute used only as a hard limit (e.g. "must not require relocation")."""

    name: str
    good: bool


@dataclass(frozen=True)
class Domain:
    name: str
    noun: str
    attrs: tuple[Attr, ...]
    flags: tuple[Flag, ...] = ()


DOMAINS: tuple[Domain, ...] = (
    Domain("career", "job offer", (
        Attr("salary", 4, 40, MAX, "concave", "LPA"),
        Attr("commute_minutes", 5, 120, MIN, "convex", "min", True),
        Attr("growth_score", 1, 10, MAX, "linear", None, True),
        Attr("hours_per_week", 35, 70, MIN, "threshold", "h", True),
    ), (Flag("requires_relocation", False),)),
    Domain("personal_finance", "investment", (
        Attr("expected_return_pct", 2, 14, MAX, "concave", "%"),
        Attr("risk_score", 1, 10, MIN, "convex", None, True),
        Attr("lock_in_days", 0, 1825, MIN, "concave", "days", True),
        Attr("fees_pct", 0, 2.5, MIN, "linear", "%"),
    )),
    Domain("tech_architecture", "architecture", (
        Attr("monthly_cost_usd", 0, 5000, MIN, "concave", "USD", True),
        Attr("p95_latency_ms", 5, 800, MIN, "threshold", "ms", True),
        Attr("team_familiarity", 1, 10, MAX, "linear", None, True),
        Attr("ops_burden", 1, 10, MIN, "linear", None, True),
    ), (Flag("open_source", True),)),
    Domain("project_planning", "project plan", (
        Attr("cost_k", 1, 500, MIN, "linear", "kUSD"),
        Attr("duration_weeks", 1, 52, MIN, "concave", "weeks", True),
        Attr("success_prob_pct", 30, 99, MAX, "convex", "%", True),
        Attr("staff_needed", 1, 30, MIN, "linear", None, True),
    )),
    Domain("productivity", "tool", (
        Attr("hours_saved_week", 0, 15, MAX, "concave", "h"),
        Attr("setup_hours", 0, 40, MIN, "linear", "h", True),
        Attr("monthly_cost", 0, 50, MIN, "threshold", "USD"),
        Attr("learning_curve", 1, 10, MIN, "linear", None, True),
    )),
    Domain("relocation", "city", (
        Attr("rent", 5000, 80000, MIN, "concave", "INR", True),
        Attr("commute_minutes", 5, 120, MIN, "convex", "min", True),
        Attr("safety_score", 1, 10, MAX, "concave", None, True),
        Attr("distance_to_family_km", 0, 3000, MIN, "concave", "km", True),
    )),
    Domain("purchase_electronics", "laptop", (
        Attr("price", 10000, 250000, MIN, "concave", "INR", True),
        Attr("performance_score", 1, 100, MAX, "concave", None, True),
        Attr("battery_hours", 2, 24, MAX, "concave", "h"),
        Attr("weight_kg", 0.9, 3.5, MIN, "linear", "kg"),
        Attr("warranty_years", 0, 3, MAX, "linear", "years", True),
    )),
    # Held out entirely: never in train/val, used to test unseen-domain generalisation.
    Domain("travel", "itinerary", (
        Attr("price", 2000, 150000, MIN, "concave", "INR", True),
        Attr("travel_hours", 1, 30, MIN, "convex", "h"),
        Attr("comfort_score", 1, 10, MAX, "linear", None, True),
        Attr("layovers", 0, 3, MIN, "threshold", None, True),
    ), (Flag("refundable", True),)),
    Domain("education", "programme", (
        Attr("tuition", 0, 4000000, MIN, "concave", "INR", True),
        Attr("duration_months", 3, 48, MIN, "linear", "months", True),
        Attr("placement_rate_pct", 20, 99, MAX, "convex", "%", True),
        Attr("ranking_score", 1, 100, MAX, "concave", None, True),
    ), (Flag("online", True),)),
)

UNSEEN_DOMAINS = frozenset({"travel", "education"})
