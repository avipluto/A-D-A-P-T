"""Render synthetic scenarios as natural-language text, with the gold frame to recover.

The text is template-generated, so it is cleaner and more regular than real writing.
It tests extraction mechanics (find the options, values, priorities and limits)
and unseen-domain transfer; it is not evidence of handling real users' messages.

Priorities: people don't state numeric weights, so the text uses priority words and
the gold frame stores the matching level (high / medium / low → 3 / 2 / 1).
"""

from __future__ import annotations

import re

import numpy as np

from adapt.schema import Constraint, Criterion, DecisionFrame, Direction, Operator, Option
from adapt.synthetic.generator import Scenario

LEVEL_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}

# How each attribute is named in text (first entry is the default).
PHRASES: dict[str, tuple[str, ...]] = {
    "salary": ("salary", "pay", "annual package"),
    "commute_minutes": ("commute", "commute time"),
    "growth_score": ("growth score", "growth prospects"),
    "hours_per_week": ("weekly hours", "working hours"),
    "expected_return_pct": ("expected return", "return"),
    "risk_score": ("risk score", "risk"),
    "lock_in_days": ("lock-in period", "lock-in"),
    "fees_pct": ("fees", "fee"),
    "monthly_cost_usd": ("monthly cost", "cost per month"),
    "p95_latency_ms": ("p95 latency", "latency"),
    "team_familiarity": ("team familiarity", "familiarity"),
    "ops_burden": ("ops burden", "operational burden"),
    "cost_k": ("cost", "budget"),
    "duration_weeks": ("duration", "timeline"),
    "success_prob_pct": ("success probability", "chance of success"),
    "staff_needed": ("staff needed", "headcount"),
    "hours_saved_week": ("hours saved per week", "time saved"),
    "setup_hours": ("setup time", "setup effort"),
    "monthly_cost": ("monthly cost", "subscription cost"),
    "learning_curve": ("learning curve",),
    "rent": ("rent", "monthly rent"),
    "safety_score": ("safety score", "safety"),
    "distance_to_family_km": ("distance to family", "distance from family"),
    "price": ("price", "cost"),
    "performance_score": ("performance score", "benchmark score"),
    "battery_hours": ("battery life", "battery"),
    "weight_kg": ("weight",),
    "warranty_years": ("warranty",),
    "travel_hours": ("travel time", "journey time"),
    "comfort_score": ("comfort score", "comfort"),
    "layovers": ("layovers", "stops"),
    "tuition": ("tuition", "tuition fees"),
    "duration_months": ("duration", "programme length"),
    "placement_rate_pct": ("placement rate", "placements"),
    "ranking_score": ("ranking score", "ranking"),
    # yes/no attributes used as hard limits
    "requires_relocation": ("relocation",),
    "open_source": ("open source",),
    "refundable": ("refundable",),
    "online": ("online",),
}

# How a value is written, per attribute.
FORMATS: dict[str, str] = {
    "salary": "{v} LPA", "commute_minutes": "{v} minutes", "growth_score": "{v}/10",
    "hours_per_week": "{v} hours a week", "expected_return_pct": "{v}%", "risk_score": "{v}/10",
    "lock_in_days": "{v} days", "fees_pct": "{v}%", "monthly_cost_usd": "${v}", "p95_latency_ms": "{v} ms",
    "team_familiarity": "{v}/10", "ops_burden": "{v}/10", "cost_k": "${v}k", "duration_weeks": "{v} weeks",
    "success_prob_pct": "{v}%", "staff_needed": "{v} people", "hours_saved_week": "{v} hours",
    "setup_hours": "{v} hours", "monthly_cost": "${v}", "learning_curve": "{v}/10", "rent": "₹{v}",
    "safety_score": "{v}/10", "distance_to_family_km": "{v} km", "price": "₹{v}",
    "performance_score": "{v}/100", "battery_hours": "{v} hours", "weight_kg": "{v} kg",
    "warranty_years": "{v} years", "travel_hours": "{v} hours", "comfort_score": "{v}/10",
    "layovers": "{v}", "tuition": "₹{v}", "duration_months": "{v} months",
    "placement_rate_pct": "{v}%", "ranking_score": "{v}/100",
}

# Yes/no attributes: (sentence when True, sentence when False)
FLAG_TEXT: dict[str, tuple[str, str]] = {
    "requires_relocation": ("requires relocation", "doesn't require relocation"),
    "open_source": ("is open source", "is proprietary"),
    "refundable": ("is refundable", "is non-refundable"),
    "online": ("is online", "is in-person"),
}
FLAG_LIMIT: dict[tuple[str, bool], str] = {
    ("requires_relocation", False): "I can't relocate, so it must not require relocation.",
    ("open_source", True): "It has to be open source.",
    ("refundable", True): "It must be refundable.",
    ("online", True): "It has to be online.",
}


def level(weight: float) -> str:
    return "high" if weight >= 0.4 else "medium" if weight >= 0.2 else "low"


def fmt_number(x: float | int) -> str:
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if isinstance(x, int):
        return f"{x:,}" if abs(x) >= 10000 else str(x)
    return f"{x:.2f}".rstrip("0").rstrip(".")


def fmt_value(attr: str, x) -> str:
    return FORMATS.get(attr, "{v}").format(v=fmt_number(x))


def gold_frame(s: Scenario) -> DecisionFrame:
    """What a perfect extractor should produce from the text: the scenario frame
    with numeric weights replaced by the priority level the text expresses."""
    criteria = [
        c.model_copy(update={"weight": None if c.weight is None else LEVEL_WEIGHT[level(c.weight)]})
        for c in s.frame.criteria
    ]
    return s.frame.model_copy(update={"criteria": criteria})


def _pick(rng: np.random.Generator, xs):
    return xs[int(rng.integers(len(xs)))]


def _option_sentence(rng: np.random.Generator, o: Option) -> str:
    parts, flags = [], []
    for attr, v in o.attributes.items():
        if isinstance(v.value, bool):
            yes, no = FLAG_TEXT[attr]
            flags.append(yes if v.value else no)
        else:
            p = _pick(rng, PHRASES.get(attr, (attr.replace("_", " "),)))
            parts.append(_pick(rng, ("a {p} of {v}", "{p} {v}", "{p} of {v}")).format(p=p, v=fmt_value(attr, v.value)))
    order = list(rng.permutation(len(parts)))
    parts = [parts[i] for i in order]
    body = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + (parts[-1] if parts else "")
    tmpl = _pick(rng, ("{l} has {b}.", "{l}: {b}.", "With {l} I'd get {b}."))
    out = tmpl.format(l=o.label, b=body) if parts else ""
    for f in flags:
        out += f" {o.label} {f}."
    return out.strip()


def _criterion_sentence(rng: np.random.Generator, c: Criterion) -> str:
    p = _pick(rng, PHRASES.get(c.attribute, (c.attribute.replace("_", " "),)))
    more = c.direction is Direction.MAXIMIZE
    dir_clause = _pick(
        rng,
        (f"{'Higher' if more else 'Lower'} {p} is better", f"I want {'more' if more else 'less'} {p}"),
    )
    if c.weight is None:
        return dir_clause + "."
    lv = level(c.weight)
    lv_clause = _pick(rng, {
        "high": ("it matters most to me", "it's my top priority"),
        "medium": ("it's fairly important", "it matters a fair bit"),
        "low": ("it only matters a little", "it's a nice-to-have"),
    }[lv])
    return f"{dir_clause}, and {lv_clause}."


def _constraint_sentence(rng: np.random.Generator, c: Constraint) -> str:
    if isinstance(c.value, bool):
        return FLAG_LIMIT[(c.attribute, c.value)]
    p = _pick(rng, PHRASES.get(c.attribute, (c.attribute.replace("_", " "),)))
    v = fmt_value(c.attribute, c.value)
    if c.operator is Operator.LE:
        return _pick(rng, (f"The {p} can't be more than {v}.", f"The {p} must be at most {v}."))
    return _pick(rng, (f"The {p} has to be at least {v}.", f"I need a {p} of at least {v}."))


def render(s: Scenario, seed: int = 0) -> str:
    rng = np.random.default_rng([seed, int(s.truth.scenario_id.rsplit("-", 1)[1])])
    noun = s.frame.situation.split(" between ")[-1].removesuffix(" options.").split(" ", 1)[-1]
    f = s.frame
    if not f.options:
        intro = _pick(rng, (f"I need to choose a {noun} but haven't shortlisted any yet.", f"Help me think about picking a {noun}."))
    else:
        intro = _pick(rng, (f"I'm choosing between {len(f.options)} {noun} options.", f"Help me pick a {noun}."))
    options = [_option_sentence(rng, o) for o in f.options]
    prefs = [_criterion_sentence(rng, c) for c in f.criteria] + [_constraint_sentence(rng, c) for c in f.constraints]
    prefs = [prefs[i] for i in rng.permutation(len(prefs))]
    text = " ".join([intro, *options, *prefs]).replace("  ", " ")
    return re.sub(r"\ba (?=[aeiou])", "an ", text)  # "a ops burden" -> "an ops burden"; never touches labels like "Offer A"
