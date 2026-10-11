"""Render synthetic scenarios as natural-language text, with the gold frame to recover.

The text is template-generated, so it is cleaner and more regular than real writing.
It tests extraction mechanics (find the options, values, priorities and limits)
and unseen-domain transfer; it is not evidence of handling real users' messages.

Priorities: people don't state numeric weights, so the text uses priority words and
the gold frame stores the matching level (high / medium / low → 3 / 2 / 1).
"""

from __future__ import annotations

import re
import string

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


# --- rendering ---------------------------------------------------------------
#
# Text is built from (text, tag) segments, so the same code produces both the plain text
# and word-level labels for training a tagger. "O" means "not part of anything to extract".
#
#   OPT  option label            ATTR  attribute named next to a value    VAL  that value
#   FLAGT / FLAGF  yes/no fact (true / false), e.g. "is open source" / "is proprietary"
#   CMAX / CMIN    attribute in a "more/less is better" sentence
#   PHIGH / PMED / PLOW  priority phrase
#   LIMLE / LIMGE  attribute in an "at most / at least" limit      LIMVAL  the limit's value
#   LIMFT / LIMFF  yes/no limit that must be true / false

TAGS = ("OPT", "ATTR", "VAL", "FLAGT", "FLAGF", "CMAX", "CMIN", "PHIGH", "PMED", "PLOW",
        "LIMLE", "LIMGE", "LIMVAL", "LIMFT", "LIMFF")

FLAG_LIMIT_SPAN: dict[tuple[str, bool], str] = {
    ("requires_relocation", False): "require relocation",
    ("open_source", True): "open source",
    ("refundable", True): "refundable",
    ("online", True): "online",
}

Seg = tuple[str, str]


def _fill(template: str, **slots: Seg | list[Seg]) -> list[Seg]:
    """'a {p} of {v}' with p=("price", "ATTR") -> [("a ", O), ("price", ATTR), (" of ", O), ...]"""
    out: list[Seg] = []
    for literal, name, _, _ in string.Formatter().parse(template):
        if literal:
            out.append((literal, "O"))
        if name is not None:
            v = slots[name]
            out.extend(v if isinstance(v, list) else [v])
    return out


def _pick(rng: np.random.Generator, xs):
    return xs[int(rng.integers(len(xs)))]


def _option_sentence(rng: np.random.Generator, o: Option) -> list[Seg]:
    parts: list[list[Seg]] = []
    flags: list[Seg] = []
    for attr, v in o.attributes.items():
        if isinstance(v.value, bool):
            yes, no = FLAG_TEXT[attr]
            flags.append((yes, "FLAGT") if v.value else (no, "FLAGF"))
        else:
            p = _pick(rng, PHRASES.get(attr, (attr.replace("_", " "),)))
            tmpl = _pick(rng, ("a {p} of {v}", "{p} {v}", "{p} of {v}"))
            parts.append(_fill(tmpl, p=(p, "ATTR"), v=(fmt_value(attr, v.value), "VAL")))
    order = list(rng.permutation(len(parts)))
    parts = [parts[i] for i in order]
    body: list[Seg] = []
    for i, part in enumerate(parts):
        if i:
            body.append((" and " if i == len(parts) - 1 else ", ", "O"))
        body += part
    tmpl = _pick(rng, ("{l} has {b}.", "{l}: {b}.", "With {l} I'd get {b}."))
    out = _fill(tmpl, l=(o.label, "OPT"), b=body) if parts else []
    for f in flags:
        out += [(" ", "O"), (o.label, "OPT"), (" ", "O"), f, (".", "O")]
    return out


def _criterion_sentence(rng: np.random.Generator, c: Criterion) -> list[Seg]:
    p = _pick(rng, PHRASES.get(c.attribute, (c.attribute.replace("_", " "),)))
    more = c.direction is Direction.MAXIMIZE
    attr = (p, "CMAX" if more else "CMIN")
    out = list(_pick(rng, (
        [(f"{'Higher' if more else 'Lower'} ", "O"), attr, (" is better", "O")],
        [(f"I want {'more' if more else 'less'} ", "O"), attr],
    )))
    if c.weight is None:
        return out + [(".", "O")]
    lv = level(c.weight)
    lv_clause = _pick(rng, {
        "high": ("it matters most to me", "it's my top priority"),
        "medium": ("it's fairly important", "it matters a fair bit"),
        "low": ("it only matters a little", "it's a nice-to-have"),
    }[lv])
    tag = {"high": "PHIGH", "medium": "PMED", "low": "PLOW"}[lv]
    return out + [(", and ", "O"), (lv_clause, tag), (".", "O")]


def _constraint_sentence(rng: np.random.Generator, c: Constraint) -> list[Seg]:
    if isinstance(c.value, bool):
        sentence, span = FLAG_LIMIT[(c.attribute, c.value)], FLAG_LIMIT_SPAN[(c.attribute, c.value)]
        before, after = sentence.split(span, 1)
        return [(before, "O"), (span, "LIMFT" if c.value else "LIMFF"), (after, "O")]
    p = _pick(rng, PHRASES.get(c.attribute, (c.attribute.replace("_", " "),)))
    v = (fmt_value(c.attribute, c.value), "LIMVAL")
    if c.operator is Operator.LE:
        tmpl = _pick(rng, ("The {p} can't be more than {v}.", "The {p} must be at most {v}."))
        return _fill(tmpl, p=(p, "LIMLE"), v=v)
    tmpl = _pick(rng, ("The {p} has to be at least {v}.", "I need a {p} of at least {v}."))
    return _fill(tmpl, p=(p, "LIMGE"), v=v)


def render_tagged(s: Scenario, seed: int = 0) -> tuple[list[str], list[str]]:
    """Words and their BIO tags ("B-VAL", "I-ATTR", "O", ...). " ".join(words) == render(s, seed)."""
    rng = np.random.default_rng([seed, int(s.truth.scenario_id.rsplit("-", 1)[1])])
    noun = s.frame.situation.split(" between ")[-1].removesuffix(" options.").split(" ", 1)[-1]
    f = s.frame
    if not f.options:
        intro = _pick(rng, (f"I need to choose a {noun} but haven't shortlisted any yet.", f"Help me think about picking a {noun}."))
    else:
        intro = _pick(rng, (f"I'm choosing between {len(f.options)} {noun} options.", f"Help me pick a {noun}."))
    sentences: list[list[Seg]] = [[(intro, "O")]]
    sentences += [_option_sentence(rng, o) for o in f.options]
    prefs = [_criterion_sentence(rng, c) for c in f.criteria] + [_constraint_sentence(rng, c) for c in f.constraints]
    sentences += [prefs[i] for i in rng.permutation(len(prefs))]

    # character-level tags and span ids, then words
    chars: list[str] = []
    char_tag: list[str] = []
    char_span: list[int] = []
    span = 0
    for i, sent in enumerate(sentences):
        segs = ([(" ", "O")] if i else []) + sent
        for text, tag in segs:
            span += 1
            chars += text
            char_tag += [tag] * len(text)
            char_span += [span] * len(text)
    text = "".join(chars)
    words, tags = [], []
    prev_span = None
    for m in re.finditer(r"\S+", text):
        idx = next((j for j in range(m.start(), m.end()) if char_tag[j] != "O"), None)
        if idx is None:
            words.append(m.group())
            tags.append("O")
            prev_span = None
            continue
        bio = "I" if char_span[idx] == prev_span else "B"
        words.append(m.group())
        tags.append(f"{bio}-{char_tag[idx]}")
        prev_span = char_span[idx]
    # "a ops burden" -> "an ops burden"; only the bare word "a", never labels like "Offer A"
    for j in range(len(words) - 1):
        if words[j] == "a" and words[j + 1][:1] in "aeiou" and words[j + 1][:1]:
            words[j] = "an"
    return words, tags


def render(s: Scenario, seed: int = 0) -> str:
    return " ".join(render_tagged(s, seed)[0])

