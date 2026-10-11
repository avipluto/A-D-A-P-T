"""Turn search hits into evidence for missing values, with checks before anything is used.

For a missing (option, attribute) slot:

1. Candidates: a number counts only if the page text contains it in a sentence that
   also names the attribute and (part of) the option. The sentence is kept as a quote.
   If the attribute's name implies a unit ("battery_hours"), the number must be written
   with that unit, and "8 hours and 40 minutes" reads as 8.67.
2. Each source gets a quality tier and a freshness status (see below).
3. Candidates from usable sources are grouped by value (within 2%). A value is
   auto-applicable only if it is fresh, from a usable source, either confirmed by two
   different websites or from an official / manufacturer site, AND no usable source
   states a different value. Everything else is "needs confirmation" with reasons.

Nothing here decides; it only proposes values marked RETRIEVED with their source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from adapt.retrieval.providers import Hit

# --- source quality -------------------------------------------------------------

USER_GENERATED = ("reddit.com", "quora.com", "medium.com", "youtube.com", "facebook.com", "x.com",
                  "twitter.com", "tiktok.com", "forum", "community.", "answers.")
REFERENCE = ("wikipedia.org",)
OFFICIAL_SUFFIXES = (".gov", ".gov.in", ".nic.in", ".edu", ".ac.in", ".ac.uk", ".gov.uk")


def domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def quality(url: str, option_label: str) -> tuple[str, list[str]]:
    """(tier, flags). Tiers: official, manufacturer, reference, general, user_generated, rejected."""
    if urlparse(url).scheme != "https":
        return "rejected", ["not https"]
    d = domain(url)
    if any(d.endswith(s) for s in OFFICIAL_SUFFIXES):
        return "official", []
    if any(u in d for u in USER_GENERATED):
        return "user_generated", ["user-generated content; shown, never applied automatically"]
    brand = re.sub(r"[^a-z0-9]", "", option_label.split()[0].lower()) if option_label.split() else ""
    if len(brand) >= 3 and brand in d.replace(".", "").replace("-", ""):
        return "manufacturer", []
    if any(d.endswith(r) for r in REFERENCE):
        return "reference", []
    return "general", []


# --- freshness ------------------------------------------------------------------

# Attributes that change quickly must have a recent, dated source. Matched by word in the name.
VOLATILE_DAYS = {"price": 30, "cost": 30, "fee": 60, "fees": 60, "rent": 90, "salary": 180, "tuition": 365,
                 "deadline": 7, "stock": 1, "rate": 30}
DEFAULT_MAX_AGE_DAYS = 3 * 365


def max_age_days(attribute: str) -> tuple[int, bool]:
    """(max age in days, must the source be dated?)"""
    words = attribute.lower().split("_")
    hits = [VOLATILE_DAYS[w] for w in words if w in VOLATILE_DAYS]
    return (min(hits), True) if hits else (DEFAULT_MAX_AGE_DAYS, False)


def freshness(hit: Hit, attribute: str, now: datetime | None = None) -> tuple[str, str]:
    """(status, explanation). Status: fresh, stale, undated."""
    now = now or datetime.now(timezone.utc)
    limit, needs_date = max_age_days(attribute)
    if hit.published_at is None:
        return ("stale" if needs_date else "undated"), (
            f"no publication date; {attribute} changes often" if needs_date else "no publication date"
        )
    age = (now - hit.published_at).days
    if age > limit:
        return "stale", f"{age} days old (limit {limit} for {attribute})"
    return "fresh", f"{age} days old"


# --- candidates from text ---------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])([₹$€£]\s?)?(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)\s*(k\b)?")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


# If the attribute's name carries a unit, the number on the page must be written with it.
# "battery_hours" must read "8 hours", so "52.6 Wh" (capacity) is not taken as battery life.
UNIT_WORDS = {
    "hours": r"(?:hours?|hrs?|h)\b", "hour": r"(?:hours?|hrs?|h)\b", "minutes": r"(?:minutes?|mins?)\b",
    "kg": r"(?:kg|kilograms?)\b", "km": r"(?:km|kilomet(?:er|re)s?)\b", "days": r"days?\b",
    "weeks": r"weeks?\b", "months": r"months?\b", "years": r"(?:years?|yrs?)\b", "ms": r"ms\b",
    "pct": r"(?:%|percent)", "percent": r"(?:%|percent)", "gb": r"gb\b", "inches": r"(?:inch(?:es)?|in\b|\")",
}
_MINUTES_AFTER = re.compile(r"^\s*(?:hours?|hrs?|h)\s*(?:and\s*)?(\d{1,2})\s*(?:minutes?|mins?|m)\b", re.IGNORECASE)


def unit_pattern(attribute: str) -> re.Pattern | None:
    for w in reversed(attribute.lower().split("_")):
        if w in UNIT_WORDS:
            return re.compile(r"^\s*" + UNIT_WORDS[w], re.IGNORECASE)
    return None


def attribute_words(attribute: str, aliases: tuple[str, ...] = ()) -> list[str]:
    words = [w for w in attribute.lower().split("_") if len(w) > 2 and w not in {"the", "per", "and"}]
    return words + [a.lower() for a in aliases]


def option_words(label: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", label.lower()) if len(w) > 1]


@dataclass
class Candidate:
    value: float
    quote: str
    hit: Hit
    tier: str
    quality_flags: list[str]
    fresh: str
    fresh_note: str

    @property
    def usable(self) -> bool:
        return self.tier not in ("rejected", "user_generated") and self.fresh != "stale"


def candidates(hit: Hit, option_label: str, attribute: str, aliases: tuple[str, ...] = (),
               now: datetime | None = None) -> list[Candidate]:
    tier, flags = quality(hit.url, option_label)
    fresh, note = freshness(hit, attribute, now)
    a_words = [w for w in attribute_words(attribute, aliases) if w not in UNIT_WORDS]
    unit = unit_pattern(attribute)
    o_words = option_words(option_label)
    title = hit.title.lower()
    out = []
    for sentence in _SENT.split(hit.content):
        s = sentence.lower()
        if not any(w in s for w in a_words):
            continue
        # the option must be named in the sentence, or the page must be about it (title)
        if not (sum(w in s for w in o_words) >= max(1, len(o_words) // 2) or all(w in title for w in o_words)):
            continue
        label_numbers = set(re.findall(r"\d+(?:\.\d+)?", option_label))
        anchors = [m.start() for w in a_words for m in re.finditer(re.escape(w), s)]
        best: tuple[int, float] | None = None
        for m in _NUM.finditer(sentence):
            raw = m.group(2).replace(",", "")
            x = float(raw)
            after = sentence[m.end():]
            has_unit = unit is not None and bool(unit.match(after))
            if raw in label_numbers and not has_unit:  # "XPS 13" is a name; "13 hours" is a value
                continue
            if not m.group(1) and x.is_integer() and 1900 <= x <= 2100:  # a bare year
                continue
            if unit is not None and not has_unit:  # wrong kind of number (e.g. Wh, not hours)
                continue
            if m.group(3):
                x *= 1000
            if unit is not None and (mm := _MINUTES_AFTER.match(after)):  # "8 hours and 40 minutes"
                x = round(x + int(mm.group(1)) / 60, 2)
            dist = min(abs(m.start() - a) for a in anchors)
            if best is None or dist < best[0]:
                best = (dist, x)
        if best is not None:  # one value per sentence: the number nearest the attribute word
            out.append(Candidate(best[1], sentence.strip()[:300], hit, tier, flags, fresh, note))
    return out


# --- consensus --------------------------------------------------------------------

@dataclass
class Proposal:
    option_id: str
    option_label: str
    attribute: str
    status: str  # "apply" | "confirm" | "none"
    value: float | None = None
    support: list[Candidate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    others: list[Candidate] = field(default_factory=list)  # considered but not supporting


def _groups(cands: list[Candidate], tol: float = 0.02) -> list[list[Candidate]]:
    groups: list[list[Candidate]] = []
    for c in sorted(cands, key=lambda c: c.value):
        if groups and abs(c.value - groups[-1][0].value) <= tol * max(1.0, abs(groups[-1][0].value)):
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def decide_value(option_id: str, option_label: str, attribute: str, cands: list[Candidate]) -> Proposal:
    p = Proposal(option_id, option_label, attribute, "none")
    usable = [c for c in cands if c.usable]
    p.others = [c for c in cands if not c.usable]
    if not usable:
        p.reasons.append("no usable source states a value" + (" (only stale or user-generated ones)" if cands else ""))
        return p

    def support(g: list[Candidate]) -> tuple[int, bool]:
        domains = {domain(c.hit.url) for c in g}
        return len(domains), any(c.tier in ("official", "manufacturer") for c in g)

    groups = sorted(_groups(usable), key=lambda g: support(g), reverse=True)
    best = groups[0]
    n_domains, authoritative = support(best)
    p.value = best[0].value
    p.support = best
    p.others += [c for g in groups[1:] for c in g]

    if len(groups) > 1:
        # Any usable source stating a different value means ask, not pick (a live test showed
        # a "majority" value being applied while the manufacturer's site said something else).
        p.status = "confirm"
        p.reasons.append(f"sources disagree: {sorted({g[0].value for g in groups})}")
    elif n_domains >= 2 or authoritative:
        p.status = "apply"
        p.reasons.append(f"{n_domains} independent site(s) agree" + ("; includes an official/manufacturer site" if authoritative else ""))
    else:
        p.status = "confirm"
        p.reasons.append("only one non-official website states this value")
    if any(c.fresh == "undated" for c in best):
        p.reasons.append("some supporting sources are undated")
    return p
