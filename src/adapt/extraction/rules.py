"""Rule-based extractor: the transparent baseline for phase 3.

Regular expressions plus a vocabulary of attribute phrases taken ONLY from the
training domains. On unseen domains it doesn't know the words, which is exactly
the gap a learned extractor has to close.

Written against the same templates that generate the benchmark text, so its scores
on synthetic text are an optimistic upper bound for this approach.
"""

from __future__ import annotations

import re

from adapt.extraction.schema import Extraction, XConstraint, XCriterion, XOption, XValue
from adapt.extraction.textgen import FLAG_LIMIT, FLAG_TEXT, PHRASES
from adapt.synthetic.domains import DOMAINS, UNSEEN_DOMAINS

_SEEN_ATTRS = {
    name
    for d in DOMAINS
    if d.name not in UNSEEN_DOMAINS
    for name in [*(a.name for a in d.attrs), *(f.name for f in d.flags)]
}
# phrase → attribute, longest phrases first so "monthly cost" wins over "cost".
# A phrase can belong to several attributes ("monthly cost"); _lexicon_for resolves that per text.
LEXICON = sorted(
    ((p, a) for a, ps in PHRASES.items() if a in _SEEN_ATTRS for p in ps),
    key=lambda pa: -len(pa[0]),
)
_ATTR_DOMAINS = {
    name: d.name for d in DOMAINS if d.name not in UNSEEN_DOMAINS
    for name in [*(a.name for a in d.attrs), *(f.name for f in d.flags)]
}


def _lexicon_for(text: str) -> list[tuple[str, str]]:
    """One attribute per phrase: for ambiguous phrases, prefer the domain whose
    unambiguous phrases appear most in this text."""
    low = text.lower()
    owners: dict[str, list[str]] = {}
    for p, a in LEXICON:
        owners.setdefault(p, []).append(a)
    votes: dict[str, int] = {}
    for p, attrs in owners.items():
        if len(attrs) == 1 and re.search(rf"\b{re.escape(p)}\b", low):
            d = _ATTR_DOMAINS[attrs[0]]
            votes[d] = votes.get(d, 0) + 1
    pick = {p: max(attrs, key=lambda a: votes.get(_ATTR_DOMAINS[a], 0)) for p, attrs in owners.items()}
    return [(p, a) for p, a in LEXICON if pick[p] == a]
_LABEL = re.compile(r"\b(?!With\b)((?:[A-Z][a-z]+ ){1,2}[A-F])\b")  # "With Offer A" -> "Offer A"
_NUMBER = re.compile(r"^\s*(?:of\s+)?[₹$]?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b)?")
_PRIORITY = [
    (re.compile(r"matters most|top priority"), "high"),
    (re.compile(r"fairly important|fair bit"), "medium"),
    (re.compile(r"a little|nice-to-have"), "low"),
]


def _number(s: str) -> float | int | None:
    m = _NUMBER.match(s)
    if not m:
        return None
    x = float(m.group(1).replace(",", ""))
    return int(x) if x.is_integer() else x


def _attr(phrase: str, lexicon: list[tuple[str, str]]) -> str | None:
    phrase = phrase.strip().lower()
    return next((a for p, a in lexicon if p == phrase), None)


def _values(sentence: str, lexicon: list[tuple[str, str]]) -> dict[str, XValue]:
    out: dict[str, XValue] = {}
    low = sentence.lower()
    taken = [False] * len(low)
    for p, a in lexicon:
        for m in re.finditer(rf"\b{re.escape(p)}\b", low):
            if any(taken[m.start() : m.end()]) or a in out:
                continue
            x = _number(low[m.end() : m.end() + 30])
            if x is not None:
                out[a] = XValue(value=x)
                taken[m.start() : m.end()] = [True] * (m.end() - m.start())
    return out


def extract(text: str) -> Extraction:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    options: dict[str, XOption] = {}
    criteria: list[XCriterion] = []
    constraints: list[XConstraint] = []
    flag_sentences = {v: k for k, v in FLAG_LIMIT.items() if k[0] in _SEEN_ATTRS}
    lex = _lexicon_for(text)

    for s in sentences:
        if s in flag_sentences:
            attr, good = flag_sentences[s]
            constraints.append(XConstraint(attribute=attr, operator="==", value=good))
            continue

        m = re.match(r"(?:The\s+|I need an?\s+)?(.+?)\s+(?:can't be more than|must be at most|has to be at least|of at least)\s+(.+?)\.$", s)
        if m and (attr := _attr(m.group(1), lex)) and (x := _number(m.group(2))) is not None:
            op = "<=" if ("more than" in s or "at most" in s) else ">="
            constraints.append(XConstraint(attribute=attr, operator=op, value=x))
            continue

        m = re.match(r"(?:(Higher|Lower)\s+(.+?)\s+is better|I want (more|less)\s+(.+?))(?:,|\.)", s)
        if m:
            word = (m.group(1) or m.group(3)).lower()
            attr = _attr(m.group(2) or m.group(4), lex)
            if attr:
                priority = next((lv for rx, lv in _PRIORITY if rx.search(s)), None)
                criteria.append(
                    XCriterion(attribute=attr, direction="maximize" if word in ("higher", "more") else "minimize", priority=priority)
                )
            continue

        labels = _LABEL.findall(s)
        if not labels:
            continue
        label = labels[0]
        opt = options.setdefault(label, XOption(label=label))
        opt.attributes.update(_values(s, lex))
        for attr, (yes, no) in FLAG_TEXT.items():
            if attr not in _SEEN_ATTRS:
                continue
            if s.endswith(f"{label} {yes}."):
                opt.attributes[attr] = XValue(value=True)
            elif s.endswith(f"{label} {no}."):
                opt.attributes[attr] = XValue(value=False)

    return Extraction(options=list(options.values()), criteria=criteria, constraints=constraints)
