"""Labels → Extraction, by fixed rules (no learning here).

Within each sentence:
- an OPT span sets the current option;
- each ATTR span is paired with the next VAL span (an attribute and its value);
- FLAGT / FLAGF spans are yes/no facts about the current option;
- CMAX / CMIN spans are criteria, with the priority from a PHIGH/PMED/PLOW span in the
  same sentence;
- LIMLE / LIMGE spans pair with the next LIMVAL; LIMFT / LIMFF are yes/no limits.

Attribute names are the span's words in snake_case. Yes/no names drop negation and
helper words ("doesn't require relocation" and "requires relocation" both become
"require_relocation") so the fact and the limit line up. Antonyms ("proprietary" vs
"open source") can't be resolved this way: a known limitation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from adapt.extraction.schema import Extraction, XConstraint, XCriterion, XOption, XValue

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DROP = {"doesn't", "does", "not", "is", "isn't", "no", "it", "be", "must", "has", "have", "to", "a", "an", "the"}
_PRIORITY = {"PHIGH": "high", "PMED": "medium", "PLOW": "low"}


@dataclass
class Span:
    kind: str
    words: list[str]
    sentence: int

    @property
    def text(self) -> str:
        return " ".join(self.words).rstrip(".,:;!?")


def spans(words: list[str], labels: list[str]) -> list[Span]:
    out: list[Span] = []
    sentence = 0
    cur: Span | None = None
    for w, lab in zip(words, labels):
        if lab == "O":
            cur = None
        else:
            bio, kind = lab.split("-", 1)
            if cur is None or bio == "B" or cur.kind != kind:
                cur = Span(kind, [], sentence)
                out.append(cur)
            cur.words.append(w)
        if w.endswith((".", "!", "?")):  # decimals like "15.17" never end in "."
            sentence += 1
            cur = None
    return out


def _name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _flag_name(text: str) -> str:
    ws = [w.lower().rstrip(".,") for w in text.replace("non-", "").split()]
    ws = [w for w in ws if w not in _DROP]
    if ws and len(ws[0]) > 3 and ws[0].endswith("s") and not ws[0].endswith("ss"):
        ws[0] = ws[0][:-1]  # "requires" -> "require"
    return _name(" ".join(ws))


def _unit(text: str) -> str | None:
    """What's written around the number: "16.88 LPA" -> "LPA", "7/10" -> "/10", "$120k" -> "$k"."""
    rest = _NUM.sub("", text, count=1).strip(" .,;:")
    return rest or None


def _number(text: str) -> float | int | None:
    m = _NUM.search(text)
    if not m:
        return None
    x = float(m.group().replace(",", ""))
    return int(x) if x.is_integer() else x


def assemble(words: list[str], labels: list[str]) -> Extraction:
    options: dict[str, XOption] = {}
    criteria: list[XCriterion] = []
    constraints: list[XConstraint] = []
    by_sentence: dict[int, list[Span]] = {}
    for sp in spans(words, labels):
        by_sentence.setdefault(sp.sentence, []).append(sp)

    for sps in by_sentence.values():
        current: XOption | None = None
        pending_attr: str | None = None
        pending_limit: tuple[str, str] | None = None
        priority = next((_PRIORITY[s.kind] for s in sps if s.kind in _PRIORITY), None)
        for sp in sps:
            if sp.kind == "OPT":
                current = options.setdefault(sp.text, XOption(label=sp.text))
            elif sp.kind == "ATTR":
                pending_attr = _name(sp.text)
            elif sp.kind == "VAL" and current is not None and pending_attr:
                x = _number(sp.text)
                if x is not None:
                    current.attributes.setdefault(pending_attr, XValue(value=x, unit=_unit(sp.text)))
                pending_attr = None
            elif sp.kind in ("FLAGT", "FLAGF") and current is not None:
                current.attributes[_flag_name(sp.text)] = XValue(value=sp.kind == "FLAGT")
            elif sp.kind in ("CMAX", "CMIN"):
                criteria.append(XCriterion(attribute=_name(sp.text), direction="maximize" if sp.kind == "CMAX" else "minimize",
                                           priority=priority))
            elif sp.kind in ("LIMLE", "LIMGE"):
                pending_limit = (_name(sp.text), "<=" if sp.kind == "LIMLE" else ">=")
            elif sp.kind == "LIMVAL" and pending_limit:
                x = _number(sp.text)
                if x is not None:
                    constraints.append(XConstraint(attribute=pending_limit[0], operator=pending_limit[1], value=x))
                pending_limit = None
            elif sp.kind in ("LIMFT", "LIMFF"):
                constraints.append(XConstraint(attribute=_flag_name(sp.text), operator="==", value=sp.kind == "LIMFT"))
    return Extraction(options=list(options.values()), criteria=criteria, constraints=constraints)
