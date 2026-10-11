"""Fill a frame's missing values from search evidence, without ever deciding.

`lookup_missing` searches for every value a criterion or hard limit needs but an
option lacks, and returns proposals. `apply_proposals` writes only the "apply" ones
into a copy of the frame, as RETRIEVED values carrying their source. The user's own
values are never overwritten, and "confirm" proposals are left for the user.
"""

from __future__ import annotations

from adapt.retrieval.evidence import Proposal, candidates, decide_value
from adapt.retrieval.providers import SearchProvider
from adapt.schema import DecisionFrame, Provenance, Source, Value


def query_for(option_label: str, attribute: str) -> str:
    return f"{option_label} {attribute.replace('_', ' ')}"


def lookup_missing(frame: DecisionFrame, provider: SearchProvider, max_results: int = 5,
                   max_searches: int = 10) -> list[Proposal]:
    needed = list(dict.fromkeys([c.attribute for c in frame.criteria] + [c.attribute for c in frame.constraints]))
    slots = [(o, a) for o in frame.options for a in needed if a not in o.attributes]
    proposals = []
    for o, a in slots[:max_searches]:  # each search costs a credit; cap it
        cands = [c for h in provider.search(query_for(o.label, a), max_results) for c in candidates(h, o.label, a)]
        proposals.append(decide_value(o.id, o.label, a, cands))
    for o, a in slots[max_searches:]:
        proposals.append(Proposal(o.id, o.label, a, "none", reasons=[f"search budget of {max_searches} reached"]))
    return proposals


def apply_proposals(frame: DecisionFrame, proposals: list[Proposal]) -> DecisionFrame:
    by_id = {o.id: o for o in frame.options}
    updated = dict(by_id)
    for p in proposals:
        if p.status != "apply" or p.value is None:
            continue
        o = updated[p.option_id]
        if p.attribute in o.attributes:  # never overwrite what the user said
            continue
        best = max(p.support, key=lambda c: (c.tier in ("official", "manufacturer"), c.hit.published_at is not None))
        v = Value(
            value=int(p.value) if float(p.value).is_integer() else p.value,
            provenance=Provenance.RETRIEVED,
            source=Source(url=best.hit.url, title=best.hit.title or None, published_at=best.hit.published_at,
                          retrieved_at=best.hit.retrieved_at),
        )
        updated[p.option_id] = o.model_copy(update={"attributes": {**o.attributes, p.attribute: v}})
    return frame.model_copy(update={"options": [updated[o.id] for o in frame.options]})
