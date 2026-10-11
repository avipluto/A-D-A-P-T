"""Look up missing facts for real products with Tavily and show the evidence.

    python scripts/lookup.py --option "Dell XPS 13 9350" --option "MacBook Air M3 13" --attribute battery_hours

Shows, for each option and attribute: the proposed value (if any), whether it would be
applied automatically or needs your confirmation and why, and every quote with its
source, date, quality tier and freshness. Each search costs 1 Tavily credit; results
are cached for 24 h under ADAPT_DATA_DIR/cache/search.
"""

from __future__ import annotations

import argparse

from adapt.retrieval.evidence import domain
from adapt.retrieval.lookup import lookup_missing
from adapt.retrieval.providers import TavilyProvider
from adapt.schema import Criterion, DecisionFrame, Direction, Option


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--option", action="append", required=True, help="product or option name (repeatable)")
    p.add_argument("--attribute", action="append", required=True, help="snake_case attribute, e.g. battery_hours")
    p.add_argument("--max-searches", type=int, default=6, help="credit cap for this run")
    args = p.parse_args()

    frame = DecisionFrame(
        situation="lookup",
        options=[Option(id=chr(97 + i), label=label) for i, label in enumerate(args.option)],
        criteria=[Criterion(attribute=a, direction=Direction.MAXIMIZE) for a in args.attribute],
    )
    provider = TavilyProvider()
    for pr in lookup_missing(frame, provider, max_searches=args.max_searches):
        value = "-" if pr.value is None else f"{pr.value:g}"
        print(f"\n{pr.option_label} / {pr.attribute}: {value}  [{pr.status.upper()}]  {'; '.join(pr.reasons)}")
        for c in pr.support:
            date = c.hit.published_at.date().isoformat() if c.hit.published_at else "undated"
            print(f"  + {c.value:g}  {domain(c.hit.url)}  ({c.tier}, {date}, {c.fresh_note})")
            print(f"      \"{c.quote[:160]}\"")
        for c in pr.others[:4]:
            date = c.hit.published_at.date().isoformat() if c.hit.published_at else "undated"
            why = "; ".join(c.quality_flags) or c.fresh_note
            print(f"  - {c.value:g}  {domain(c.hit.url)}  ({c.tier}, {date}: {why})")
    print(f"\nTavily credits used this run: {provider.credits_used} (cached results cost 0)")


if __name__ == "__main__":
    main()
