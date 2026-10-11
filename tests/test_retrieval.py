from datetime import datetime, timedelta, timezone

import pytest

from adapt.engine import Outcome, decide
from adapt.retrieval.evidence import candidates, decide_value, freshness, quality
from adapt.retrieval.lookup import apply_proposals, lookup_missing, query_for
from adapt.retrieval.providers import Hit, StaticProvider, TavilyProvider
from adapt.schema import Criterion, DecisionFrame, Direction, Option, Provenance, Value

NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)


def hit(url, content, days_old=10, title=""):
    published = None if days_old is None else NOW - timedelta(days=days_old)
    return Hit(url=url, title=title, content=content, published_at=published, retrieved_at=NOW, provider="static")


# --- quality and freshness ------------------------------------------------------


def test_source_tiers():
    assert quality("https://www.dell.com/xps-13", "Dell XPS 13")[0] == "manufacturer"
    assert quality("https://education.gov.in/x", "IIT Programme")[0] == "official"
    assert quality("https://www.reddit.com/r/laptops", "Dell XPS 13")[0] == "user_generated"
    assert quality("http://shop.example.com", "Dell XPS 13")[0] == "rejected"
    assert quality("https://www.notebookcheck.net/x", "Dell XPS 13")[0] == "general"


def test_prices_must_be_recent_and_dated_specs_can_be_old():
    assert freshness(hit("https://a.com", "", days_old=45), "price", NOW)[0] == "stale"
    assert freshness(hit("https://a.com", "", days_old=None), "price", NOW)[0] == "stale"
    assert freshness(hit("https://a.com", "", days_old=400), "battery_hours", NOW)[0] == "fresh"
    assert freshness(hit("https://a.com", "", days_old=None), "battery_hours", NOW)[0] == "undated"


# --- reading values from text ---------------------------------------------------


def test_value_must_be_in_a_sentence_naming_attribute_and_option():
    h = hit("https://www.notebookcheck.net/x",
            "The Dell XPS 13 battery lasts 12 hours in our test. Its screen is 13.4 inches. Released in 2025.")
    c = candidates(h, "Dell XPS 13", "battery_hours", now=NOW)
    assert [x.value for x in c] == [12.0]  # not 13 (the name), not 13.4 (another attribute), not 2025
    assert "battery lasts 12 hours" in c[0].quote


def test_prices_with_currency_and_commas():
    h = hit("https://www.flipkart.com/x", "Dell XPS 13 price in India is ₹1,24,990 as of today.", days_old=2)
    assert [c.value for c in candidates(h, "Dell XPS 13", "price", now=NOW)] == [124990.0]


def test_live_run_regressions_hours_minutes_and_wrong_units():
    # Found in the first live Tavily run (2026-10-11).
    h = hit("https://www.laptopmag.com/x",
            "The OLED XPS 13 (9350) battery lasted 8 hours and 40 minutes. It has a 55 Wh battery. "
            "The review took 3 hours.", title="Dell XPS 13 9350 review")
    # not 8 (minutes dropped), not 55 (capacity), not 3 (no mention of the battery)
    assert [c.value for c in candidates(h, "Dell XPS 13 9350", "battery_hours", now=NOW)] == [8.67]


def test_number_from_the_name_counts_when_written_with_the_unit():
    h = hit("https://www.dell.com/x", "Dell XPS 13 battery life up to 13 hours.")
    assert [c.value for c in candidates(h, "Dell XPS 13", "battery_hours", now=NOW)] == [13.0]


def test_any_disagreeing_usable_source_means_confirm():
    # Two review sites agree on 8.67 h, but the manufacturer's page says 13 h: ask, don't pick.
    hs = [hit("https://www.laptopmag.com/a", "Dell XPS 13 battery lasted 8 hours and 40 minutes."),
          hit("https://www.tomshardware.com/b", "Dell XPS 13 battery: 8 hours and 40 minutes."),
          hit("https://www.dell.com/c", "Dell XPS 13 battery life up to 13 hours.", days_old=None)]
    p = decide_value("a", "Dell XPS 13", "battery_hours", [c for h in hs for c in candidates(h, "Dell XPS 13", "battery_hours", now=NOW)])
    assert p.status == "confirm" and "disagree" in p.reasons[0]


# --- deciding whether a value can be used ---------------------------------------


def test_two_sites_agreeing_is_applied():
    hs = [hit("https://www.notebookcheck.net/a", "Dell XPS 13 battery life: 12 hours."),
          hit("https://www.rtings.com/b", "The Dell XPS 13 battery lasted 12.1 hours.")]
    p = decide_value("a", "Dell XPS 13", "battery_hours", [c for h in hs for c in candidates(h, "Dell XPS 13", "battery_hours", now=NOW)])
    assert p.status == "apply" and p.value == pytest.approx(12.0)


def test_disagreeing_sources_need_confirmation():
    hs = [hit("https://www.notebookcheck.net/a", "Dell XPS 13 battery life: 12 hours."),
          hit("https://www.rtings.com/b", "Dell XPS 13 battery life: 8 hours.")]
    p = decide_value("a", "Dell XPS 13", "battery_hours", [c for h in hs for c in candidates(h, "Dell XPS 13", "battery_hours", now=NOW)])
    assert p.status == "confirm" and "disagree" in p.reasons[0]


def test_single_general_site_needs_confirmation_but_manufacturer_is_enough():
    one = [hit("https://www.notebookcheck.net/a", "Dell XPS 13 battery life: 12 hours.")]
    p = decide_value("a", "Dell XPS 13", "battery_hours", [c for h in one for c in candidates(h, "Dell XPS 13", "battery_hours", now=NOW)])
    assert p.status == "confirm"
    maker = [hit("https://www.dell.com/a", "Dell XPS 13 battery life: up to 12 hours.")]
    p = decide_value("a", "Dell XPS 13", "battery_hours", [c for h in maker for c in candidates(h, "Dell XPS 13", "battery_hours", now=NOW)])
    assert p.status == "apply"


def test_forums_and_stale_prices_are_never_applied():
    hs = [hit("https://www.reddit.com/r/x", "Dell XPS 13 price is $999 here."),
          hit("https://www.quora.com/y", "Dell XPS 13 price is $999."),
          hit("https://www.bestbuy.com/z", "Dell XPS 13 price $999.", days_old=200)]
    p = decide_value("a", "Dell XPS 13", "price", [c for h in hs for c in candidates(h, "Dell XPS 13", "price", now=NOW)])
    assert p.status == "none" and p.value is None and len(p.others) == 3


# --- end to end with the engine -------------------------------------------------


def frame():
    stated = lambda x: Value(value=x, provenance=Provenance.STATED)  # noqa: E731
    return DecisionFrame(
        situation="laptop",
        options=[Option(id="a", label="Dell XPS 13", attributes={"weight_kg": stated(1.2)}),
                 Option(id="b", label="Acer Swift Go", attributes={"weight_kg": stated(1.3), "battery_hours": stated(9)})],
        criteria=[Criterion(attribute="battery_hours", direction=Direction.MAXIMIZE, weight=2),
                  Criterion(attribute="weight_kg", direction=Direction.MINIMIZE, weight=1)],
    )


def test_lookup_fills_only_missing_values_with_sources_and_engine_cites_them():
    provider = StaticProvider({query_for("Dell XPS 13", "battery_hours"): [
        hit("https://www.dell.com/xps", "Dell XPS 13 battery life: up to 14 hours."),
    ]})
    f = frame()
    proposals = lookup_missing(f, provider)
    assert provider.queries == ["Dell XPS 13 battery hours"]  # only the missing value was searched
    filled = apply_proposals(f, proposals)
    v = filled.options[0].attributes["battery_hours"]
    assert v.value == 14 and v.provenance is Provenance.RETRIEVED and "dell.com" in str(v.source.url)
    assert filled.options[1].attributes["battery_hours"].provenance is Provenance.STATED

    r = decide(filled)
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "a"
    assert [(c.option_id, c.attribute, c.value) for c in r.citations] == [("a", "battery_hours", 14)]


def test_unconfirmed_evidence_is_not_applied_so_engine_still_asks():
    provider = StaticProvider(default=[hit("https://www.notebookcheck.net/x", "Dell XPS 13 battery life: 14 hours.")])
    f = frame()
    filled = apply_proposals(f, lookup_missing(f, provider))
    assert "battery_hours" not in filled.options[0].attributes
    assert decide(filled).outcome is Outcome.CLARIFY


def test_tavily_without_key_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr("adapt.retrieval.providers.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        TavilyProvider(cache_dir=tmp_path)
