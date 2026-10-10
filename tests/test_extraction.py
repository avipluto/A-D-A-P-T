import pytest

from adapt.engine import decide
from adapt.extraction.llm import parse_json
from adapt.extraction.metrics import attr_match, evaluate_extractor, score_fields, value_match
from adapt.extraction.rules import extract
from adapt.extraction.schema import Extraction, XCriterion, XOption, XValue, to_frame
from adapt.extraction.textgen import LEVEL_WEIGHT, gold_frame, level, render
from adapt.schema import Provenance
from adapt.synthetic import generate


@pytest.fixture(scope="module")
def scenarios():
    return list(generate(40, seed=2))


def first(scenarios, domain, kind="complete"):
    return next(s for s in scenarios if s.truth.domain == domain and s.truth.kind == kind)


def test_render_is_deterministic_and_mentions_every_option(scenarios):
    s = first(scenarios, "career")
    assert render(s) == render(s)
    assert all(o.label in render(s) for o in s.frame.options)


def test_article_fix_never_touches_option_labels(scenarios):
    for s in scenarios:
        text = render(s)
        assert "An I'd" not in text and " An " not in text.replace(". An ", ". ")


def test_gold_frame_uses_priority_levels(scenarios):
    s = first(scenarios, "career")
    g = gold_frame(s)
    for c, gc in zip(s.frame.criteria, g.criteria):
        assert gc.weight == LEVEL_WEIGHT[level(c.weight)]


def test_to_frame_marks_everything_inferred_and_normalises_names():
    x = Extraction(
        options=[XOption(label="Gym A", attributes={"Monthly Fee": XValue(value=1800, unit="₹")})],
        criteria=[XCriterion(attribute="monthly fee", direction="minimize", priority="high")],
    )
    f = to_frame(x, "t")
    v = f.options[0].attributes["monthly_fee"]
    assert v.unit == "INR" and v.provenance is Provenance.INFERRED
    assert f.criteria[0].attribute == "monthly_fee" and f.criteria[0].weight == 3.0


def test_rules_extract_seen_domain_perfectly(scenarios):
    s = first(scenarios, "career")
    sc = score_fields(to_frame(extract(render(s)), s.frame.situation), gold_frame(s))
    assert all(c.fp == 0 and c.fn == 0 for c in sc.counts.values()) and sc.consistent


def test_rules_miss_unseen_domain_vocabulary(scenarios):
    s = first(scenarios, "education")
    sc = score_fields(to_frame(extract(render(s)), s.frame.situation), gold_frame(s))
    assert sc.counts["criteria"].fn > 0  # doesn't know "placement rate", "tuition", ...


def test_perfect_extraction_gives_same_decision(scenarios):
    subset = [s for s in scenarios if s.truth.domain not in ("travel", "education")][:60]
    _, report = evaluate_extractor(subset, extract)
    assert report["decision_agreement"] >= 0.95


def test_value_and_attr_matching():
    assert value_match(120, 120000) and value_match(72000, 72000.4) and not value_match(1, 2)
    assert value_match(True, True) and not value_match(True, 1)
    assert attr_match("commute_time", "commute_minutes")
    assert attr_match("monthly_cost_usd", "monthly_cost")  # share the phrase "monthly cost"
    assert not attr_match("rent", "salary")


def test_inconsistent_names_are_not_perfect(scenarios):
    s = first(scenarios, "productivity")
    g = gold_frame(s)
    x = g.model_copy(deep=True)
    a = x.criteria[0].attribute
    x.criteria[0] = x.criteria[0].model_copy(update={"attribute": a + "_renamed"})
    assert not score_fields(x, g).consistent


def test_parse_json_tolerates_chatter_and_trailing_commas():
    raw = 'Sure! Here it is:\n{"options": [{"label": "A", "attributes": {"x": {"value": 1}},}], "criteria": [],}\nDone.'
    x = parse_json(raw)
    assert x.options[0].label == "A"
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_grounding_drops_numbers_not_in_text():
    from adapt.extraction.grounding import ground
    from adapt.extraction.schema import XConstraint

    text = "Gym A costs ₹1,800 and Gym B costs $2.5k. Lower cost is better."
    x = Extraction(
        options=[
            XOption(label="Gym A", attributes={"cost": XValue(value=1800), "pool": XValue(value=True)}),
            XOption(label="Gym B", attributes={"cost": XValue(value=2500), "rating": XValue(value=9)}),
        ],
        constraints=[XConstraint(attribute="cost", operator="<=", value=100000)],  # never stated
    )
    g, dropped = ground(x, text)
    assert dropped == 2  # the made-up rating and the made-up limit
    assert set(g.options[0].attributes) == {"cost", "pool"} and set(g.options[1].attributes) == {"cost"}
    assert g.constraints == []


def test_paraphrased_priority_and_direction_are_normalised():
    c = XCriterion(attribute="x", direction="lower", priority="nice_to_have")
    assert (c.direction, c.priority) == ("minimize", "low")
    assert XCriterion(attribute="x", direction="max", priority="most_important").priority == "high"
    assert XCriterion(attribute="x", direction="maximize", priority="whatever").priority is None
    with pytest.raises(ValueError):
        XCriterion(attribute="x", direction="sideways")


def test_extracted_frame_runs_through_policy(scenarios):
    s = first(scenarios, "purchase_electronics")
    r = decide(to_frame(extract(render(s)), s.frame.situation))
    assert r.outcome.value in ("recommend", "clarify", "abstain")
