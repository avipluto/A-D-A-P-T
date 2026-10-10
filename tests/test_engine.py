import pytest

from adapt.engine import Outcome, decide
from adapt.engine.constraints import Status, check
from adapt.engine.scoring import dominant_option, score_options
from adapt.schema import (
    Constraint,
    Criterion,
    DecisionFrame,
    Direction,
    Operator,
    Option,
    Provenance,
    Value,
)

MAX, MIN = Direction.MAXIMIZE, Direction.MINIMIZE


def v(x, unit=None):
    return Value(value=x, unit=unit, provenance=Provenance.STATED)


def opt(id, **attrs):
    return Option(id=id, label=id.upper(), attributes={k: v(x) for k, x in attrs.items()})


def frame(options, criteria, constraints=()):
    return DecisionFrame(situation="test", options=options, criteria=criteria, constraints=list(constraints))


# --- constraints -------------------------------------------------------------


def test_constraint_pass_fail_unknown():
    c = Constraint(attribute="price", operator=Operator.LE, value=100)
    assert check(opt("a", price=90), c).status is Status.PASS
    assert check(opt("a", price=110), c).status is Status.FAIL
    assert check(opt("a"), c).status is Status.UNKNOWN


def test_constraint_unit_mismatch_is_unknown_not_pass():
    c = Constraint(attribute="price", operator=Operator.LE, value=100, unit="USD")
    o = Option(id="a", label="A", attributes={"price": v(90, "INR")})
    assert check(o, c).status is Status.UNKNOWN


def test_ordering_strings_is_unknown():
    c = Constraint(attribute="size", operator=Operator.GE, value=3)
    assert check(opt("a", size="large"), c).status is Status.UNKNOWN


# --- scoring -----------------------------------------------------------------


def test_weighted_score_arithmetic():
    options = [opt("a", price=100, speed=10), opt("b", price=200, speed=30), opt("c", price=150, speed=20)]
    criteria = [Criterion(attribute="price", direction=MIN), Criterion(attribute="speed", direction=MAX)]
    s = {x.option_id: x for x in score_options(options, criteria, {"price": 3, "speed": 1})}
    # weights normalise to 0.75/0.25; price a=1, b=0, c=0.5; speed a=0, b=1, c=0.5
    assert s["a"].low == pytest.approx(0.75)
    assert s["b"].low == pytest.approx(0.25)
    assert s["c"].low == pytest.approx(0.5)


def test_equal_values_score_half():
    options = [opt("a", x=5), opt("b", x=5)]
    s = score_options(options, [Criterion(attribute="x", direction=MAX)], {"x": 1})
    assert [x.low for x in s] == [0.5, 0.5]


def test_missing_value_gives_interval():
    options = [opt("a", x=1, y=1), opt("b", x=2)]
    crit = [Criterion(attribute="x", direction=MAX), Criterion(attribute="y", direction=MAX)]
    b = next(s for s in score_options(options, crit, {"x": 1, "y": 1}) if s.option_id == "b")
    assert (b.low, b.high) == pytest.approx((0.5, 1.0))
    assert b.missing == ["y"]


def test_dominance():
    crit = [Criterion(attribute="cost", direction=MIN), Criterion(attribute="q", direction=MAX)]
    assert dominant_option([opt("a", cost=1, q=9), opt("b", cost=2, q=5)], crit) == "a"
    assert dominant_option([opt("a", cost=1, q=5), opt("b", cost=2, q=9)], crit) is None


# --- policy ------------------------------------------------------------------


def test_no_options_asks():
    r = decide(frame([], [Criterion(attribute="x", direction=MAX, weight=1)]))
    assert r.outcome is Outcome.CLARIFY


def test_all_infeasible_abstains_and_explains():
    c = Constraint(attribute="price", operator=Operator.LE, value=50)
    r = decide(frame([opt("a", price=90), opt("b", price=70)], [Criterion(attribute="price", direction=MIN, weight=1)], [c]))
    assert r.outcome is Outcome.ABSTAIN
    assert {e.option_id for e in r.excluded} == {"a", "b"}
    assert r.excluded[0].failed == ["price <= 50"]


def test_hard_limit_beats_score():
    c = Constraint(attribute="relocate", operator=Operator.EQ, value=False)
    options = [opt("a", salary=40, relocate=True), opt("b", salary=10, relocate=False)]
    r = decide(frame(options, [Criterion(attribute="salary", direction=MAX, weight=1)], [c]))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "b"


def test_clear_winner_recommended():
    options = [opt("a", cost=1, q=9), opt("b", cost=5, q=2)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.5), Criterion(attribute="q", direction=MAX, weight=0.5)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "a"
    assert [x.option_id for x in r.ranking] == ["a", "b"]


def test_missing_value_that_matters_asks_about_it():
    # c leads at 0.7; b scores 0.3..1.0 depending on its unknown q, so it could overtake c
    options = [opt("a", cost=5, q=5), opt("b", cost=4), opt("c", cost=6, q=9)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.3), Criterion(attribute="q", direction=MAX, weight=0.7)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.CLARIFY
    assert [(q.attribute, q.option_ids) for q in r.questions] == [("q", ["b"])]


def test_single_known_value_is_uncertain_too():
    # b is cheaper; its q is unknown and a's q is the only reference point,
    # so neither q contribution is certain. b still can't lose: recommend b.
    options = [opt("a", cost=5, q=5), opt("b", cost=4)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.5), Criterion(attribute="q", direction=MAX, weight=0.5)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "b"
    a = next(x for x in r.ranking if x.option_id == "a")
    assert (a.score_low, a.score_high) == (0.0, 0.5)


def test_missing_value_that_cannot_matter_is_not_asked():
    # b is worse on cost (weight 0.9); even the best possible q can't save it
    options = [opt("a", cost=1, q=5), opt("b", cost=9), opt("c", cost=5, q=1)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.9), Criterion(attribute="q", direction=MAX, weight=0.1)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "a"


def test_unknown_weights_with_dominance_recommends():
    crit = [Criterion(attribute="cost", direction=MIN), Criterion(attribute="q", direction=MAX)]
    r = decide(frame([opt("a", cost=1, q=9), opt("b", cost=2, q=5)], crit))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "a"


def test_unknown_weights_without_dominance_asks_priorities():
    crit = [Criterion(attribute="cost", direction=MIN), Criterion(attribute="q", direction=MAX)]
    r = decide(frame([opt("a", cost=1, q=5), opt("b", cost=2, q=9)], crit))
    assert r.outcome is Outcome.CLARIFY
    assert {q.attribute for q in r.questions} == {"cost", "q"}


def test_unknown_constraint_on_strong_option_asks():
    c = Constraint(attribute="open_source", operator=Operator.EQ, value=True)
    options = [opt("a", cost=1), opt("b", cost=9, open_source=True)]
    r = decide(frame(options, [Criterion(attribute="cost", direction=MIN, weight=1)], [c]))
    assert r.outcome is Outcome.CLARIFY
    assert r.questions[0].attribute == "open_source"
