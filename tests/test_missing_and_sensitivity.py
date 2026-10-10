import pytest

from adapt.engine import Outcome, decide
from adapt.engine.missing import Slot, fill_values
from adapt.engine.scoring import score_options
from adapt.engine.sensitivity import weight_flips
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


def opt(id, **attrs):
    return Option(id=id, label=id.upper(), attributes={k: Value(value=x, provenance=Provenance.STATED) for k, x in attrs.items()})


# --- plausible fill values ---------------------------------------------------


def test_fill_values_extend_beyond_known_range_and_stay_non_negative():
    options = [opt("a", q=2), opt("b", q=6), opt("c")]
    vals = fill_values(Slot("c", "q"), options, [Criterion(attribute="q", direction=MAX)], [], extension=1.0)
    assert vals[0] == 0.0 and vals[-1] == 10.0  # 2 - 4 clamped at 0; 6 + 4
    assert {2.0, 4.0, 6.0} <= set(vals)


def test_fill_values_use_given_plausible_range():
    crit = [Criterion(attribute="q", direction=MAX, plausible_range=(0, 100))]
    vals = fill_values(Slot("c", "q"), [opt("a", q=2), opt("c")], crit, [], extension=1.0)
    assert vals[0] == 0 and vals[-1] == 100


def test_bool_slot_tries_both_values():
    c = Constraint(attribute="remote", operator=Operator.EQ, value=True)
    assert fill_values(Slot("a", "remote"), [opt("a")], [], [c], 1.0) == [True, False]


def test_text_slot_cannot_be_bounded():
    assert fill_values(Slot("b", "colour"), [opt("a", colour="red"), opt("b")], [], [], 1.0) is None


def test_plausible_range_must_be_ordered():
    with pytest.raises(ValueError):
        Criterion(attribute="q", direction=MAX, plausible_range=(5, 1))


# --- missing values in the policy ---------------------------------------------


def frame(options, criteria, constraints=()):
    return DecisionFrame(situation="t", options=options, criteria=criteria, constraints=list(constraints))


def test_missing_value_beyond_observed_range_now_triggers_question():
    # Within the known q range [5, 6], a always wins. But b's q could plausibly be 7
    # (one spread above the known values); then b scores 0.95 vs a's 0.75. So ask.
    options = [opt("a", cost=5, q=6), opt("b", cost=5.2), opt("c", cost=7, q=5)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.5), Criterion(attribute="q", direction=MAX, weight=0.5)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.CLARIFY
    assert [(q.attribute, q.option_ids) for q in r.questions] == [("q", ["b"])]


def test_robust_recommendation_states_its_assumption():
    options = [opt("a", cost=1, q=5), opt("b", cost=9), opt("c", cost=5, q=1)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.9), Criterion(attribute="q", direction=MAX, weight=0.1)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "a"
    assert any("Assumed q of B" in n for n in r.notes)


def test_only_deciding_slot_is_asked_about():
    # c's missing q can't save it (cost dominates); b's missing q can.
    options = [opt("a", cost=5, q=6), opt("b", cost=5.5), opt("c", cost=100)]
    crit = [Criterion(attribute="cost", direction=MIN, weight=0.5), Criterion(attribute="q", direction=MAX, weight=0.5)]
    r = decide(frame(options, crit))
    assert r.outcome is Outcome.CLARIFY
    assert [q.option_ids for q in r.questions] == [["b"]]


def test_never_recommends_on_an_assumed_hard_limit():
    # a wins whatever its remote value... only if it is allowed at all. Must ask, not assume.
    c = Constraint(attribute="remote", operator=Operator.EQ, value=True)
    options = [opt("a", cost=1), opt("b", cost=9, remote=False)]
    r = decide(frame(options, [Criterion(attribute="cost", direction=MIN, weight=1)], [c]))
    assert r.outcome is Outcome.CLARIFY
    assert [(q.attribute, q.option_ids) for q in r.questions] == [("remote", ["a"])]


def test_missing_constraint_value_that_cannot_matter_is_not_asked():
    c = Constraint(attribute="remote", operator=Operator.EQ, value=True)
    options = [opt("a", cost=1, remote=True), opt("b", cost=9)]
    r = decide(frame(options, [Criterion(attribute="cost", direction=MIN, weight=1)], [c]))
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "a"


# --- sensitivity -------------------------------------------------------------


def test_weight_flip_is_exact():
    # u: a=(price 1, speed 0), b=(price 0, speed 1). With price share t: a=t, b=1-t, flip at 0.5.
    options = [opt("a", price=1, speed=0), opt("b", price=2, speed=1)]
    crit = [Criterion(attribute="price", direction=MIN), Criterion(attribute="speed", direction=MAX)]
    weights = {"price": 0.7, "speed": 0.3}
    flips = weight_flips(score_options(options, crit, weights), weights, "a")
    by_attr = {f.attribute: f for f in flips}
    assert by_attr["price"].flip_weight == pytest.approx(0.5)
    assert by_attr["price"].new_leader == "b"
    assert by_attr["speed"].flip_weight == pytest.approx(0.5)


def test_flip_matches_brute_force_rescoring():
    options = [opt("a", x=3, y=9, z=1), opt("b", x=7, y=2, z=4), opt("c", x=5, y=5, z=9)]
    crit = [Criterion(attribute=k, direction=MAX) for k in "xyz"]
    w = {"x": 0.5, "y": 0.3, "z": 0.2}
    scored = score_options(options, crit, w)
    leader = max(scored, key=lambda s: s.low).option_id
    for f in weight_flips(scored, w, leader):
        # just past the flip point, rescoring from scratch must give the new leader
        step = 1e-4 if f.flip_weight > f.current_weight else -1e-4
        t = f.flip_weight + step
        rest = {k: w[k] * (1 - t) / (1 - w[f.attribute]) for k in w if k != f.attribute}
        s2 = score_options(options, crit, {f.attribute: t, **rest})
        assert max(s2, key=lambda s: s.low).option_id == f.new_leader


def test_close_call_is_marked_fragile():
    options = [opt("a", price=1, speed=0), opt("b", price=2, speed=1)]
    crit = [Criterion(attribute="price", direction=MIN, weight=0.55), Criterion(attribute="speed", direction=MAX, weight=0.45)]
    r = decide(frame(options, crit))
    assert r.recommended == "a" and r.fragile
    assert "B would win if price's share of importance went from 0.55 to 0.50." in r.notes
