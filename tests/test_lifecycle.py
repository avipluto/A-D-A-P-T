import json

import pytest

from adapt.api.store import Store
from adapt.engine import decide
from adapt.engine.scoring import BASELINE, Scored
from adapt.lifecycle import feedback_data, registry
from adapt.lifecycle.gate import Tolerances, run_gate
from adapt.schema import Constraint, Criterion, DecisionFrame, Direction, Operator, Option, Provenance, Value


def stated(x):
    return Value(value=x, provenance=Provenance.STATED)


def frame(seed: int = 0) -> DecisionFrame:
    return DecisionFrame(
        situation=f"laptops {seed}",
        options=[Option(id="a", label="A", attributes={"price": stated(70 + seed), "battery": stated(9)}),
                 Option(id="b", label="B", attributes={"price": stated(55), "battery": stated(6)}),
                 Option(id="c", label="C", attributes={"price": stated(99), "battery": stated(12)})],
        criteria=[Criterion(attribute="price", direction=Direction.MINIMIZE, weight=3),
                  Criterion(attribute="battery", direction=Direction.MAXIMIZE, weight=1)],
        constraints=[Constraint(attribute="price", operator=Operator.LE, value=90)],  # c is excluded
    )


def store_with(tmp_path, feedback: list[tuple[str, int | None, str | None, str]]) -> tuple[Store, list[str]]:
    """feedback: (kind, rating, chosen, review_status)"""
    s = Store(tmp_path / "f.sqlite")
    f = frame()
    did = s.save_decision(input_kind="form", status="decided", request={}, frame=f.model_dump(mode="json"),
                          result=decide(f).model_dump(mode="json"), versions={})
    ids = []
    for kind, rating, chosen, status in feedback:
        fid = s.save_feedback(decision_id=did, kind=kind, chosen_option_id=chosen, rating=rating, details=None, notes=None)
        ids.append(fid)
        if status != "pending":
            import sqlite3

            with sqlite3.connect(s.path) as c:
                c.execute("UPDATE feedback SET review_status=? WHERE id=?", (status, fid))
    return s, ids


# --- what counts as a training signal -----------------------------------------


def test_weights_separate_satisfaction_from_outcome():
    assert feedback_data.weight_for(outcome=5, satisfaction=1) == 1.0  # it worked, even if they didn't like the advice
    assert feedback_data.weight_for(outcome=3, satisfaction=None) == 0.5
    assert feedback_data.weight_for(outcome=1, satisfaction=5) == 0.0  # liked it, but it turned out badly
    assert feedback_data.weight_for(outcome=None, satisfaction=5) == 0.5
    assert feedback_data.weight_for(outcome=None, satisfaction=2) == 0.0


def test_only_accepted_feedback_is_exported(tmp_path):
    s, _ = store_with(tmp_path, [("outcome", 5, "a", "pending")])
    assert feedback_data.collect(s.path) == []
    s2, _ = store_with(tmp_path / "x", [("outcome", 5, "a", "rejected")])
    assert feedback_data.collect(s2.path) == []
    s3, _ = store_with(tmp_path / "y", [("outcome", 5, "a", "accepted")])
    [ex] = feedback_data.collect(s3.path)
    assert ex.chosen == "a" and ex.weight == 1.0


def test_choosing_an_excluded_option_is_not_training_data(tmp_path):
    s, _ = store_with(tmp_path, [("outcome", 5, "c", "accepted")])  # c broke the hard limit
    assert feedback_data.collect(s.path) == []


def test_export_is_versioned_with_manifest(tmp_path):
    s, ids = store_with(tmp_path, [("satisfaction", 5, "b", "accepted")])
    path = feedback_data.export(s.path, tmp_path / "out")
    m = json.loads(path.with_suffix(".manifest.json").read_text())
    assert m["examples"] == 1 and m["feedback_ids"] == ids and m["sha256_12"] in path.name
    assert feedback_data.load(path)[0].chosen == "b"


def test_real_pairs_put_the_chosen_option_above_the_other_feasible_ones():
    torch = pytest.importorskip("torch")
    from adapt.learning.data import build_real_pairs, concat

    ex = feedback_data.RealExample("d1", frame(), "a", 1.0, "train", ["f1"])
    d = build_real_pairs([ex], lambda a: 0, scale=5.0)
    assert d.n_options == 2  # c is excluded by the hard limit
    assert d.pairs.tolist() == [[0, 1]] and d.pair_weight.tolist() == [5.0]
    both = concat(d, d)
    assert both.pairs.tolist() == [[0, 1], [2, 3]] and torch.equal(both.pair_weight, torch.tensor([5.0, 5.0]))


# --- regression gate --------------------------------------------------------------


class Inverted:
    """A deliberately bad scorer: ranks options upside down."""

    version = "inverted"

    def __call__(self, options, criteria, weights):
        return [Scored(s.option_id, 1 - s.high, 1 - s.low, s.contributions, {k: (1 - b, 1 - a) for k, (a, b) in s.unit.items()},
                       s.missing) for s in BASELINE(options, criteria, weights)]


def test_gate_passes_an_identical_model():
    report = run_gate(BASELINE, BASELINE, n_per_domain=30)
    assert report["passed"], report["checks"]


def test_gate_rejects_a_worse_model():
    report = run_gate(Inverted(), BASELINE, n_per_domain=30)
    assert not report["passed"]
    failed = [c["check"] for c in report["checks"] if not c["passed"]]
    assert any("accuracy" in c for c in failed)


def test_too_few_real_decisions_are_reported_not_gated():
    ex = feedback_data.RealExample("d", frame(), "b", 1.0, "heldout", [])
    report = run_gate(BASELINE, BASELINE, n_per_domain=20, real_heldout=[ex], tol=Tolerances(min_real=30))
    real_check = next(c for c in report["checks"] if "real" in c["check"])
    assert real_check["passed"] and "not gated" in real_check["detail"]


# --- promotion --------------------------------------------------------------------


def test_promotion_needs_a_passing_report_for_that_checkpoint(tmp_path):
    ck = tmp_path / "m.pt"
    ck.write_text("x")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"passed": False, "candidate": str(ck)}))
    with pytest.raises(ValueError, match="did not pass"):
        registry.promote(ck, bad, "x", registry_dir=tmp_path)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"passed": True, "candidate": str(tmp_path / "different.pt")}))
    with pytest.raises(ValueError, match="different checkpoint"):
        registry.promote(ck, other, "x", registry_dir=tmp_path)

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"passed": True, "candidate": str(ck)}))
    first = registry.promote(ck, good, "first", registry_dir=tmp_path)
    second = registry.promote(ck, good, "again", registry_dir=tmp_path)
    assert first["previous"] is None and second["previous"] == str(ck)
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 2
