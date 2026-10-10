import pytest

torch = pytest.importorskip("torch")

from adapt.engine import Outcome, decide  # noqa: E402
from adapt.learning.data import build_pairs, vocabulary  # noqa: E402
from adapt.learning.features import criterion_features  # noqa: E402
from adapt.learning.model import LearnedScorer, PreferenceModel  # noqa: E402
from adapt.learning.train import TrainConfig, train  # noqa: E402
from adapt.schema import (  # noqa: E402
    Constraint,
    Criterion,
    DecisionFrame,
    Direction,
    Operator,
    Option,
    Provenance,
    Value,
)
from adapt.synthetic import generate  # noqa: E402


def opt(id, **attrs):
    return Option(id=id, label=id.upper(), attributes={k: Value(value=x, provenance=Provenance.STATED) for k, x in attrs.items()})


def test_features_are_direction_aware_and_scale_free():
    options = [opt("a", cost=100), opt("b", cost=300)]
    f = criterion_features(options, Criterion(attribute="cost", direction=Direction.MINIMIZE))
    assert f.known["a"][0] == 1.0 and f.known["b"][0] == 0.0
    g = criterion_features(
        [opt("a", cost=1), opt("b", cost=3)], Criterion(attribute="cost", direction=Direction.MINIMIZE)
    )
    # same relative situation → same features, whatever the units
    for k in f.known:
        assert g.known[k] == pytest.approx(f.known[k])


def test_unknown_attribute_maps_to_generic_index():
    m = PreferenceModel(["salary"])
    assert m.attr_index("salary") == 1 and m.attr_index("never_seen") == 0
    assert PreferenceModel(["salary"], use_attr=False).attr_index("salary") == 0


def test_scorer_scores_in_unit_range_and_widens_for_missing():
    scorer = LearnedScorer(PreferenceModel(["x", "y"]), "test")
    options = [opt("a", x=1, y=2), opt("b", x=3), opt("c", x=2, y=5)]
    crit = [Criterion(attribute="x", direction=Direction.MAXIMIZE), Criterion(attribute="y", direction=Direction.MAXIMIZE)]
    s = {r.option_id: r for r in scorer(options, crit, {"x": 0.5, "y": 0.5})}
    for r in s.values():
        assert 0.0 <= r.low <= r.high <= 1.0
    assert s["b"].missing == ["y"] and s["b"].high > s["b"].low
    assert s["a"].low == s["a"].high


def test_learned_scorer_cannot_override_hard_limits():
    scorer = LearnedScorer(PreferenceModel(["salary"]), "test")
    frame = DecisionFrame(
        situation="t",
        options=[opt("a", salary=99, relocate=True), opt("b", salary=1, relocate=False)],
        criteria=[Criterion(attribute="salary", direction=Direction.MAXIMIZE, weight=1)],
        constraints=[Constraint(attribute="relocate", operator=Operator.EQ, value=False)],
    )
    r = decide(frame, scorer=scorer)
    assert r.outcome is Outcome.RECOMMEND and r.recommended == "b"
    assert r.scorer_version == "test"


def test_training_improves_pair_accuracy_and_roundtrips(tmp_path):
    scenarios = list(generate(40, seed=5))
    tr = [s for s in scenarios if s.truth.split != "test_unseen_domain"]
    model = PreferenceModel(vocabulary(tr))
    d = build_pairs(tr, model.attr_index)
    assert len(d.pairs) > 0
    stats = train(model, d, d, TrainConfig(epochs=150, patience=150), torch.device("cpu"))
    assert stats["val_pair_accuracy"] > 0.8

    path = tmp_path / "m.pt"
    model.save(path, {"model_version": "t"})
    loaded = LearnedScorer.from_checkpoint(path)
    options, crit = [opt("a", salary=10), opt("b", salary=30)], [Criterion(attribute="salary", direction=Direction.MAXIMIZE)]
    assert [x.low for x in loaded(options, crit, {"salary": 1})] == pytest.approx(
        [x.low for x in LearnedScorer(model, "t")(options, crit, {"salary": 1})]
    )
