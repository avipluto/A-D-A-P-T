from collections import defaultdict

import pytest

from adapt.engine import decide
from adapt.evaluation.metrics import evaluate, ndcg
from adapt.synthetic import generate
from adapt.synthetic.domains import UNSEEN_DOMAINS


@pytest.fixture(scope="module")
def scenarios():
    return list(generate(n_per_domain=80, seed=1))


def test_generation_is_deterministic():
    a = [s.model_dump_json() for s in generate(20, seed=3)]
    b = [s.model_dump_json() for s in generate(20, seed=3)]
    assert a == b


def test_every_kind_and_domain_is_covered(scenarios):
    assert {s.truth.kind for s in scenarios} == {
        "complete", "missing_weight", "missing_value", "infeasible", "no_options"
    }
    assert len({s.truth.domain for s in scenarios}) == 9


def test_infeasible_has_no_true_best(scenarios):
    for s in scenarios:
        if s.truth.kind == "infeasible":
            assert s.truth.best_ids == []
        elif s.truth.kind != "no_options":
            assert s.truth.best_ids, s.truth.scenario_id


def test_missing_value_is_actually_hidden(scenarios):
    for s in scenarios:
        if s.truth.kind == "missing_value":
            assert s.frame.missing_attributes() or any(
                c.attribute not in o.attributes for o in s.frame.options for c in s.frame.constraints
            )


def test_families_never_cross_splits(scenarios):
    splits = defaultdict(set)
    for s in scenarios:
        splits[s.truth.family].add(s.truth.split)
    assert all(len(v) == 1 for v in splits.values())


def test_unseen_domains_only_in_their_test_split(scenarios):
    for s in scenarios:
        assert (s.truth.domain in UNSEEN_DOMAINS) == (s.truth.split == "test_unseen_domain")


def test_ndcg():
    gains = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg(["a", "b", "c"], gains) == pytest.approx(1.0)
    assert ndcg(["c", "b", "a"], gains) < 1.0


def test_baseline_never_violates_hard_limits(scenarios):
    _, report = evaluate(scenarios, decide)
    assert report["overall"]["constraint_violations"] == 0
    assert report["by_kind"]["infeasible"]["abstain_recall_infeasible"] == 1.0
    assert report["by_kind"]["no_options"]["outcome_accuracy"] == 1.0
