from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from adapt.schema import (
    Constraint,
    Criterion,
    DecisionFrame,
    Direction,
    Operator,
    Option,
    Provenance,
    Source,
    Value,
)


def stated(v, unit=None):
    return Value(value=v, unit=unit, provenance=Provenance.STATED)


def laptop_frame() -> DecisionFrame:
    return DecisionFrame(
        situation="Choosing a laptop for ML coursework under a fixed budget.",
        options=[
            Option(id="a", label="Laptop A", attributes={"price": stated(72000, "INR"), "vram_gb": stated(6)}),
            Option(id="b", label="Laptop B", attributes={"price": stated(58000, "INR")}),
        ],
        criteria=[
            Criterion(attribute="price", direction=Direction.MINIMIZE, weight=0.4),
            Criterion(attribute="vram_gb", direction=Direction.MAXIMIZE),
        ],
        constraints=[Constraint(attribute="price", operator=Operator.LE, value=75000, unit="INR")],
    )


def test_same_schema_fits_unrelated_domain():
    frame = DecisionFrame(
        situation="Pick between two job offers.",
        domain_hint="career",
        options=[
            Option(id="x", label="Offer X", attributes={"salary": stated(12, "LPA"), "relocate": stated(True)}),
            Option(id="y", label="Offer Y", attributes={"salary": stated(9, "LPA"), "relocate": stated(False)}),
        ],
        criteria=[Criterion(attribute="salary", direction=Direction.MAXIMIZE)],
        constraints=[Constraint(attribute="relocate", operator=Operator.EQ, value=False)],
    )
    assert frame.missing_attributes() == []


def test_missing_attributes_are_reported():
    assert laptop_frame().missing_attributes() == [("b", "vram_gb")]


def test_unknown_weights_are_surfaced_not_defaulted():
    assert laptop_frame().unweighted_criteria() == ["vram_gb"]


def test_retrieved_value_requires_source():
    with pytest.raises(ValidationError):
        Value(value=64999, provenance=Provenance.RETRIEVED)
    ok = Value(
        value=64999,
        provenance=Provenance.RETRIEVED,
        source=Source(url="https://example.com/p", retrieved_at=datetime.now(timezone.utc)),
    )
    assert ok.source is not None


def test_assumptions_are_listed():
    frame = laptop_frame()
    frame.options[1].attributes["vram_gb"] = Value(value=4, provenance=Provenance.ASSUMED)
    assert frame.assumptions() == ["b.vram_gb"]


def test_duplicate_option_ids_rejected():
    with pytest.raises(ValidationError):
        DecisionFrame(situation="s", options=[Option(id="a", label="1"), Option(id="a", label="2")])


def test_negative_weight_rejected():
    with pytest.raises(ValidationError):
        Criterion(attribute="price", direction=Direction.MINIMIZE, weight=-1)
