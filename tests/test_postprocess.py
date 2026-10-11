from adapt.extraction.postprocess import clean
from adapt.extraction.schema import Extraction, XConstraint, XCriterion, XOption, XValue


def opt(label, **attrs):
    return XOption(label=label, attributes={k: XValue(value=v) for k, v in attrs.items()})


TEXT = (
    "Itinerary A has price ₹2,785 and 2 stops. Itinerary B has price ₹9,000 and layovers 1. "
    "Lower price is better. I want fewer stops. The price can't be more than ₹10,000."
)


def test_limit_invented_from_a_preference_is_dropped():
    x = Extraction(
        options=[opt("Itinerary A", price=2785), opt("Itinerary B", price=9000)],
        constraints=[
            XConstraint(attribute="price", operator="<=", value=2785),  # number is in the text, but no limit words
            XConstraint(attribute="price", operator="<=", value=10000),  # really stated
        ],
    )
    out, log = clean(x, TEXT, names=False)
    assert [c.value for c in out.constraints] == [10000]
    assert log.dropped_limits == ["price <= 2785"]


def test_yes_no_limit_needs_limit_words_near_its_name():
    text = "Option A is refundable. It must be refundable."
    x = Extraction(options=[opt("Option A", refundable=True)],
                   constraints=[XConstraint(attribute="refundable", operator="==", value=True)])
    assert len(clean(x, text)[0].constraints) == 1
    assert clean(x, "Option A is refundable.")[0].constraints == []


def test_two_names_for_one_attribute_are_merged():
    x = Extraction(
        options=[opt("Itinerary A", price=2785, stops=2), opt("Itinerary B", price=9000, layovers=1)],
        criteria=[XCriterion(attribute="layovers", direction="minimize")],
    )
    out, log = clean(x, TEXT)
    assert log.merged_names == [("layovers", "stops")]
    assert all("stops" in o.attributes and "layovers" not in o.attributes for o in out.options)
    assert out.criteria[0].attribute == "stops"


def test_attributes_that_coexist_are_never_merged():
    x = Extraction(options=[opt("A", price=10, weight=2), opt("B", price=12, weight=3)])
    assert clean(x, "A: price 10, weight 2. B: price 12, weight 3.")[1].merged_names == []


def test_very_different_magnitudes_are_not_merged():
    x = Extraction(options=[opt("A", rent=25000), opt("B", rating=7)])
    assert clean(x, "A has rent 25000. B has rating 7.")[1].merged_names == []
