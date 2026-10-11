import pytest

torch = pytest.importorskip("torch")

from adapt.extraction.metrics import score_fields  # noqa: E402
from adapt.extraction.schema import to_frame  # noqa: E402
from adapt.extraction.textgen import gold_frame, render, render_tagged  # noqa: E402
from adapt.synthetic import generate  # noqa: E402
from adapt.tagger.decode import _flag_name, _unit, assemble, spans  # noqa: E402
from adapt.tagger.extractor import TaggerExtractor  # noqa: E402
from adapt.tagger.model import LABELS, TaggerModel  # noqa: E402
from adapt.tagger.tokenizer import NUM, UNK, Vocab, norm, trigrams  # noqa: E402


@pytest.fixture(scope="module")
def scenarios():
    return list(generate(20, seed=4))


def test_tagged_words_rebuild_the_exact_text(scenarios):
    for s in scenarios:
        w, t = render_tagged(s)
        assert " ".join(w) == render(s) and len(w) == len(t)
        assert all(x == "O" or x[:2] in ("B-", "I-") for x in t)


def test_decoding_gold_labels_recovers_options_and_values(scenarios):
    s = next(s for s in scenarios if s.truth.kind == "complete" and s.truth.domain == "career")
    x = to_frame(assemble(*render_tagged(s)), s.frame.situation)
    sc = score_fields(x, gold_frame(s))
    assert sc.counts["options"].fn == 0 and sc.counts["values"].fn == 0 and sc.counts["criteria"].fn == 0


def test_sentence_ending_in_a_number_still_ends_the_sentence():
    ws = "Tool A has stops of 1. Tool B has stops of 2.".split()
    labs = ["B-OPT", "I-OPT", "O", "B-ATTR", "O", "B-VAL", "B-OPT", "I-OPT", "O", "B-ATTR", "O", "B-VAL"]
    assert [sp.sentence for sp in spans(ws, labs) if sp.kind == "OPT"] == [0, 1]
    x = assemble(ws, labs)
    assert [o.attributes["stops"].value for o in x.options] == [1, 2]


def test_flag_names_line_up_between_fact_and_limit():
    assert _flag_name("doesn't require relocation.") == _flag_name("requires relocation.") == "require_relocation"
    assert _flag_name("is non-refundable.") == _flag_name("refundable") == "refundable"


def test_unit_is_what_surrounds_the_number():
    assert _unit("16.88 LPA.") == "LPA" and _unit("7/10,") == "/10" and _unit("45") is None


def test_tokenizer_handles_unseen_words():
    v = Vocab.build(["price of 5 price"], min_freq=1)
    assert v.id("price") > NUM and v.id("₹72,000") == NUM and v.id("tuition") == UNK
    assert norm("LPA.") == "lpa" and len(trigrams("tuition")) > 0


def test_untrained_model_runs_end_to_end(scenarios):
    texts = [render(s) for s in scenarios[:3]]
    model = TaggerModel(Vocab.build(texts, min_freq=1), d=32, layers=1, heads=2, ff=64)
    logits = model(**model.encode(texts))
    assert logits.shape[-1] == len(LABELS)
    out = TaggerExtractor(model, "t").batch(texts)
    assert len(out) == 3
