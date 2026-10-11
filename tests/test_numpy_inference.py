"""NumPy inference must match PyTorch: same scores, same labels, same decisions."""

from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adapt import config  # noqa: E402
from adapt.api.services import DEFAULT_SCORER, DEFAULT_TAGGER  # noqa: E402
from adapt.engine import decide  # noqa: E402
from adapt.extraction.textgen import render  # noqa: E402
from adapt.inference.export import export_scorer, export_tagger  # noqa: E402
from adapt.inference.numpy_models import NumpyScorer, NumpyTagger  # noqa: E402
from adapt.learning.model import LearnedScorer  # noqa: E402
from adapt.synthetic import generate  # noqa: E402
from adapt.tagger.extractor import TaggerExtractor  # noqa: E402

needs_checkpoints = pytest.mark.skipif(not (DEFAULT_SCORER.exists() and DEFAULT_TAGGER.exists()),
                                       reason="trained checkpoints not available")


@pytest.fixture(scope="module")
def scenarios():
    return [s for s in generate(60, seed=7) if s.truth.split in ("test_seen", "test_unseen_domain")]


@needs_checkpoints
def test_scorer_matches_torch(tmp_path, scenarios):
    torch_s = LearnedScorer.from_checkpoint(DEFAULT_SCORER)
    np_s = NumpyScorer.from_file(export_scorer(DEFAULT_SCORER, tmp_path / "s.npz"))
    assert np_s.version == torch_s.version
    compared = 0
    for s in scenarios:
        f = s.frame
        if not f.options or any(c.weight is None for c in f.criteria):
            continue
        w = {c.attribute: c.weight for c in f.criteria}
        a = {x.option_id: (x.low, x.high) for x in torch_s(f.options, f.criteria, w)}
        b = {x.option_id: (x.low, x.high) for x in np_s(f.options, f.criteria, w)}
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], abs=1e-5)
        compared += 1
    assert compared > 50


@needs_checkpoints
def test_same_decisions_with_numpy_scorer(tmp_path, scenarios):
    torch_s = LearnedScorer.from_checkpoint(DEFAULT_SCORER)
    np_s = NumpyScorer.from_file(export_scorer(DEFAULT_SCORER, tmp_path / "s.npz"))
    for s in scenarios:
        a, b = decide(s.frame, scorer=torch_s), decide(s.frame, scorer=np_s)
        assert (a.outcome, a.recommended) == (b.outcome, b.recommended), s.truth.scenario_id


@needs_checkpoints
def test_tagger_matches_torch(tmp_path, scenarios):
    torch_t = TaggerExtractor.from_checkpoint(DEFAULT_TAGGER)
    np_t = NumpyTagger.from_file(export_tagger(DEFAULT_TAGGER, tmp_path / "t.npz"))
    texts = [render(s) for s in scenarios[:40]]
    with torch.no_grad():
        enc = torch_t.model.encode(texts)
        ref = torch_t.model(**enc).numpy()
    got, _ = np_t.logits(texts)
    mask = enc["mask"].numpy()
    assert np.abs(ref[mask] - got[mask]).max() < 1e-3
    assert torch_t.labels(texts) == np_t.labels(texts)
    assert [x.model_dump() for x in torch_t.batch(texts)] == [x.model_dump() for x in np_t.batch(texts)]


def test_committed_numpy_weights_load_without_torch_code():
    s, t = config.REPO_ROOT / "models" / "scorer.npz", config.REPO_ROOT / "models" / "tagger.npz"
    if not (s.exists() and t.exists()):
        pytest.skip("models/ not exported")
    scorer, tagger = NumpyScorer.from_file(s), NumpyTagger.from_file(t)
    x = tagger("Help me pick a laptop. Laptop A has a price of ₹72,000. Laptop B: price ₹58,000. Lower price is better.")
    assert [o.label for o in x.options] == ["Laptop A", "Laptop B"]
    assert scorer.version
