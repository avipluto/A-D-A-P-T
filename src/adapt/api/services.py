"""Load the components the API uses, falling back to simpler ones when needed.

- Scorer: learned preference model if its checkpoint loads (decision D6), else the
  linear baseline.
- Extractor: the from-scratch tagger + clean-up if its checkpoint loads, else rules.
  Extracted frames always need user confirmation before a decision (phase 3 finding).
- Retrieval: Tavily if TAVILY_API_KEY is available, else off.

Every fallback is recorded and shown by /model-info, so the running setup is never a guess.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from adapt import config
from adapt.engine import ENGINE_VERSION
from adapt.engine.scoring import BASELINE, Scorer
from adapt.extraction.schema import Extraction
from adapt.schema import SCHEMA_VERSION

log = logging.getLogger(__name__)

DEFAULT_SCORER = config.ARTIFACTS_DIR / "models" / "pref-v1-s0" / "model.pt"
DEFAULT_TAGGER = config.ARTIFACTS_DIR / "models" / "tagger-v1-s0" / "model.pt"


@dataclass
class Services:
    scorer: Scorer
    extract: Callable[[str], Extraction]
    extractor_version: str
    provider: object | None  # SearchProvider
    notes: list[str] = field(default_factory=list)  # fallbacks and why

    def versions(self) -> dict:
        return {
            "engine": ENGINE_VERSION,
            "schema": SCHEMA_VERSION,
            "scorer": self.scorer.version,
            "extractor": self.extractor_version,
            "retrieval": getattr(self.provider, "name", None),
        }


NUMPY_SCORER = config.REPO_ROOT / "models" / "scorer.npz"
NUMPY_TAGGER = config.REPO_ROOT / "models" / "tagger.npz"


def _first(loaders: list[tuple[str, Callable]], notes: list[str], what: str):
    """Try each (label, loader) in order; record why earlier ones failed."""
    errors = []
    for label, loader in loaders:
        try:
            return loader()
        except Exception as e:  # noqa: BLE001 - any failure means: try the next, and say so
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:100]}")
    notes.append(f"{what}: fallback in use ({'; '.join(errors)})")
    return None


def load(scorer_path: Path | None = None, tagger_path: Path | None = None, enable_retrieval: bool = True) -> Services:
    """ADAPT_RUNTIME=numpy skips PyTorch entirely (deployment); the default tries PyTorch
    checkpoints first, then the NumPy exports in models/, then baseline / rules."""
    notes: list[str] = []
    numpy_only = os.environ.get("ADAPT_RUNTIME", "auto") == "numpy"

    from adapt.lifecycle.registry import production_scorer

    sp = Path(os.environ.get("ADAPT_SCORER_CHECKPOINT", scorer_path or production_scorer(DEFAULT_SCORER)))
    snp = Path(os.environ.get("ADAPT_SCORER_NPZ", NUMPY_SCORER))

    def torch_scorer():
        from adapt.learning.model import LearnedScorer

        return LearnedScorer.from_checkpoint(sp)

    def numpy_scorer():
        from adapt.inference.numpy_models import NumpyScorer

        return NumpyScorer.from_file(snp)

    loaders = [("numpy", numpy_scorer)] if numpy_only else [("torch", torch_scorer), ("numpy", numpy_scorer)]
    scorer: Scorer = _first(loaders, notes, "scorer") or BASELINE

    from adapt.extraction.postprocess import clean
    from adapt.extraction.rules import extract as rules_extract

    tp = Path(os.environ.get("ADAPT_TAGGER_CHECKPOINT", tagger_path or DEFAULT_TAGGER))
    tnp = Path(os.environ.get("ADAPT_TAGGER_NPZ", NUMPY_TAGGER))

    def torch_tagger():
        from adapt.tagger.extractor import TaggerExtractor

        return TaggerExtractor.from_checkpoint(tp)

    def numpy_tagger():
        from adapt.inference.numpy_models import NumpyTagger

        return NumpyTagger.from_file(tnp)

    loaders = [("numpy", numpy_tagger)] if numpy_only else [("torch", torch_tagger), ("numpy", numpy_tagger)]
    tagger = _first(loaders, notes, "extractor")
    if tagger is not None:
        extract = lambda text: clean(tagger(text), text)[0]  # noqa: E731
        extractor_version = f"{tagger.version}+clean"
    else:
        extract, extractor_version = rules_extract, "rules-v1"

    provider = None
    if enable_retrieval:
        try:
            from adapt.retrieval.providers import TavilyProvider

            provider = TavilyProvider()
        except Exception as e:  # noqa: BLE001
            notes.append(f"retrieval: off ({str(e)[:120]})")

    return Services(scorer, extract, extractor_version, provider, notes)
