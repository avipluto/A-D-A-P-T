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


def load(scorer_path: Path | None = None, tagger_path: Path | None = None, enable_retrieval: bool = True) -> Services:
    notes: list[str] = []

    scorer: Scorer = BASELINE
    sp = Path(os.environ.get("ADAPT_SCORER_CHECKPOINT", scorer_path or DEFAULT_SCORER))
    try:
        from adapt.learning.model import LearnedScorer

        scorer = LearnedScorer.from_checkpoint(sp)
    except Exception as e:  # noqa: BLE001 - any failure means: use the baseline and say so
        notes.append(f"scorer: using linear baseline ({type(e).__name__}: {str(e)[:120]})")

    from adapt.extraction.postprocess import clean
    from adapt.extraction.rules import extract as rules_extract

    tp = Path(os.environ.get("ADAPT_TAGGER_CHECKPOINT", tagger_path or DEFAULT_TAGGER))
    try:
        from adapt.tagger.extractor import TaggerExtractor

        tagger = TaggerExtractor.from_checkpoint(tp)
        extract = lambda text: clean(tagger(text), text)[0]  # noqa: E731
        extractor_version = f"{tagger.version}+clean"
    except Exception as e:  # noqa: BLE001
        extract, extractor_version = rules_extract, "rules-v1"
        notes.append(f"extractor: using rules ({type(e).__name__}: {str(e)[:120]})")

    provider = None
    if enable_retrieval:
        try:
            from adapt.retrieval.providers import TavilyProvider

            provider = TavilyProvider()
        except Exception as e:  # noqa: BLE001
            notes.append(f"retrieval: off ({str(e)[:120]})")

    return Services(scorer, extract, extractor_version, provider, notes)
