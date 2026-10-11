"""Paths kept outside the OneDrive-synced code folder.

Datasets, checkpoints and the SQLite feedback store live under ADAPT_DATA_DIR
(default D:/adapt-data) so OneDrive never syncs or locks large training files.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ADAPT_DATA_DIR", "D:/adapt-data"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"  # never mixed with training data
FEEDBACK_DB = DATA_DIR / "feedback" / "feedback.sqlite"  # production feedback, reviewed before use
ARTIFACTS_DIR = DATA_DIR / "artifacts"  # checkpoints, tokenizers, run outputs


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, EVALUATION_DIR, FEEDBACK_DB.parent, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
