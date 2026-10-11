"""PyTorch checkpoint → .npz weights + .json metadata for NumPy inference. Needs torch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_scorer(checkpoint: Path, out: Path) -> Path:
    from adapt.learning.model import PreferenceModel

    model, meta = PreferenceModel.load(checkpoint)
    return _write(model.state_dict(), {"kind": "preference", "vocab": model.vocab, "config": model.config,
                                       "meta": meta}, out)


def export_tagger(checkpoint: Path, out: Path) -> Path:
    from adapt.tagger.model import LABELS, TaggerModel

    model, meta = TaggerModel.load(checkpoint)
    return _write(model.state_dict(), {"kind": "tagger", "vocab": model.vocab.itos[3:], "config": model.config,
                                       "labels": LABELS, "meta": meta}, out)


def _write(state_dict, meta: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in state_dict.items()})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out
