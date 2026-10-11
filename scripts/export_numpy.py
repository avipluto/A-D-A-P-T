"""Export the production scorer and the tagger to NumPy weights in the repo's models/ folder.

    python scripts/export_numpy.py

The deployed server (Dockerfile, render.yaml) loads these and doesn't need PyTorch.
Re-run after promoting a new model, then commit models/.
"""

from __future__ import annotations

from adapt import config
from adapt.api.services import DEFAULT_SCORER, DEFAULT_TAGGER
from adapt.inference.export import export_scorer, export_tagger
from adapt.lifecycle.registry import production_scorer


def main() -> None:
    out = config.REPO_ROOT / "models"
    s = export_scorer(production_scorer(DEFAULT_SCORER), out / "scorer.npz")
    t = export_tagger(DEFAULT_TAGGER, out / "tagger.npz")
    for p in (s, t):
        print(f"{p.relative_to(config.REPO_ROOT)}  {p.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
