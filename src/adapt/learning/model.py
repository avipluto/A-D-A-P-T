"""Additive learned preference model and its Scorer adapter.

    score(option) = Σ_c  w_c · g(features_c, attribute_c)

g is a small MLP with a sigmoid output, so every contribution stays in [0, w_c]
and scores stay on the same [0, 1] scale as the baseline. Keeping the sum additive
keeps per-criterion explanations possible.

Attribute identity is an embedding over names seen in training; index 0 means
"unknown attribute". During training names are randomly replaced by 0 so the model
also learns a generic shape it can use on attributes and domains it has never seen.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from adapt.learning.additive import AdditiveScorer
from adapt.learning.features import FEATURE_DIM


class PreferenceModel(nn.Module):
    def __init__(self, vocab: list[str], use_attr: bool = True, emb_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.vocab = list(vocab)
        self.use_attr = use_attr
        self.config = {"use_attr": use_attr, "emb_dim": emb_dim, "hidden": hidden}
        self._index = {name: i + 1 for i, name in enumerate(self.vocab)}
        self.emb = nn.Embedding(len(self.vocab) + 1, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM + emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.log_scale = nn.Parameter(torch.tensor(2.3))  # pairwise logit scale, ~10

    def attr_index(self, name: str) -> int:
        return self._index.get(name, 0) if self.use_attr else 0

    def value(self, feats: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        """g in (0, 1) for each row."""
        x = torch.cat([feats, self.emb(attr)], dim=-1)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "vocab": self.vocab, "config": self.config, "meta": meta},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple[PreferenceModel, dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(ckpt["vocab"], **ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt["meta"]


class LearnedScorer(AdditiveScorer):
    """Plugs a PreferenceModel into the decision policy. Runs on CPU: inference is tiny."""

    def __init__(self, model: PreferenceModel, version: str):
        self.model = model.cpu().eval()
        self.version = version

    @classmethod
    def from_checkpoint(cls, path: Path) -> LearnedScorer:
        model, meta = PreferenceModel.load(path)
        return cls(model, meta["model_version"])

    def attr_index(self, name: str) -> int:
        return self.model.attr_index(name)

    @torch.no_grad()
    def values(self, rows: list[tuple[float, float, float]], attrs: list[int]) -> list[float]:
        return self.model.value(torch.tensor(rows, dtype=torch.float32), torch.tensor(attrs)).tolist()
