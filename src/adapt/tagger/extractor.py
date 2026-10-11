"""The from-scratch tagger as an extractor: text → labels → Extraction."""

from __future__ import annotations

from pathlib import Path

import torch

from adapt.extraction.schema import Extraction
from adapt.tagger.decode import assemble
from adapt.tagger.model import LABELS, TaggerModel
from adapt.tagger.tokenizer import words


class TaggerExtractor:
    def __init__(self, model: TaggerModel, version: str, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.version = version
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> TaggerExtractor:
        model, meta = TaggerModel.load(path)
        return cls(model, meta["model_version"], device)

    @torch.no_grad()
    def labels(self, texts: list[str]) -> list[list[str]]:
        batch = {k: v.to(self.device) for k, v in self.model.encode(texts).items()}
        pred = self.model(**batch).argmax(-1).cpu()
        return [[LABELS[i] for i in pred[b, : len(words(t)[: self.model.max_len])]] for b, t in enumerate(texts)]

    def batch(self, texts: list[str]) -> list[Extraction | Exception]:
        return [assemble(words(t), labs) for t, labs in zip(texts, self.labels(texts))]

    def __call__(self, text: str) -> Extraction:
        return self.batch([text])[0]
