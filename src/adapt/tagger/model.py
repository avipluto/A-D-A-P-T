"""A small transformer encoder trained from random initialisation to label words.

    word id emb + mean(trigram embs) + shape features + position  →  N encoder layers  →  label per word

No pretrained weights anywhere. Sized for the RTX 2050 (4 GiB); trains in minutes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from adapt.extraction.textgen import TAGS
from adapt.tagger.tokenizer import MAX_TRIGRAMS, N_BUCKETS, N_SHAPE, Vocab, shape, trigrams, words

LABELS = ["O", *(f"{b}-{t}" for t in TAGS for b in "BI")]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}


class TaggerModel(nn.Module):
    def __init__(self, vocab: Vocab, d: int = 128, layers: int = 4, heads: int = 4, ff: int = 256,
                 dropout: float = 0.1, max_len: int = 384):
        super().__init__()
        self.vocab = vocab
        self.config = {"d": d, "layers": layers, "heads": heads, "ff": ff, "dropout": dropout, "max_len": max_len}
        self.word = nn.Embedding(len(vocab), d, padding_idx=0)
        self.tri = nn.Embedding(N_BUCKETS, d, padding_idx=0)
        self.shape = nn.Linear(N_SHAPE, d)
        self.pos = nn.Embedding(max_len, d)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, len(LABELS))
        self.max_len = max_len

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenise a batch into padded tensors (on CPU)."""
        ws = [words(t)[: self.max_len] for t in texts]
        n = max(1, max(len(w) for w in ws))
        pad_tri = (0,) * MAX_TRIGRAMS
        pad_shape = (0.0,) * N_SHAPE
        # build plain nested lists, then one tensor each (per-word tensors were very slow)
        ids = [[self.vocab.id(x) for x in w] + [0] * (n - len(w)) for w in ws]
        tri = [[(trigrams(x) + pad_tri)[:MAX_TRIGRAMS] for x in w] + [pad_tri] * (n - len(w)) for w in ws]
        shp = [[shape(x) for x in w] + [pad_shape] * (n - len(w)) for w in ws]
        ids_t = torch.tensor(ids, dtype=torch.long)
        return {"ids": ids_t, "tri": torch.tensor(tri, dtype=torch.long), "shape": torch.tensor(shp),
                "mask": ids_t != 0}

    def forward(self, ids: torch.Tensor, tri: torch.Tensor, shape: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t = self.tri(tri)
        tri_mean = t.sum(2) / (tri != 0).sum(2, keepdim=True).clamp(min=1)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.word(ids) + tri_mean + self.shape(shape) + self.pos(pos)
        x = self.encoder(self.norm(x), src_key_padding_mask=~mask)
        return self.head(x)

    def save(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "vocab": self.vocab.itos[3:], "config": self.config,
                    "labels": LABELS, "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> tuple[TaggerModel, dict]:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["labels"] != LABELS:
            raise ValueError("checkpoint labels don't match this code's label set")
        m = cls(Vocab(ck["vocab"]), **ck["config"])
        m.load_state_dict(ck["state_dict"])
        return m.eval(), ck["meta"]
