"""Run the two trained models with NumPy only, for deployments without PyTorch.

Weights come from `scripts/export_numpy.py`: an .npz of float32 arrays (loaded with
allow_pickle=False, so a weights file can't run code) plus a .json with vocabulary and
configuration. Training stays in PyTorch; tests check that these produce the same
outputs as the PyTorch models.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from adapt.extraction.schema import Extraction
from adapt.learning.additive import AdditiveScorer
from adapt.tagger.decode import assemble
from adapt.tagger.tokenizer import MAX_TRIGRAMS, N_SHAPE, Vocab, shape, trigrams, words

_erf = np.frompyfunc(math.erf, 1, 1)


def _gelu(x: np.ndarray) -> np.ndarray:  # exact GELU, as torch.nn.GELU() defaults to
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float64))


def _linear(x: np.ndarray, w: dict, name: str) -> np.ndarray:
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def _layer_norm(x: np.ndarray, w: dict, name: str, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w[f"{name}.weight"] + w[f"{name}.bias"]


def load_weights(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """path: the .npz; the .json with the same stem holds vocabulary and config."""
    with np.load(path, allow_pickle=False) as z:
        weights = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return weights, meta


class NumpyScorer(AdditiveScorer):
    def __init__(self, weights: dict[str, np.ndarray], meta: dict):
        self.w = {k: v.astype(np.float64) for k, v in weights.items()}
        self.vocab = meta["vocab"]
        self.use_attr = meta["config"]["use_attr"]
        self._index = {n: i + 1 for i, n in enumerate(self.vocab)}
        self.version = meta["meta"]["model_version"]

    @classmethod
    def from_file(cls, path: Path) -> NumpyScorer:
        return cls(*load_weights(path))

    def attr_index(self, name: str) -> int:
        return self._index.get(name, 0) if self.use_attr else 0

    def values(self, rows: list[tuple[float, float, float]], attrs: list[int]) -> list[float]:
        x = np.concatenate([np.asarray(rows, dtype=np.float64), self.w["emb.weight"][np.asarray(attrs)]], axis=1)
        h = _gelu(_linear(x, self.w, "mlp.0"))
        h = _gelu(_linear(h, self.w, "mlp.2"))
        z = _linear(h, self.w, "mlp.4")[:, 0]
        return (1.0 / (1.0 + np.exp(-z))).tolist()


class NumpyTagger:
    """Same interface as tagger.extractor.TaggerExtractor (call with text, or .batch)."""

    def __init__(self, weights: dict[str, np.ndarray], meta: dict):
        self.w = {k: v.astype(np.float32) for k, v in weights.items()}
        self.vocab = Vocab(meta["vocab"])
        self.labels_list = meta["labels"]
        cfg = meta["config"]
        self.heads, self.layers, self.max_len = cfg["heads"], cfg["layers"], cfg["max_len"]
        self.version = meta["meta"]["model_version"]

    @classmethod
    def from_file(cls, path: Path) -> NumpyTagger:
        return cls(*load_weights(path))

    def _encode(self, texts: list[str]):
        ws = [words(t)[: self.max_len] for t in texts]
        n = max(1, max(len(w) for w in ws))
        pad_tri, pad_shape = (0,) * MAX_TRIGRAMS, (0.0,) * N_SHAPE
        ids = np.array([[self.vocab.id(x) for x in w] + [0] * (n - len(w)) for w in ws], dtype=np.int64)
        tri = np.array([[(trigrams(x) + pad_tri)[:MAX_TRIGRAMS] for x in w] + [pad_tri] * (n - len(w)) for w in ws],
                       dtype=np.int64)
        shp = np.array([[shape(x) for x in w] + [pad_shape] * (n - len(w)) for w in ws], dtype=np.float32)
        return ids, tri, shp, ws

    def _attention(self, x: np.ndarray, pad: np.ndarray, i: int) -> np.ndarray:
        w = self.w
        b, n, d = x.shape
        h, hd = self.heads, d // self.heads
        qkv = x @ w[f"encoder.layers.{i}.self_attn.in_proj_weight"].T + w[f"encoder.layers.{i}.self_attn.in_proj_bias"]
        q, k, v = (t.reshape(b, n, h, hd).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, axis=-1))
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
        att = np.where(pad[:, None, None, :], -np.inf, att)
        att = np.exp(att - att.max(-1, keepdims=True))
        att = att / att.sum(-1, keepdims=True)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        return _linear(out, w, f"encoder.layers.{i}.self_attn.out_proj")

    def logits(self, texts: list[str]) -> tuple[np.ndarray, list[list[str]]]:
        w = self.w
        ids, tri, shp, ws = self._encode(texts)
        t = w["tri.weight"][tri]
        tri_mean = t.sum(2) / np.maximum((tri != 0).sum(2, keepdims=True), 1)
        x = w["word.weight"][ids] + tri_mean + _linear(shp, w, "shape") + w["pos.weight"][: ids.shape[1]]
        x = _layer_norm(x, w, "norm")
        pad = ids == 0
        for i in range(self.layers):  # norm_first encoder layer, ReLU feed-forward
            x = x + self._attention(_layer_norm(x, w, f"encoder.layers.{i}.norm1"), pad, i)
            y = _layer_norm(x, w, f"encoder.layers.{i}.norm2")
            x = x + _linear(np.maximum(_linear(y, w, f"encoder.layers.{i}.linear1"), 0), w, f"encoder.layers.{i}.linear2")
        return _linear(x, w, "head"), ws

    def labels(self, texts: list[str]) -> list[list[str]]:
        z, ws = self.logits(texts)
        pred = z.argmax(-1)
        return [[self.labels_list[j] for j in pred[b, : len(ws[b])]] for b in range(len(texts))]

    def batch(self, texts: list[str]) -> list[Extraction | Exception]:
        return [assemble(words(t), labs) for t, labs in zip(texts, self.labels(texts))]

    def __call__(self, text: str) -> Extraction:
        return self.batch([text])[0]
