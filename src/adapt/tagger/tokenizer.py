"""From-scratch word tokenizer and features for the tagger.

Text is split on whitespace (the same words the labels are defined on). Each word is
represented by three things, so the model is never fully blind to an unseen word:

1. a word id from a vocabulary built on TRAINING texts only (unseen words → <unk>);
   numbers share one id (<num>), since their exact value doesn't help labelling;
2. hashed character trigrams ("<pr", "pri", "ric", ...), fastText-style, which unseen
   words still share with seen ones ("tuition" and "position" share "tio", "ion");
3. a few shape features: is it a number, capitalised, a lone capital letter (option
   labels like "A"), followed by a comma / full stop, and so on.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from functools import lru_cache

PAD, UNK, NUM = 0, 1, 2
N_BUCKETS = 4096  # hashed trigram embedding rows
MAX_TRIGRAMS = 12  # per word; longer words keep the first 12

_NUMBER = re.compile(r"^[₹$]?\d")
_STRIP = ".,:;!?"


def words(text: str) -> list[str]:
    return text.split()


def norm(word: str) -> str:
    w = word.lower().rstrip(_STRIP)
    return "<num>" if _NUMBER.match(w) else w


@lru_cache(maxsize=200_000)  # words repeat constantly; cache per word
def trigrams(word: str) -> tuple[int, ...]:
    w = f"<{word.lower().rstrip(_STRIP)}>"
    grams = [w[i : i + 3] for i in range(max(1, len(w) - 2))][:MAX_TRIGRAMS]
    return tuple(1 + zlib.crc32(g.encode()) % (N_BUCKETS - 1) for g in grams)  # 0 is padding


@lru_cache(maxsize=200_000)
def shape(word: str) -> tuple[float, ...]:
    core = word.rstrip(_STRIP)
    return (
        float(bool(_NUMBER.match(core))),
        float(core[:1].isupper()),
        float(len(core) == 1 and core.isupper()),
        float(word.endswith(",")),
        float(word.endswith(".")),
        float(word.endswith(":")),
        float("/" in core),
        float("%" in core),
        float(core[:1] in "₹$"),
        float(core.lower() in {"not", "doesn't", "can't", "non-refundable", "isn't", "no"} or core.lower().startswith("non-")),
    )


N_SHAPE = len(shape("x"))


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = ["<pad>", "<unk>", "<num>", *tokens]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2) -> Vocab:
        counts = Counter(norm(w) for t in texts for w in words(t))
        return cls(sorted(t for t, c in counts.items() if c >= min_freq and t != "<num>"))

    def __len__(self) -> int:
        return len(self.itos)

    def id(self, word: str) -> int:
        n = norm(word)
        return NUM if n == "<num>" else self.stoi.get(n, UNK)
