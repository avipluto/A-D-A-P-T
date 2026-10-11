"""Train the from-scratch tagger on TRAINING-split texts only.

Each training scenario is rendered with several text seeds (different phrasings of the
same facts). Validation texts pick the best epoch; test splits are never touched here.
Words are randomly replaced by <unk> during training so the model learns to use context
and character trigrams, which is what it must rely on for unseen domains.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from adapt import config
from adapt.extraction.textgen import render_tagged
from adapt.synthetic import DATASET_VERSION, generate
from adapt.tagger.model import LABEL_ID, TaggerModel
from adapt.tagger.tokenizer import UNK, Vocab

log = logging.getLogger("train_tagger")


def examples(scenarios, text_seeds) -> list[tuple[str, list[int]]]:
    out = []
    for s in scenarios:
        for ts in text_seeds:
            w, t = render_tagged(s, ts)
            out.append((" ".join(w), [LABEL_ID[x] for x in t]))
    return out


def batches(data, size: int, model: TaggerModel, shuffle: bool, rng: random.Random):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = [data[j] for j in idx[i : i + size]]
        enc = model.encode([t for t, _ in chunk])
        y = torch.full(enc["ids"].shape, -100, dtype=torch.long)  # -100 = ignored (padding)
        for b, (_, labels) in enumerate(chunk):
            n = min(len(labels), y.shape[1])
            y[b, :n] = torch.tensor(labels[:n])
        yield enc, y


@torch.no_grad()
def span_f1(model: TaggerModel, data, device, size: int = 64) -> float:
    """Micro F1 over non-O labels at word level (fast proxy for span F1)."""
    model.eval()
    tp = fp = fn = 0
    for enc, y in batches(data, size, model, False, random.Random(0)):
        pred = model(**{k: v.to(device) for k, v in enc.items()}).argmax(-1).cpu()
        valid = y != -100
        p_pos, y_pos = (pred != 0) & valid, (y != 0) & valid
        tp += int(((pred == y) & y_pos).sum())
        fp += int((p_pos & ((pred != y) | ~y_pos)).sum())
        fn += int((y_pos & (pred != y)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="tagger-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--text-seeds", type=int, default=4, help="phrasings per training scenario")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--word-dropout", type=float, default=0.2)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    scenarios = list(generate(600, 0))
    train_s = [s for s in scenarios if s.truth.split == "train"]
    val_s = [s for s in scenarios if s.truth.split == "val"]
    train = examples(train_s, range(1, 1 + args.text_seeds))  # test texts use seed 0; train uses others
    val = examples(val_s, [0])
    vocab = Vocab.build([t for t, _ in train])
    model = TaggerModel(vocab, d=args.d, layers=args.layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("train texts %d | val texts %d | vocab %d | params %d | %s",
             len(train), len(val), len(vocab), model.n_params(), device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best, best_state, best_epoch = -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for enc, y in batches(train, args.batch_size, model, True, rng):
            ids = enc["ids"]
            drop = (torch.rand(ids.shape) < args.word_dropout) & (ids > 2)  # never drop pad/unk/<num>
            enc["ids"] = ids.masked_fill(drop, UNK)
            logits = model(**{k: v.to(device) for k, v in enc.items()})
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        f1 = span_f1(model, val, device)
        log.info("epoch %d loss %.4f val_label_f1 %.4f", epoch, total, f1)
        if f1 > best:
            best, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    version = f"{args.name}-s{args.seed}"
    meta = {
        "model_version": version, "dataset_version": DATASET_VERSION, "seed": args.seed,
        "train_texts": len(train), "val_texts": len(val), "text_seeds": args.text_seeds,
        "vocab_size": len(vocab), "n_params": model.n_params(), "config": model.config,
        "word_dropout": args.word_dropout, "epochs": args.epochs, "best_epoch": best_epoch,
        "val_label_f1": round(best, 4), "train_seconds": round(time.perf_counter() - t0, 1),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "torch": str(torch.__version__), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = config.ARTIFACTS_DIR / "models" / version / "model.pt"
    model.cpu().save(out, meta)
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info(json.dumps(meta, indent=2))
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
