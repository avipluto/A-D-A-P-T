"""Train the learned preference model on the synthetic train split.

Uses only `train` for fitting and `val` for early stopping. Test splits are left
for scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import torch

from adapt import config
from adapt.learning.data import build_pairs, vocabulary
from adapt.learning.model import PreferenceModel
from adapt.learning.train import TrainConfig, train
from adapt.synthetic import DATASET_VERSION, generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pref-v1")
    p.add_argument("--no-attr", action="store_true", help="ignore attribute names (generic shapes only)")
    p.add_argument("--n-per-domain", type=int, default=600)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scenarios = list(generate(args.n_per_domain, args.data_seed))
    train_s = [s for s in scenarios if s.truth.split == "train"]
    val_s = [s for s in scenarios if s.truth.split == "val"]

    torch.manual_seed(args.seed)
    model = PreferenceModel(vocabulary(train_s), use_attr=not args.no_attr)
    train_d = build_pairs(train_s, model.attr_index)
    val_d = build_pairs(val_s, model.attr_index)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(
        "train: %d scenarios, %d pairs | val: %d scenarios, %d pairs | %d params | %s",
        train_d.n_scenarios, len(train_d.pairs), val_d.n_scenarios, len(val_d.pairs), model.n_params(), device,
    )

    stats = train(model, train_d, val_d, TrainConfig(epochs=args.epochs, seed=args.seed), device)
    version = f"{args.name}{'-noattr' if args.no_attr else ''}-s{args.seed}"
    meta = {
        "model_version": version,
        "dataset_version": DATASET_VERSION,
        "data_seed": args.data_seed,
        "n_per_domain": args.n_per_domain,
        "seed": args.seed,
        "use_attr": not args.no_attr,
        "vocab_size": len(model.vocab),
        "n_params": model.n_params(),
        "train_scenarios": train_d.n_scenarios,
        "train_pairs": len(train_d.pairs),
        "torch": str(torch.__version__),  # plain str so weights_only loading accepts it
        "created_at": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    out = config.ARTIFACTS_DIR / "models" / version / "model.pt"
    model.cpu().save(out, meta)
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    logging.info(json.dumps(meta, indent=2))
    logging.info("saved %s", out)


if __name__ == "__main__":
    main()
