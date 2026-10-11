"""Bradley–Terry pairwise training for the PreferenceModel.

    P(a preferred over b) = sigmoid(scale · (score_a − score_b))

Full-batch: the whole training set fits easily in 4 GiB. Early stopping uses
pairwise accuracy on the validation split; test splits are never touched here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from adapt.learning.data import PairData
from adapt.learning.model import PreferenceModel

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 3e-3
    weight_decay: float = 1e-4
    attr_dropout: float = 0.3  # chance a name is hidden, to learn a generic shape
    patience: int = 400  # val accuracy dips early before improving; short patience stopped at epoch ~4
    seed: int = 0


def option_scores(model: PreferenceModel, d: PairData, attr: torch.Tensor | None = None) -> torch.Tensor:
    g = model.value(d.feats, d.attr if attr is None else attr)
    return torch.zeros(d.n_options, device=g.device).index_add(0, d.row_option, d.weight * g)


@torch.no_grad()
def pair_accuracy(model: PreferenceModel, d: PairData) -> float:
    model.eval()
    s = option_scores(model, d)
    return float((s[d.pairs[:, 0]] > s[d.pairs[:, 1]]).float().mean())


def train(model: PreferenceModel, train_d: PairData, val_d: PairData, cfg: TrainConfig, device: torch.device) -> dict:
    torch.manual_seed(cfg.seed)
    model.to(device)
    train_d, val_d = train_d.to(device), val_d.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc, best_state, best_epoch, stale = -1.0, None, 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        attr = train_d.attr
        if model.use_attr and cfg.attr_dropout > 0:
            attr = attr.masked_fill(torch.rand_like(attr, dtype=torch.float) < cfg.attr_dropout, 0)
        s = option_scores(model, train_d, attr)
        diff = model.log_scale.exp() * (s[train_d.pairs[:, 0]] - s[train_d.pairs[:, 1]])
        nll = F.softplus(-diff)  # = −log sigmoid(diff)
        pw = train_d.pair_weight
        loss = nll.mean() if pw is None else (nll * pw).sum() / pw.sum()  # real feedback pairs can be weighted
        opt.zero_grad()
        loss.backward()
        opt.step()

        acc = pair_accuracy(model, val_d)
        if acc > best_acc + 1e-5:
            best_acc, best_epoch, stale = acc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 50 == 0:
            log.info("epoch %d loss %.4f val_pair_acc %.4f", epoch, loss.item(), acc)
        if stale >= cfg.patience:
            break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    return {
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "val_pair_accuracy": round(best_acc, 4),
        "train_pair_accuracy": round(pair_accuracy(model, train_d), 4),
        "train_seconds": round(time.perf_counter() - t0, 2),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "device": str(device),
    }
