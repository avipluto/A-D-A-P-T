"""Reviewed feedback → a versioned training dataset of real preferences.

Only feedback with review_status = 'accepted' is used. A decision becomes training data
when the person said which option they chose:

    chosen option ≻ every other option that met all their hard limits

Weighting keeps "liked it" and "it worked" apart:
- outcome rating 4-5 → weight 1.0 (the choice turned out well)
- outcome rating 3   → weight 0.5
- outcome rating 1-2 → skipped: regretting a choice doesn't tell us which option was right
- no outcome yet: satisfaction 4-5 → weight 0.5, otherwise skipped

Each decision is assigned to train or held-out by a hash of its id, so the gate can test
on real decisions the candidate never trained on.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from adapt.engine.constraints import Status, check
from adapt.schema import DecisionFrame


@dataclass
class RealExample:
    decision_id: str
    frame: DecisionFrame
    chosen: str
    weight: float
    split: str  # 'train' | 'heldout'
    feedback_ids: list[str]


def weight_for(outcome: int | None, satisfaction: int | None) -> float:
    if outcome is not None:
        return 1.0 if outcome >= 4 else 0.5 if outcome == 3 else 0.0
    if satisfaction is not None and satisfaction >= 4:
        return 0.5
    return 0.0


def split_for(decision_id: str, heldout_pct: int = 20) -> str:
    return "heldout" if int(hashlib.sha256(decision_id.encode()).hexdigest(), 16) % 100 < heldout_pct else "train"


def feasible(frame: DecisionFrame) -> list[str]:
    return [o.id for o in frame.options
            if all(check(o, c).status is Status.PASS for c in frame.constraints)]


def collect(db_path: Path) -> list[RealExample]:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT f.*, d.frame_json, d.status AS d_status FROM feedback f JOIN decisions d ON d.id = f.decision_id "
        "WHERE f.review_status = 'accepted' AND d.status = 'decided' ORDER BY f.created_at"
    ).fetchall()
    c.close()

    by_decision: dict[str, dict] = {}
    for r in rows:
        d = by_decision.setdefault(r["decision_id"], {"frame": r["frame_json"], "chosen": None, "outcome": None,
                                                     "satisfaction": None, "ids": []})
        d["ids"].append(r["id"])
        if r["chosen_option_id"]:
            d["chosen"] = r["chosen_option_id"]  # latest statement of the choice wins
        if r["kind"] == "outcome":
            d["outcome"] = r["rating"]
        elif r["kind"] == "satisfaction":
            d["satisfaction"] = r["rating"]

    out = []
    for did, d in by_decision.items():
        w = weight_for(d["outcome"], d["satisfaction"])
        frame = DecisionFrame.model_validate_json(d["frame"])
        if not d["chosen"] or w == 0 or d["chosen"] not in feasible(frame) or len(feasible(frame)) < 2:
            continue
        out.append(RealExample(did, frame, d["chosen"], w, split_for(did), d["ids"]))
    return out


def export(db_path: Path, out_dir: Path) -> Path:
    """Write examples + manifest; the file name carries a content fingerprint."""
    examples = collect(db_path)
    lines = [json.dumps({"decision_id": e.decision_id, "frame": e.frame.model_dump(mode="json"), "chosen": e.chosen,
                         "weight": e.weight, "split": e.split, "feedback_ids": e.feedback_ids}, sort_keys=True)
             for e in examples]
    body = "\n".join(lines) + ("\n" if lines else "")
    digest = hashlib.sha256(body.encode()).hexdigest()[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"real-preferences-{digest}.jsonl"
    path.write_text(body, encoding="utf-8")
    manifest = {
        "dataset": path.name, "sha256_12": digest, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db_path), "examples": len(examples),
        "train": sum(e.split == "train" for e in examples), "heldout": sum(e.split == "heldout" for e in examples),
        "feedback_ids": sorted(i for e in examples for i in e.feedback_ids),
        "rules": "chosen > other feasible options; weight by outcome, else satisfaction>=4 at 0.5; accepted only",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load(path: Path) -> list[RealExample]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out.append(RealExample(d["decision_id"], DecisionFrame.model_validate(d["frame"]), d["chosen"], d["weight"],
                               d["split"], d["feedback_ids"]))
    return out
