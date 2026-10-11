"""Which model is in production, and how it got there.

`production.json` points at the scorer checkpoint the API loads. It only changes through
`promote()`, which requires a passing gate report; every change is appended to
`history.jsonl`, so a rollback is just promoting the previous checkpoint again.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from adapt import config

REGISTRY_DIR = config.ARTIFACTS_DIR / "models"
PRODUCTION = REGISTRY_DIR / "production.json"
HISTORY = REGISTRY_DIR / "history.jsonl"


def production_scorer(default: Path) -> Path:
    if PRODUCTION.exists():
        return Path(json.loads(PRODUCTION.read_text(encoding="utf-8"))["scorer"])
    return default


def promote(checkpoint: Path, gate_report: Path, reason: str, *, registry_dir: Path = REGISTRY_DIR) -> dict:
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise ValueError("refusing to promote: the gate report did not pass")
    if Path(report["candidate"]).resolve() != checkpoint.resolve():
        raise ValueError("refusing to promote: the gate report is for a different checkpoint")
    entry = {
        "scorer": str(checkpoint),
        "gate_report": str(gate_report),
        "reason": reason,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "previous": json.loads((registry_dir / "production.json").read_text(encoding="utf-8"))["scorer"]
        if (registry_dir / "production.json").exists() else None,
    }
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "production.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    with (registry_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
