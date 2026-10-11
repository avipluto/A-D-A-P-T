"""Compare a candidate scorer with production; promote only if it passes AND you ask.

    python scripts/gate.py --candidate D:/adapt-data/artifacts/models/<v>/model.pt
    python scripts/gate.py --candidate ... --promote --reason "retrained with 2026-11 feedback"

Production is production.json if it exists, else the default checkpoint (pref-v1-s0).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from adapt import config
from adapt.api.services import DEFAULT_SCORER
from adapt.learning.model import LearnedScorer
from adapt.lifecycle import feedback_data, registry
from adapt.lifecycle.gate import run_gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--production", type=Path, default=None)
    p.add_argument("--real-data", type=Path, default=None, help="exported real-preferences-*.jsonl")
    p.add_argument("--n-per-domain", type=int, default=2000)
    p.add_argument("--promote", action="store_true", help="make the candidate production if it passes")
    p.add_argument("--reason", default="")
    args = p.parse_args()

    prod_path = args.production or registry.production_scorer(DEFAULT_SCORER)
    cand, prod = LearnedScorer.from_checkpoint(args.candidate), LearnedScorer.from_checkpoint(prod_path)
    real = feedback_data.load(args.real_data) if args.real_data else None
    report = run_gate(cand, prod, n_per_domain=args.n_per_domain, real_heldout=real)
    report |= {"candidate": str(args.candidate), "production": str(prod_path),
               "created_at": datetime.now(timezone.utc).isoformat()}

    out = config.ARTIFACTS_DIR / "gate" / f"{report['candidate_version']}-vs-{report['production_version']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"candidate {report['candidate_version']}  vs  production {report['production_version']}")
    for c in report["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"\nGATE {'PASSED' if report['passed'] else 'FAILED'}  ->  {out}")

    if args.promote:
        if not report["passed"]:
            raise SystemExit("not promoted: the gate failed")
        entry = registry.promote(args.candidate, out, args.reason or "passed regression gate")
        print(f"promoted: {entry['scorer']} (previous: {entry['previous']}). Restart the API to load it.")
    elif report["passed"]:
        print("not promoted (run again with --promote to make it production)")


if __name__ == "__main__":
    main()
