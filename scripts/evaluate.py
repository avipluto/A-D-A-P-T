"""Generate the synthetic benchmark and score one or more scorers on it.

    python scripts/evaluate.py                                   # baseline only
    python scripts/evaluate.py --model D:/adapt-data/artifacts/models/<version>/model.pt [...]

The baseline is always included so every learned model is compared against it.
Writes the dataset and per-scorer reports to ADAPT_DATA_DIR.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from adapt import config
from adapt.engine import ENGINE_VERSION, decide
from adapt.engine.scoring import BASELINE
from adapt.evaluation.metrics import evaluate
from adapt.synthetic import DATASET_VERSION, generate

COLUMNS = [
    ("n", "n", "{:d}"),
    ("decision_accuracy", "decision", "{:.3f}"),
    ("top1_complete", "top1", "{:.3f}"),
    ("ndcg_complete", "ndcg", "{:.3f}"),
    ("ndcg_random", "ndcg_rand", "{:.3f}"),
    ("coverage", "coverage", "{:.3f}"),
    ("error_rate_issued", "err@issued", "{:.3f}"),
    ("missed_question_rate", "missed_q", "{:.3f}"),
    ("unnecessary_question_rate", "extra_q", "{:.3f}"),
    ("constraint_violations", "violations", "{:d}"),
]
TEST_SPLITS = ("test_seen", "test_unseen_domain")


def table(title: str, groups: dict[str, dict], width: int = 22) -> str:
    head = f"{title:<{width}}" + "".join(f"{h:>11}" for _, h, _ in COLUMNS)
    lines = [head, "-" * len(head)]
    for name, m in groups.items():
        cells = "".join(f"{'-' if m[k] is None else fmt.format(m[k]):>11}" for k, _, fmt in COLUMNS)
        lines.append(f"{name:<{width}}{cells}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, action="append", default=[], help="learned model checkpoint(s)")
    p.add_argument("--n-per-domain", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detail", action="store_true", help="also print per-kind and per-domain tables")
    args = p.parse_args()

    scorers = [BASELINE]
    if args.model:
        from adapt.learning.model import LearnedScorer  # torch only needed for learned models

        scorers += [LearnedScorer.from_checkpoint(m) for m in args.model]

    config.ensure_dirs()
    scenarios = list(generate(args.n_per_domain, args.seed))
    data_path = config.EVALUATION_DIR / f"{DATASET_VERSION}-seed{args.seed}.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(s.model_dump_json() + "\n")

    reports = {}
    for scorer in scorers:
        t0 = time.perf_counter()
        rows, report = evaluate(scenarios, partial(decide, scorer=scorer))
        ms_per = (time.perf_counter() - t0) * 1000 / len(scenarios)
        reports[scorer.version] = report

        run_dir = config.ARTIFACTS_DIR / "eval" / f"{ENGINE_VERSION}+{scorer.version}-{DATASET_VERSION}-seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows.to_csv(run_dir / "rows.csv", index=False)
        meta = {
            "engine_version": ENGINE_VERSION,
            "scorer_version": scorer.version,
            "dataset_version": DATASET_VERSION,
            "seed": args.seed,
            "n_per_domain": args.n_per_domain,
            "ms_per_decision": round(ms_per, 3),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "report.json").write_text(json.dumps({"meta": meta, **report}, indent=2))
        print(f"{scorer.version}: {ms_per:.3f} ms per decision -> {run_dir}")

        if args.detail:
            print(table("kind", report["by_kind"]))
            print(table("domain", report["by_domain"]))

    for split in TEST_SPLITS:
        print()
        print(table(split, {v: r["by_split"][split] for v, r in reports.items()}, width=26))
    print(f"\ndataset: {data_path}")


if __name__ == "__main__":
    main()
