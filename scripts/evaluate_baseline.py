"""Generate the synthetic benchmark and score the baseline engine on it.

Writes the dataset and the results to ADAPT_DATA_DIR so runs are reproducible
and nothing large lands in the OneDrive-synced repo.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from adapt import config
from adapt.engine import ENGINE_VERSION, decide
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


def table(title: str, groups: dict[str, dict]) -> str:
    head = f"{title:<22}" + "".join(f"{h:>11}" for _, h, _ in COLUMNS)
    lines = [head, "-" * len(head)]
    for name, m in groups.items():
        cells = "".join(
            f"{'-' if m[k] is None else fmt.format(m[k]):>11}" for k, _, fmt in COLUMNS
        )
        lines.append(f"{name:<22}{cells}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-domain", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config.ensure_dirs()
    scenarios = list(generate(args.n_per_domain, args.seed))

    data_path = config.EVALUATION_DIR / f"{DATASET_VERSION}-seed{args.seed}.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(s.model_dump_json() + "\n")

    t0 = time.perf_counter()
    rows, report = evaluate(scenarios, decide)
    ms_per = (time.perf_counter() - t0) * 1000 / len(scenarios)

    run_dir = config.ARTIFACTS_DIR / "eval" / f"{ENGINE_VERSION}-{DATASET_VERSION}-seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "rows.csv", index=False)
    meta = {
        "engine_version": ENGINE_VERSION,
        "dataset_version": DATASET_VERSION,
        "seed": args.seed,
        "n_per_domain": args.n_per_domain,
        "ms_per_decision_incl_scoring": round(ms_per, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "report.json").write_text(json.dumps({"meta": meta, **report}, indent=2))

    print(table("split", {"overall": report["overall"], **report["by_split"]}))
    print()
    print(table("kind", report["by_kind"]))
    print()
    print(table("domain", report["by_domain"]))
    print(f"\n{ms_per:.3f} ms per scenario | dataset: {data_path} | report: {run_dir}")


if __name__ == "__main__":
    main()
