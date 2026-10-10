"""Score an extractor on the synthetic text benchmark.

    python scripts/evaluate_extraction.py --extractor rules
    python scripts/evaluate_extraction.py --extractor llm --n 100     # local Qwen, slower

Only test splits are used. Results go to ADAPT_DATA_DIR/artifacts/eval/extraction.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from adapt import config
from adapt.extraction.metrics import evaluate_extractor
from adapt.synthetic import DATASET_VERSION, generate

SPLITS = ("test_seen", "test_unseen_domain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extractor", choices=("rules", "llm"), default="rules")
    p.add_argument("--model", default=None, help="Hugging Face model id for --extractor llm")
    p.add_argument("--n", type=int, default=0, help="scenarios per split (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="scenario dataset seed")
    p.add_argument("--text-seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.extractor == "rules":
        from adapt.extraction.rules import extract as extractor

        name = "rules-v1"
    else:
        from adapt.extraction.llm import DEFAULT_MODEL, LLMExtractor

        extractor = LLMExtractor(args.model or DEFAULT_MODEL)
        name = (args.model or DEFAULT_MODEL).split("/")[-1]

    scenarios = list(generate(600, args.seed))
    out_dir = config.ARTIFACTS_DIR / "eval" / "extraction" / f"{name}-{DATASET_VERSION}-seed{args.seed}-n{args.n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split in SPLITS:
        subset = [s for s in scenarios if s.truth.split == split]
        if args.n:
            subset = subset[:: max(1, len(subset) // args.n)][: args.n]  # spread across domains and kinds
        rows, report = evaluate_extractor(subset, extractor, args.text_seed)
        rows.to_csv(out_dir / f"{split}.csv", index=False)
        reports[split] = report

        print(f"\n{name} on {split} (n={report['n']})")
        for k in ("options", "values", "criteria", "constraints"):
            m = report[k]
            print(f"  {k:<12} P={_f(m['precision'])}  R={_f(m['recall'])}  F1={_f(m['f1'])}")
        for k in ("priority_accuracy", "naming_consistency", "perfect_rate", "outcome_agreement", "decision_agreement",
                  "wrong_recommendation_rate", "failure_rate"):
            print(f"  {k:<26} {_f(report[k])}")
        print(f"  {'ms_per_text':<26} {report['ms_per_text']:.1f}")

    meta = {"extractor": name, "dataset_version": DATASET_VERSION, "seed": args.seed, "n": args.n,
            "text_seed": args.text_seed, "created_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / "report.json").write_text(json.dumps({"meta": meta, **reports}, indent=2))
    print(f"\nreport: {out_dir}")


def _f(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
