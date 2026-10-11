"""Score an extractor on the synthetic text benchmark.

    python scripts/evaluate_extraction.py --extractor rules
    python scripts/evaluate_extraction.py --extractor llm --n 100     # local Qwen, slower

Only test splits are used. Results go to ADAPT_DATA_DIR/artifacts/eval/extraction.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from adapt import config
from adapt.extraction.metrics import evaluate_extractor
from adapt.synthetic import DATASET_VERSION, generate

SPLITS = ("test_seen", "test_unseen_domain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extractor", choices=("rules", "llm", "tagger"), default="rules")
    p.add_argument("--checkpoint", default=None, help="tagger model.pt for --extractor tagger")
    p.add_argument("--model", default=None, help="Hugging Face model id for --extractor llm")
    p.add_argument("--n", type=int, default=0, help="scenarios per split (0 = all)")
    p.add_argument("--seed", type=int, default=0, help="scenario dataset seed")
    p.add_argument("--text-seed", type=int, default=0)
    p.add_argument("--prompt", default="v2", help="prompt version for --extractor llm")
    p.add_argument("--splits", default=",".join(SPLITS), help="use 'val' while tuning prompts")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--from-cache", default=None, help="raw-outputs .jsonl from an earlier llm run (no GPU)")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    scenarios = list(generate(600, args.seed))
    subsets = {}
    for split in args.splits.split(","):
        subset = [s for s in scenarios if s.truth.split == split]
        if args.n:
            subset = subset[:: max(1, len(subset) // args.n)][: args.n]  # spread across domains and kinds
        subsets[split] = subset

    from adapt.extraction.postprocess import clean

    ms_per = None
    if args.extractor == "rules":
        from adapt.extraction.rules import extract as base

        base_name = "rules-v1"
        variants = {base_name: base}
    elif args.extractor == "tagger":
        from adapt.tagger.extractor import TaggerExtractor

        base = TaggerExtractor.from_checkpoint(Path(args.checkpoint))
        base_name = base.version
        variants = {base_name: base}
    else:
        from adapt.extraction.grounding import ground
        from adapt.extraction.schema import Extraction
        from adapt.extraction.textgen import render

        # The model runs ONCE; raw outputs are saved, and every variant below is scored from
        # the same generations. --from-cache rescoring needs no GPU at all.
        texts = [render(s, args.text_seed) for sub in subsets.values() for s in sub]
        cache: dict[str, object] = {}
        if args.from_cache:
            for line in Path(args.from_cache).read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                cache[rec["text"]] = Extraction.model_validate(rec["extraction"]) if rec["ok"] else ValueError(rec["error"])
            base_name = json.loads(Path(args.from_cache).read_text(encoding="utf-8").splitlines()[0])["model"]
            missing = [t for t in texts if t not in cache]
            if missing:
                raise SystemExit(f"{len(missing)} texts are not in the cache; run without --from-cache")
        else:
            from adapt.extraction.llm import DEFAULT_MODEL, LLMExtractor

            llm = LLMExtractor(args.model or DEFAULT_MODEL, prompt=args.prompt, grounding=False)
            base_name = llm.version
            t0 = time.perf_counter()
            for i in range(0, len(texts), args.batch_size):
                chunk = texts[i : i + args.batch_size]
                cache.update(zip(chunk, llm.batch(chunk)))
                print(f"  extracted {min(i + args.batch_size, len(texts))}/{len(texts)}", flush=True)
            ms_per = (time.perf_counter() - t0) * 1000 / max(1, len(texts))
            raw_path = config.ARTIFACTS_DIR / "eval" / "extraction" / f"{base_name}-raw-{'_'.join(subsets)}-n{args.n}.jsonl"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_path.open("w", encoding="utf-8") as f:
                for t, r in cache.items():
                    ok = not isinstance(r, Exception)
                    f.write(json.dumps({"model": base_name, "text": t, "ok": ok,
                                        "extraction": r.model_dump() if ok else None,
                                        "error": None if ok else str(r)[:500]}) + "\n")
            print(f"raw outputs saved: {raw_path}")

        def base(text: str):
            r = cache[text]
            if isinstance(r, Exception):
                raise r
            return r

        variants = {base_name: base, f"{base_name}+grounded": lambda t: ground(base(t), t)[0]}

    # Post-processing variants, always reported next to the raw extractor.
    variants |= {
        f"{base_name}+limits": lambda t: clean(base(t), t, limits=True, names=False)[0],
        f"{base_name}+names": lambda t: clean(base(t), t, limits=False, names=True)[0],
        f"{base_name}+clean": lambda t: clean(base(t), t)[0],
    }

    reports: dict[str, dict] = {}
    for name, extractor in variants.items():
        out_dir = config.ARTIFACTS_DIR / "eval" / "extraction" / f"{name}-{DATASET_VERSION}-seed{args.seed}-n{args.n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        reports[name] = {}
        for split, subset in subsets.items():
            rows, report = evaluate_extractor(subset, extractor, args.text_seed, args.batch_size)
            if ms_per is not None:
                report["ms_per_text"] = ms_per  # model time, measured once above
            rows.to_csv(out_dir / f"{split}.csv", index=False)
            reports[name][split] = report
            _print(name, split, report)
        meta = {"extractor": name, "dataset_version": DATASET_VERSION, "seed": args.seed, "n": args.n,
                "text_seed": args.text_seed, "batch_size": args.batch_size,
                "created_at": datetime.now(timezone.utc).isoformat()}
        (out_dir / "report.json").write_text(json.dumps({"meta": meta, **reports[name]}, indent=2))
        print(f"report: {out_dir}")


def _print(name: str, split: str, report: dict) -> None:
    print(f"\n{name} on {split} (n={report['n']})")
    for k in ("options", "values", "criteria", "constraints"):
        m = report[k]
        print(f"  {k:<12} P={_f(m['precision'])}  R={_f(m['recall'])}  F1={_f(m['f1'])}")
    for k in ("priority_accuracy", "naming_consistency", "perfect_rate", "outcome_agreement", "decision_agreement",
              "wrong_recommendation_rate", "failure_rate"):
        print(f"  {k:<26} {_f(report[k])}")
    print(f"  {'ms_per_text':<26} {report['ms_per_text']:.1f}")


def _f(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
