"""Ceiling for the tagger: decode the CORRECT labels and score the result.

Separates "the decoding rules lose information" from "the model labels words wrongly".
Use --split val while developing decoding or post-processing; test splits only for reporting.
"""

from __future__ import annotations

import argparse

from adapt.extraction.metrics import evaluate_extractor
from adapt.extraction.postprocess import clean
from adapt.extraction.textgen import render_tagged
from adapt.synthetic import generate
from adapt.tagger.decode import assemble


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="val")
    args = p.parse_args()
    scenarios = list(generate(600, 0))
    gold = {}
    for s in scenarios:
        w, t = render_tagged(s)
        gold[" ".join(w)] = (w, t)

    def oracle(text: str):
        return assemble(*gold[text])

    for split in args.splits.split(","):
        subset = [s for s in scenarios if s.truth.split == split]
        for name, ex in (("oracle", oracle), ("oracle+clean", lambda t: clean(oracle(t), t)[0])):
            _, r = evaluate_extractor(subset, ex)
            print(f"{name:<13} {split:<19} values F1={r['values']['f1']:.3f} criteria F1={r['criteria']['f1']:.3f} "
                  f"limits F1={r['constraints']['f1']:.3f} naming={r['naming_consistency']:.3f} "
                  f"decision_agree={r['decision_agreement']:.3f} wrong_rec={r['wrong_recommendation_rate']:.3f}")


if __name__ == "__main__":
    main()
