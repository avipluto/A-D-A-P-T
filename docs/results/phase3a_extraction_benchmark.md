# Phase 3a — extraction benchmark and rule-based baseline

Measured 2026-10-10 on `synthetic-v1` seed 0, test splits only (657 seen-domain, 1,200
unseen-domain scenarios). Reproduce: `python scripts/evaluate_extraction.py --extractor rules`.

## The benchmark

Each synthetic scenario is rendered as text (`src/adapt/extraction/textgen.py`): the options and their
values, what's better (more or less), how much it matters (priority words → high / medium / low), and
hard limits. The same attribute is named in different ways within one text ("salary", "pay",
"annual package"). The extractor has to recover the gold frame.

**Caveat:** the text is template-generated. Real descriptions are messier (hedging, typos, implied values,
values stated relative to each other). Synthetic scores here are an upper bound; a hand-written test set is
still needed.

## Metrics

- **Field level:** precision / recall / F1 for options, (option, attribute, value) triples, (attribute, direction)
  criteria and hard limits; accuracy of priority levels.
- **Naming consistency:** each attribute must get ONE name throughout the frame. Added after finding that
  per-field matching scored a frame "perfect" even when it used two names for one attribute, which made the
  engine see a missing value.
- **End to end:** run `decide()` on the extracted frame and on the gold frame. *Decision agreement* = same
  outcome and same recommended option. *Wrong recommendation* = the extracted frame leads to a recommendation
  the gold frame wouldn't make. This is the failure that matters most.

## Rule-based baseline (`rules-v1`)

Regular expressions plus an attribute vocabulary taken only from the training domains. It was written against
the same templates, so seen-domain scores are optimistic.

| Metric | test_seen | test_unseen_domain |
|---|---|---|
| Options F1 | 1.000 | 1.000 |
| Values F1 | 1.000 | 0.226 |
| Criteria F1 | 1.000 | 0.308 |
| Hard limits F1 | 0.997 | 0.212 |
| Priority accuracy (matched criteria) | 1.000 | 1.000 |
| Perfect frames | 0.997 | 0.000 |
| **Decision agreement** | **0.997** | **0.332** |
| **Wrong recommendation rate** | 0.003 | **0.204** |
| Time per text | ~5 ms | ~5 ms |

## Findings

1. **Rules don't transfer.** With unfamiliar vocabulary, recall drops to 13–18%, and only a third of decisions
   match. This is the gap the learned extractors (Qwen, from-scratch transformer) must close.
2. **Partial extraction is worse than none.** On unseen domains, 20% of texts led to a confident *wrong*
   recommendation: the engine decided on the subset of criteria it was given and had no way to know others
   existed. Safeguards needed before any extractor is used for real:
   - the user confirms the extracted frame before deciding (planned in the frontend), and
   - a coverage check: attributes mentioned in the text but missing from the criteria should trigger a question.
3. **Bugs found by the end-to-end metric:** a text-cleanup regex that rewrote "Investment A I'd…" into
   "Investment An I'd…", a label regex that captured "With", and ambiguous phrases ("monthly cost") filed under
   two names. All fixed; the naming-consistency metric now catches the last kind.
