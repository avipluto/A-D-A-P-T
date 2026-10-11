# Phase 3c — from-scratch transformer tagger

Measured 2026-10-11. Model `tagger-v1-s0`, trained from random initialisation (no pretrained weights).
Reproduce:

```powershell
D:\adapt-env\Scripts\python.exe scripts/train_tagger.py --seed 0
D:\adapt-env\Scripts\python.exe scripts/evaluate_extraction.py --extractor tagger --checkpoint D:\adapt-data\artifacts\models\tagger-v1-s0\model.pt
D:\adapt-env\Scripts\python.exe scripts/oracle_tagger.py --splits val     # ceiling with perfect labels
```

**Synthetic, template-generated text only.** See phase 3a for what that does and doesn't show.

## Model

- **Task:** label every word (31 BIO labels: option, attribute, value, yes/no fact, criterion with direction,
  priority, limit, limit value). Fixed rules turn the labels into the shared `Extraction` format.
- **Tokenizer, from scratch:** whitespace words, a 169-word vocabulary built from training texts only, hashed
  character trigrams (so unseen words share pieces with seen ones), and 10 shape features.
- **Network:** 4-layer transformer encoder, d = 128, 4 heads, 1,130,655 parameters.
- **Data:** 3,083 training scenarios × 4 phrasings = 12,332 texts; words randomly replaced by `<unk>` 20% of the
  time; best epoch picked on 460 validation texts.
- **Cost:** 12 epochs in 737 s on the RTX 2050, peak 331 MiB GPU memory; ~14 ms per text at inference on CPU.
- Validation label F1: 0.9999 (familiar domains only).

## Ceiling: perfect labels, same decoding (validation)

| | Decision agreement | Naming consistency |
|---|---|---|
| Gold labels, raw | 0.404 | 0.137 |
| Gold labels + clean-up | **0.854** | 0.791 |

Even perfect labelling can't fix synonyms ("salary" / "pay" / "annual package"): a tagger copies words.
Keeping the unit with each value and resolving stray criterion names lifted the ceiling from ~0.6 to 0.85.

## Test results (all test texts)

| | Seen: values F1 | Seen: decision agr. | Seen: wrong rec. | Unseen: values F1 | Unseen: decision agr. | Unseen: wrong rec. |
|---|---|---|---|---|---|---|
| Tagger raw | 0.985 | 0.387 | 0.006 | 0.584 | 0.278 | **0.014** |
| Tagger + clean-up | 0.990 | **0.887** | **0.003** | 0.574 | 0.322 | 0.068 |
| Rules (for reference) | 1.000 | 0.997 | 0.003 | 0.226 | 0.332 | 0.204 |

## Same 48 + 48 texts as Qwen (phase 3b)

| | Seen: decision agr. | Seen: wrong rec. | Unseen: decision agr. | Unseen: wrong rec. | Time per text |
|---|---|---|---|---|---|
| Rules | 0.979 | 0.021 | 0.292 | 0.250 | ~5 ms |
| Qwen2.5-1.5B (raw) | 0.625 | 0.083 | 0.375 | 0.188 | 11–24 s |
| **Tagger raw** | 0.417 | **0.000** | 0.271 | **0.021** | ~14 ms |
| **Tagger + clean-up** | **0.917** | **0.000** | 0.271 | 0.104 | ~20 ms |

n = 48 per split, so treat differences under ~10 points as noise.

## Findings

1. **On familiar domains the tagger plus clean-up beats Qwen clearly:** 0.89 decision agreement (0.92 on the
   shared subset vs. Qwen's 0.63), with 0.3% wrong recommendations, about 1,000× faster and ~1,300× smaller.
   The hand-written rules are still better there (0.997), but they were written against the same templates.
2. **On unseen domains no extractor works well.** The tagger finds more values than rules (F1 0.58 vs. 0.23) but
   fewer than Qwen (0.80). Criteria are its weak spot (precision 0.31): with unfamiliar words it mislabels words
   as criteria.
3. **The raw tagger is the safest extractor on unseen domains:** 1.4% wrong recommendations, against ~19–25% for
   rules and Qwen. Because it copies words, mismatched names make the engine see missing information and ask
   instead of recommending.
4. **The clean-up helps on familiar domains and hurts on unseen ones.** It raises seen-domain agreement from 0.39
   to 0.89, but on unseen domains it merges names that shouldn't be merged, and wrong recommendations rise from
   1.4% to 6.8%. It should only be applied where it has been validated.
5. **No extractor meets the bar for automatic use on unseen domains** (≤ 2% wrong recommendations *and* useful
   agreement). The raw tagger is safe there but mostly ends up asking.

## Verdict (rule from the discussion: keep a translator only if it makes ADAPT better)

| Use | Verdict |
|---|---|
| Form input (no translator) | **Main path.** Always correct by construction. |
| Tagger + clean-up as "pre-fill the form" in familiar domains | **Keep.** Fast, small, 0.3% wrong recommendations, and the user confirms. |
| Tagger raw on unfamiliar domains | Optional pre-fill only; it rarely misleads but often leaves gaps. |
| Qwen2.5-1.5B | **Drop as the default.** Slower, invents limits, and loses to the tagger on familiar domains. Keep as a research comparison. |
| Rules | Baseline only: they only work on text written in exactly their templates. |
