# Phase 3b — local LLM extractor (Qwen2.5-1.5B-Instruct)

Measured 2026-10-10. Model `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 2.88 GiB, bf16 on the RTX 2050),
prompt `v2`, greedy decoding, batches of 8. Test splits, 48 texts each (spread across domains and kinds);
the rule-based baseline is scored on the same 48 + 48 texts. Reproduce:

```powershell
D:\adapt-env\Scripts\python.exe scripts/evaluate_extraction.py --extractor llm --prompt v2 --n 48
D:\adapt-env\Scripts\python.exe scripts/evaluate_extraction.py --extractor rules --n 48
```

**Small samples:** with n = 48, one scenario is ~2 points. Differences under ~10 points are not reliable.
**Synthetic text:** template-generated, so cleaner than real messages (see phase 3a).

## Results

| Metric | Rules, seen | Qwen, seen | Rules, unseen | Qwen, unseen |
|---|---|---|---|---|
| Values F1 | 1.000 | 0.903 | 0.232 | **0.801** |
| Criteria F1 | 1.000 | 0.854 | 0.322 | **0.822** |
| Hard limits F1 | 0.978 | 0.576 | 0.176 | 0.575 |
| Hard limits precision | 1.000 | **0.442** | 1.000 | **0.446** |
| Priority accuracy | 1.000 | 0.701 | 1.000 | 0.620 |
| Naming consistency | 1.000 | **0.562** | 1.000 | **0.604** |
| Failed (invalid output) | 0 | 0.042 | 0 | 0.062 |
| **Decision agreement** | **0.979** | 0.625 | 0.292 | **0.375** |
| **Wrong recommendation** | 0.021 | 0.083 | 0.250 | 0.188 |
| Time per text | ~5 ms | ~11–24 s | ~5 ms | ~11–24 s |

With the grounding check (drop numbers not in the text): values precision 0.951 → 0.980 on seen, otherwise
no measurable change (decision agreement 0.604 / 0.354).

## What happened along the way (validation split only, n = 24)

| Change | Failed | Decision agreement |
|---|---|---|
| Prompt v1 | 0.292 | 0.417 |
| Prompt v2 (no invented limits, priority only if stated, one name per attribute) | 0.250 | 0.458 |
| + accept paraphrased priority/direction words; compact JSON; 1,200-token cap | **0.000** | 0.500 |

Failures were schema errors ("priority": "nice_to_have", "top") and JSON truncated at the 700-token cap by
indented output. Speed: one text at a time ran at ~8 tokens/s with the GPU only ~23% busy (CPU-bound
generation); batching 8 texts brought it to ~11–15 s per text.

## Findings

1. **Qwen reads unfamiliar vocabulary far better than rules** (values F1 0.80 vs 0.23 on unseen domains),
   but **end-to-end it barely helps**: decision agreement 0.375 vs 0.292, wrong recommendations 0.19 vs 0.25.
2. **Three error types cancel out the reading ability:**
   - **Made-up hard limits:** more than half of reported limits weren't stated (precision ~0.44). Usually it
     turns a preference ("lower price is better") into a limit using a number from the text, so the grounding
     check can't catch it. A wrong hard limit silently excludes options.
   - **Two names for one attribute** in ~40% of frames ("stops" / "layovers"), which makes the engine think a
     value is missing.
   - **Wrong priorities** about a third of the time.
3. **A 1.5B local model is not reliable enough to decide on its own.** Its output must be shown to the user
   to confirm and correct before the engine runs (already planned for the frontend).
4. **Latency (~11–24 s per text on this GPU)** makes it a background or "fill the form for me" step, not
   something to wait on in an interactive loop.

## Next options

- **Cheap, deterministic fixes** on top of the LLM output (no training): keep a hard limit only if its sentence
  contains limit language ("can't", "must", "at most", "at least", "no more than"); merge attribute names that
  refer to the same values across options. Tune on val, measure on test.
- **Fine-tune a small model on the training-domain texts** (e.g. LoRA on Qwen2.5-0.5B), then test on the unseen
  domains. Training cost: tens of minutes on the RTX 2050.
- **Phase 3c, from-scratch transformer:** a token tagger rather than a JSON generator, which suits a small model.
