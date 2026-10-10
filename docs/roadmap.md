# Roadmap

Each phase ends with tests and a written evaluation. No metric is reported unless it was measured.

| Phase | Deliverable | Acceptance |
|---|---|---|
| 0 | Environment check, repo, decision-frame schema, docs | Schema tests pass; CUDA visible to torch in `D:\adapt-env` |
| 1 | Baseline engine: constraint filter, weighted scoring, recommend/clarify/abstain rules; synthetic scenario generator with hidden utility functions across ≥8 domains; metrics | 0 hard-constraint violations; correct arithmetic; baseline numbers recorded on unseen scenario families |
| 2 | Learned pairwise preference model; sensitivity analysis; value-of-information questions | Beats baseline on held-out families (NDCG, top-1 agreement), or the result is reported as a negative finding |
| 3 | Extraction: pretrained baseline vs. from-scratch transformer on a hand-labelled set (~150 scenarios) | Field-level precision/recall per provenance type; abstains on ambiguous inputs |
| 4 | Retrieval with citations, dates, freshness and source checks | No retrieved value without a source; stale evidence flagged |
| 5 | FastAPI (`/health`, `/model-info`, `/decision`, `/feedback`), SQLite feedback store, frontend | Contract tests; feedback is never trained on without review |
| 6 | Offline retraining + regression gate | New version must match or beat the old one on the frozen benchmark |

## Evaluation data rules

- Synthetic data tests mechanics, not real-world generalisation.
- Splits are by **scenario family**, not by row, to prevent leakage.
- Held-out sets include changed numbers, unseen domain combinations and cases where asking beats guessing.
- High-stakes domains (medical, legal, large financial) take a gated "analysis only, consult a professional" path.
