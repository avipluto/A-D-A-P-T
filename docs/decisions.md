# Decision log

## D1 — Text extraction: pretrained main path, from-scratch transformer as research track (2026-10-10)

Converting free text to a decision frame across domains is a language-understanding
problem. A from-scratch transformer that fits a 4 GiB GPU (~10–30M parameters) is
unlikely to do it reliably. The system uses a pretrained extractor so it works
end to end; the from-scratch model is trained and scored on the same extraction
benchmark and replaces it only if it wins.

## D2 — Code in OneDrive, data and environment on D: (2026-10-10)

Code: `C:\Users\aviral yadav\OneDrive\Desktop\adapt` (synced, small).
Virtual env: `D:\adapt-env`. Data, checkpoints, feedback DB: `D:\adapt-data`
(`ADAPT_DATA_DIR`). Reasons: OneDrive sync churn and file locks on large files,
a space in the user path, and C: has only ~18 GiB free versus ~217 GiB on D:.

## D3 — Frontend: React + TypeScript (Vite), after the API is stable (2026-10-10)

The UI shows ADAPT's reasoning, not just an answer:
- an editable decision frame (options, criteria weights, constraints) so users correct extraction directly
- ranking with per-criterion contributions and excluded options with the constraint that excluded them
- sensitivity sliders ("move the cost weight and watch the ranking flip")
- clarifying questions as interactive prompts
- assumptions and citations, each labelled with provenance and dates
- feedback controls that separate "I liked this" from "here is what actually happened"

Built after `POST /decision` has a stable response schema (phase 5), so the UI isn't
rewritten as the engine changes. FastAPI serves the built files in production; Vite's dev
server is used during development.

## D4 — Python 3.11 (2026-10-10)

The 3.11 install is healthy and every library supports it. The 3.13 install
reports "Could not find platform independent libraries" and isn't used.
