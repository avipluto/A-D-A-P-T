# ADAPT — Adaptive Decision and Planning Transformer

A general-purpose decision-making system that ranks options against a user's own
goals and constraints, uses cited evidence instead of making up facts, and asks
or abstains when it doesn't know enough.

Status: **all 6 phases built**: a working app (web page + API + feedback store) with a controlled, gated retraining pipeline. Everything so far is validated on synthetic data only. Decision-frame schema; a policy that applies hard limits,
scores options, tests missing values against plausible fills, and reports what
would flip the result; a synthetic benchmark across 9 domains; a small learned
preference model (PyTorch) that plugs into the same policy; and text extraction (rules, a local Qwen
model, and a word tagger trained from scratch). Results:
[phase 1 baseline](docs/results/phase1_baseline.md),
[phase 2a learned model](docs/results/phase2a_preference_model.md),
[phase 2b missing values and sensitivity](docs/results/phase2b_missing_values.md),
[phase 3a extraction benchmark](docs/results/phase3a_extraction_benchmark.md),
[phase 3b local Qwen extractor](docs/results/phase3b_local_llm_extractor.md),
[phase 3c from-scratch tagger](docs/results/phase3c_from_scratch_tagger.md),
[phase 4 retrieval](docs/results/phase4_retrieval.md),
[phase 6 lifecycle](docs/results/phase6_lifecycle.md).
See also [docs/roadmap.md](docs/roadmap.md), [docs/architecture.md](docs/architecture.md)
and [docs/decisions.md](docs/decisions.md).

## Verified machine (2026-10-10)

| | |
|---|---|
| CPU | Intel i5-12450H |
| GPU | NVIDIA RTX 2050, 4 GiB VRAM, driver 617.42 |
| RAM | 15.6 GiB |
| Disk | C: ~18 GiB free, D: ~217 GiB free |
| Python | 3.11.4 (system install has a **CPU-only** torch 2.14) |

## Setup

Code stays here; the environment and data go on D: (see decisions D2).

Use `python -m pip`, not `pip`: on this machine the venv was created without the
`pip.exe` launcher, but the pip module itself is installed.

```powershell
py -3.11 -m venv D:\adapt-env
D:\adapt-env\Scripts\python.exe -m pip install --upgrade pip
D:\adapt-env\Scripts\python.exe -m pip install -r requirements.txt
D:\adapt-env\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build, ~2.5 GB
D:\adapt-env\Scripts\python.exe -m pip install -e .
```

Check that the GPU is visible:

```powershell
D:\adapt-env\Scripts\python.exe scripts/check_env.py
```

Data goes to `D:\adapt-data` by default; set `$env:ADAPT_DATA_DIR` to change it.

## Tests

```powershell
python -m pytest
```

## Benchmark

Generates 5,400 synthetic scenarios into `D:\adapt-data\evaluation` and writes
reports to `D:\adapt-data\artifacts\eval`. The baseline is always included:

```powershell
D:\adapt-env\Scripts\python.exe scripts/evaluate.py
D:\adapt-env\Scripts\python.exe scripts/evaluate.py --seed 1 --n-per-domain 2000   # large test set
```

Options: `--extension` (plausible-range stretch for missing values, default 2),
`--give-ranges` (give the engine the true attribute ranges), `--splits val` (for tuning).

## Live demo

**https://adapt-r39y.onrender.com** (free Render instance: sleeps when idle, so the first visit can take
30–60 s; stored data is wiped on restart).

The public server runs without PyTorch: both models are exported to NumPy (`scripts/export_numpy.py` →
`models/*.npz`), and tests check the NumPy outputs and decisions match PyTorch. Public mode
(`ADAPT_PUBLIC=1`, set in the `Dockerfile`) turns web search off and adds rate and size limits.
`render.yaml` deploys it; every push to `main` redeploys. After promoting a new model, re-run
`export_numpy.py` and commit `models/`.

## Run the app (phase 5)

```powershell
D:\adapt-env\Scripts\python.exe -m uvicorn --factory adapt.api.app:create_app --port 8000
```

Open **http://localhost:8000**: describe a decision (or fill in a form), check and correct what ADAPT
understood, then see the recommendation, close-call warnings, what would flip it, questions, sources,
and give feedback. API docs are at http://localhost:8000/docs.

To try things without touching the real feedback database, point it elsewhere first:
`$env:ADAPT_FEEDBACK_DB = "D:\adapt-data\feedback\test-feedback.sqlite"`.

Endpoints:

| Endpoint | What it does |
|---|---|
| `GET /health` | Server is up |
| `GET /model-info` | Versions of every component, and any fallback in use |
| `POST /decision` | Form input → decided. Text input → `needs_confirmation` with the extracted frame; send it back with `"confirmed": true` to decide. `"lookup": true` fills missing facts from the web, with citations |
| `GET /decision/{id}` | A stored decision and its feedback |
| `POST /feedback` | `satisfaction` (liked it?), `outcome` (how it turned out) or `correction`. Stored as pending review; never changes a model |

Decisions and feedback are stored in `D:\adapt-data\feedback\feedback.sqlite` with the request, the frame
used, the result and all component versions. If the learned scorer or tagger can't load, the API falls back
to the baseline / rules and `/model-info` lists the fallback.

## Learning from feedback (phase 6)

Nothing learns automatically. Each step is a command you run:

```powershell
# 1. review: only accepted feedback can ever be used
D:\adapt-env\Scripts\python.exe scripts/review_feedback.py list
D:\adapt-env\Scripts\python.exe scripts/review_feedback.py accept <feedback_id> ...
# 2. export accepted feedback as a versioned dataset (+ manifest) in D:\adapt-data\processed
D:\adapt-env\Scripts\python.exe scripts/review_feedback.py export
# 3. train a candidate on synthetic + real preferences
D:\adapt-env\Scripts\python.exe scripts/train_preference.py --name pref-v2 --real-data D:\adapt-data\processed\real-preferences-<hash>.jsonl
# 4. regression gate vs production; promote only if it passes AND you ask
D:\adapt-env\Scripts\python.exe scripts/gate.py --candidate D:\adapt-data\artifacts\models\pref-v2-s0\model.pt --real-data <same file>
D:\adapt-env\Scripts\python.exe scripts/gate.py --candidate ... --real-data ... --promote --reason "why"
```

A choice counts as "the chosen option beats the other options that met your limits", weighted by how it
actually turned out (bad outcomes are skipped; satisfaction alone counts half). Real decisions are split
80/20 so the gate tests on ones the candidate never saw. Promotion writes
`D:\adapt-data\artifacts\models\production.json` (read by the API) and appends to `history.jsonl`;
rolling back means promoting the previous checkpoint again.

## Looking up current facts (phase 4)

ADAPT can search for values a decision needs but you didn't give (e.g. a laptop's battery
life), using Tavily. Values are only taken from sentences that actually appear on a page,
checked for source quality, freshness and agreement between sites, and marked as retrieved
with a citation. Only well-supported values are filled in; the rest wait for your confirmation.

1. Create a free key at tavily.com.
2. Create a file named `.env` in this folder containing `TAVILY_API_KEY=tvly-...`
   (git-ignored; never put the key in code or chat).
3. Try it (1 credit per search, cached 24 h):

```powershell
D:\adapt-env\Scripts\python.exe scripts/lookup.py --option "Dell XPS 13 9350" --option "MacBook Air M3 13" --attribute battery_hours
```

## Learned preference model

Trains in ~10 s on the RTX 2050; checkpoints go to `D:\adapt-data\artifacts\models`:

```powershell
D:\adapt-env\Scripts\python.exe scripts/train_preference.py --seed 0
D:\adapt-env\Scripts\python.exe scripts/evaluate.py --model D:\adapt-data\artifacts\models\pref-v1-s0\model.pt
```
