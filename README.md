# ADAPT — Adaptive Decision and Planning Transformer

A general-purpose decision-making system that ranks options against a user's own
goals and constraints, uses cited evidence instead of making up facts, and asks
or abstains when it doesn't know enough.

Status: **phase 4 done**. Decision-frame schema; a policy that applies hard limits,
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
[phase 4 retrieval](docs/results/phase4_retrieval.md).
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
