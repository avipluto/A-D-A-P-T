# ADAPT — Adaptive Decision and Planning Transformer

A general-purpose decision-making system that ranks options against a user's own
goals and constraints, uses cited evidence instead of making up facts, and asks
or abstains when it doesn't know enough.

Status: **phase 2a done**. Decision-frame schema, a rule-based engine
(hard limits → weighted scoring → recommend/clarify/abstain), a synthetic
benchmark across 9 domains, and a small learned preference model (PyTorch) that
plugs into the same policy. Results:
[phase 1 baseline](docs/results/phase1_baseline.md),
[phase 2a learned model](docs/results/phase2a_preference_model.md).
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
```

## Learned preference model

Trains in ~10 s on the RTX 2050; checkpoints go to `D:\adapt-data\artifacts\models`:

```powershell
D:\adapt-env\Scripts\python.exe scripts/train_preference.py --seed 0
D:\adapt-env\Scripts\python.exe scripts/evaluate.py --model D:\adapt-data\artifacts\models\pref-v1-s0\model.pt
```
