# ADAPT — Adaptive Decision and Planning Transformer

A general-purpose decision-making system that ranks options against a user's own
goals and constraints, uses cited evidence instead of making up facts, and asks
or abstains when it doesn't know enough.

Status: **phase 1 done**. Decision-frame schema, a rule-based engine
(hard limits → weighted scoring → recommend/clarify/abstain) and a synthetic
benchmark across 9 domains. Results: [docs/results/phase1_baseline.md](docs/results/phase1_baseline.md).
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

```powershell
py -3.11 -m venv D:\adapt-env
D:\adapt-env\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build, ~2.5 GB
pip install -e .
$env:ADAPT_DATA_DIR = "D:\adapt-data"
```

Check that the GPU is visible:

```powershell
python scripts/check_env.py
```

## Tests

```powershell
python -m pytest
```

## Baseline benchmark

Generates 5,400 synthetic scenarios into `D:\adapt-data\evaluation` and writes the
report to `D:\adapt-data\artifacts\eval`:

```powershell
python scripts/evaluate_baseline.py
```
