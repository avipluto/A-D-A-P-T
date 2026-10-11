"""Assemble a Hugging Face Docker Space for the public ADAPT demo.

    python scripts/build_space.py                # -> D:/adapt-space (outside OneDrive)

Copies the app code and the production checkpoints (preference scorer + tagger, ~5 MB;
no Qwen), and writes a Dockerfile (CPU-only PyTorch, port 7860, user 1000), pinned
requirements, and the Space README. Public mode is on in the image: no web search,
rate and size limits, no reading stored decisions.
"""

from __future__ import annotations

import argparse
import shutil
from importlib.metadata import version
from pathlib import Path

from adapt import config
from adapt.api.services import DEFAULT_SCORER, DEFAULT_TAGGER
from adapt.lifecycle.registry import production_scorer

RUNTIME_PACKAGES = ["numpy", "pydantic", "fastapi", "uvicorn", "httpx"]

DOCKERFILE = """\
FROM python:3.11-slim

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/home/user/app/src \\
    ADAPT_PUBLIC=1 \\
    ADAPT_DATA_DIR=/tmp/adapt-data \\
    ADAPT_FEEDBACK_DB=/tmp/adapt-data/feedback/feedback.sqlite \\
    ADAPT_SCORER_CHECKPOINT=/home/user/app/models/scorer/model.pt \\
    ADAPT_TAGGER_CHECKPOINT=/home/user/app/models/tagger/model.pt
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch=={torch} --index-url https://download.pytorch.org/whl/cpu \\
 && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user models ./models

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "--factory", "adapt.api.app:create_app", "--host", "0.0.0.0", "--port", "7860", \\
     "--proxy-headers", "--forwarded-allow-ips", "*"]
"""

README = """\
---
title: ADAPT
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Decisions that respect your limits, ask when unsure
---

# ADAPT: Adaptive Decision and Planning Transformer

Describe a decision (options, what matters, hard limits). ADAPT shows what it understood for you to check,
then ranks the options with a small learned preference model, never recommends an option that breaks a
hard limit, asks when missing information could change the answer, and tells you what would flip the result.

Research prototype, validated on synthetic data so far. Inputs and feedback are stored on this server
(wiped on restart). Not professional advice. Source and evaluation: https://github.com/avipluto/A-D-A-P-T
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("D:/adapt-space"))
    args = p.parse_args()
    out: Path = args.out
    src = config.REPO_ROOT / "src" / "adapt"

    if out.exists():
        for child in out.iterdir():  # keep the Space's .git if it was cloned here
            if child.name != ".git":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    out.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src, out / "src" / "adapt", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, ck in (("scorer", production_scorer(DEFAULT_SCORER)), ("tagger", DEFAULT_TAGGER)):
        (out / "models" / name).mkdir(parents=True)
        shutil.copy2(ck, out / "models" / name / "model.pt")

    torch_version = version("torch").split("+")[0]
    (out / "Dockerfile").write_text(DOCKERFILE.replace("{torch}", torch_version), encoding="utf-8")
    (out / "requirements.txt").write_text("".join(f"{p}=={version(p)}\n" for p in RUNTIME_PACKAGES), encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and ".git" not in f.parts)
    print(f"Space bundle written to {out} ({size / 2**20:.1f} MiB, torch {torch_version} CPU)")


if __name__ == "__main__":
    main()
