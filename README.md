# Timing + Acoustic Keyboard Recognition

This is a beginner-friendly research project scaffold for studying whether keyboard timing/rhythm can improve acoustic key or text recovery.

The initial research question is:

> Does inter-keystroke timing improve recovery compared with acoustics alone?

## Project Status

Current stage: project scaffold and planning.

The repo includes:

- a step-by-step implementation plan as Markdown and PDF
- a clean folder structure for data, models, reports, scripts, and source code
- starter Python package files
- dependency files
- beginner setup scripts

## Folder Structure

```text
configs/                 Experiment settings
data/raw/                Raw local recordings, ignored by git
data/processed/          Processed local features/clips, ignored by git
data/metadata/           Local metadata/timestamps, ignored by git
models/                  Trained models, ignored by git
notebooks/               Exploration notebooks
reports/                 Notes, PDFs, and final writeups
scripts/                 Command-line helper scripts
src/keyboard_fusion/     Python package source code
tests/                   Tests
```

## Beginner Setup

From this folder, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional later, for PyTorch-based acoustic models:

```bash
python -m pip install -r requirements-ml.txt
```

## First Milestone

Build one small, controlled experiment:

```text
1 user
1 keyboard
1 microphone
known prompts
audio + key timestamps
oracle segmentation
simple acoustic classifier
timing features
acoustic-only vs acoustic+timing comparison
```

## Ethics

Use only consenting participants and synthetic prompts. Do not collect real passwords or private text. This project should be used to measure leakage, understand limitations, and evaluate defenses.

