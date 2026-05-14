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

## First Experiment Scope

The first experiment is intentionally narrow and controlled. The goal is to build one clean end-to-end version before adding more users, microphones, rooms, or keyboards.

Target device:

```text
Keyboard/computer: current MacBook Air
Model identifier: Mac14,2
Model number: MLY13LL/A
Chip: Apple M2
Memory: 8 GB
```

Recording setup:

```text
Microphone: Scarlett Solo USB microphone/interface setup
Microphone configuration: 360-degree/omnidirectional setting
Mic muffler: none
Room: user's room only
Noise control: AC unit off during recording
```

First-version constraints:

- one participant: the project owner
- one keyboard: the built-in keyboard on the current MacBook Air
- one microphone setup: Scarlett Solo USB setup described above
- one room environment: the user's room with AC off
- known synthetic prompts only
- no real passwords or private text
- audio plus exact keydown/keyup timestamps
- oracle segmentation first, using the true keydown timestamps to cut clips

First comparison:

```text
acoustic-only baseline
vs.
timing-only baseline
vs.
acoustic + timing fusion
```

First success definition:

```text
The first version succeeds if it can collect synchronized audio and key timing data,
train a simple acoustic baseline, compute timing features, and report whether
acoustic + timing performs better than acoustic-only on the same controlled dataset.
```

## Folder Structure

```text
configs/                 Experiment settings
data/raw/                Raw local recordings and per-trial logs, ignored by git
data/processed/          Processed local features/clips, ignored by git
data/metadata/           Future dataset-level indexes/summaries, ignored by git
models/                  Trained models, ignored by git
notebooks/               Exploration notebooks
prompts/                 Synthetic prompt lists used by the collector
reports/                 Notes, PDFs, and final writeups
scripts/                 Command-line helper scripts
src/keyboard_fusion/     Python package source code
tests/                   Tests
web/                     Browser UI for foreground data collection
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

## Collecting A Trial

Run the browser-based local collection app from the project folder:

```bash
source .venv/bin/activate
python scripts/collect_trials.py
```

This starts a tiny local server and opens the collector in your browser. The app is intentionally explicit: it only records after you click `Start Trial`, and it only logs keys typed inside the app's typing box.

For each saved trial, the app writes three files under `data/raw/sessions/<session_id>/`:

```text
trial_001.wav
trial_001_events.csv
trial_001_metadata.json
```

The `.wav` file is the microphone recording. The `_events.csv` file is the digital key log, with rows like "keydown for j at 0.532 seconds" and "keyup for j at 0.612 seconds." The `_metadata.json` file connects the audio and event log to the prompt, participant ID, keyboard, microphone, and room setup.

Before recording:

- connect the Scarlett Solo USB microphone/interface setup
- confirm the microphone input is selected by macOS if needed
- turn the AC unit off
- use only synthetic prompts
- avoid real passwords or private text

Raw recordings and processed data are ignored by Git.

## Collector Architecture

There is one collection launcher:

```text
scripts/collect_trials.py
```

That launcher starts the local web server in:

```text
src/keyboard_fusion/web_collection_server.py
```

The server opens the browser interface in:

```text
web/collector.html
web/collector.css
web/collector.js
```

Shared collection helpers live in:

```text
src/keyboard_fusion/collection.py
```

The browser collector is the canonical collection path; there are no duplicate collector launchers.

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

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
