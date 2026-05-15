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

For a fuller command reference, see [COMMANDS.md](COMMANDS.md).

## Collecting A Trial

Run the browser-based local collection app from the project folder:

```bash
source .venv/bin/activate
python scripts/collect_trials.py
```

This starts a tiny local server and opens the collector in your browser. The app is intentionally explicit: it only records after you click `Start Trial`, and it only logs keys typed inside the app's typing box. Use the `Recording Input` dropdown to choose the microphone before starting a trial.

For each saved trial, the app writes three files under `data/raw/sessions/<session_id>/`:

```text
trial_001.wav
trial_001_events.csv
trial_001_metadata.json
```

The `.wav` file is the microphone recording. The `_events.csv` file is the digital key log, with rows like "keydown for j at 0.532 seconds" and "keyup for j at 0.612 seconds." The `_metadata.json` file connects the audio and event log to the prompt, participant ID, keyboard, microphone, browser-selected audio input, and room setup.

Before recording:

- connect the Scarlett Solo USB microphone/interface setup
- click `Allow / Refresh` in the collector if microphone names are hidden
- choose the Scarlett/input device from the `Recording Input` dropdown
- confirm the microphone input is selected by macOS if needed
- turn the AC unit off
- use only synthetic prompts
- avoid real passwords or private text

Raw recordings and processed data are ignored by Git.

## Aligning Recorded Trials

After collecting trials, run:

```bash
source .venv/bin/activate
python scripts/align_trials.py
```

This maps every non-repeat `keydown` event to a WAV sample index and a fixed extraction window. The browser key log and WAV stream can start a little out of sync, so the alignment step estimates a per-trial audio offset from waveform energy before creating sample windows. A beep marker can still be added later if you need tighter audio-clock calibration.

Alignment outputs are written under `data/metadata/alignment/<session_id>/`:

```text
trial_001_alignment.json
alignment_report.txt
```

## Extracting Keystroke Clips

After alignment, run:

```bash
source .venv/bin/activate
python scripts/extract_clips.py
```

This cuts one labeled `.wav` clip for every aligned non-repeat `keydown` event. For the current oracle baseline, each clip uses the configured extraction window around the offset-corrected true keydown timestamp. The default window is intentionally short, `20 ms` before keydown through `45 ms` after keydown, to reduce neighboring keys leaking into the same clip.

Clip outputs are written under `data/processed/clips/<session_id>/`:

```text
trial_001/trial_001_event_000_keyh_h.wav
clip_manifest.csv
clip_extraction_report.txt
```

The manifest is the important index for training later: it connects each clip path to the key label, trial ID, prompt, keydown time, and sample window.

## Generating Spectrograms

After extracting clips, run:

```bash
source .venv/bin/activate
python scripts/generate_spectrograms.py
```

This converts every extracted clip into a normalized log-mel spectrogram array for model training.

Spectrogram outputs are written under `data/processed/spectrograms/<session_id>/`:

```text
trial_001/trial_001_event_000_keyh_h_logmel.npz
spectrogram_manifest.csv
spectrogram_report.txt
spectrogram_preview.html
```

Each `.npz` file contains `spectrogram`, the normalized model input, and `log_mel`, the unnormalized log-mel values. Open the preview HTML to choose a trial and inspect every generated waveform and spectrogram pair for that trial; the yellow vertical line marks the logged keydown position inside each clip.

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
