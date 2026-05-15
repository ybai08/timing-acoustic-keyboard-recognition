# Timing + Acoustic Keyboard Recognition

This is a beginner-friendly research project scaffold for studying whether keyboard timing/rhythm can improve acoustic key or text recovery.

The initial research question is:

> Does inter-keystroke timing improve recovery compared with acoustics alone?

## Project Status

Current stage: reset for a new Keychron V6 dataset. The previous MacBook-keyboard data and trained models were removed, so `data/` and `models/` are intentionally empty except for placeholders. The pipeline code is ready; the next step is collecting fresh Keychron V6 trials before retraining the acoustic CNN and neural segmenter.

The repo includes:

- a step-by-step implementation plan as Markdown and PDF
- a clean folder structure for data, models, reports, scripts, and source code
- Python package code for collection, alignment, clipping, spectrograms, and acoustic models
- dependency files for the base pipeline and optional PyTorch CNN training
- beginner setup scripts

## First Experiment Scope

The first experiment is intentionally narrow and controlled. The goal is to build one clean end-to-end version before adding more users, microphones, rooms, or keyboards.

Target device:

```text
Keyboard: Keychron V6 wired mechanical keyboard
Connection: wired USB
Computer: current MacBook Air used only to run collection/training
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
- one keyboard: Keychron V6 wired mechanical keyboard
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
train acoustic-only models, compute timing features, and report whether acoustic +
timing performs better than acoustic-only on the same controlled dataset.
```

## Folder Structure

```text
configs/                 Experiment settings
data/raw/                Raw local recordings and per-trial logs, ignored by git
data/processed/          Processed local features/clips, ignored by git
data/metadata/           Future dataset-level indexes/summaries, ignored by git
models/                  Trained models, ignored by git
prompts/                 Synthetic prompt lists used by the collector
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

For the optimized acoustic CNN, install the ML add-on:

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

After a trial is saved, the typing box clears automatically. If you notice a bad take right away, use `Delete trial_###` to remove the last saved trial's raw WAV, event CSV, and metadata JSON files, then retry the same prompt.

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

When two logged keypresses are extremely close together, their full fixed windows can overlap. The alignment metadata therefore also stores a neighbor-aware `isolation_*` region for each key. That region is split at the midpoint between neighboring keydowns so the extractor can keep one key's main sound while avoiding duplicate neighboring clicks in both clips.

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

The clip duration stays fixed for model consistency. If two keydowns are very close, the extractor keeps the fixed clip length but silences audio outside that key's neighbor-aware isolation region. This gives the model same-shaped examples while reducing cases where two adjacent clips contain the same pair of clicks.

Clip outputs are written under `data/processed/clips/<session_id>/`:

```text
trial_001/trial_001_event_000_keyh_h.wav
clip_manifest.csv
clip_extraction_report.txt
```

The manifest is the important index for training later: it connects each clip path to the key label, trial ID, prompt, keydown time, sample window, and any neighbor-aware isolation/masking that was applied.

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

## Training Acoustic Models

After generating spectrograms, run:

```bash
source .venv/bin/activate
python scripts/train_acoustic_baseline.py
```

This trains the first acoustic-only baseline: logistic regression on flattened normalized log-mel spectrograms. It is intentionally simple so you can measure the pipeline before adding a neural network.

For the main stronger acoustic-only model, train across every processed spectrogram session:

```bash
source .venv/bin/activate
python scripts/train_acoustic_cnn.py --all-sessions
```

This creates `data/processed/spectrograms/all_sessions/spectrogram_manifest.csv`, then trains a compact ResNet-style CNN directly on the combined `64 x 10` log-mel spectrograms. It uses class-balanced loss, SpecAugment-style frequency/time masking, noise augmentation, mixup, AdamW, a validation split, and early stopping. This is the best acoustic-only model currently in the project; timing and fusion are still separate future steps.

For a quick per-session comparison, you can still run:

```bash
python scripts/train_acoustic_cnn.py --session <session_id>
```

Model outputs are written under `models/acoustic_baseline/<session_id>/`:

```text
model.joblib
metrics.json
test_predictions.csv
test_probabilities.csv
report.txt
```

CNN outputs are written under `models/acoustic_cnn/<session_id>/`. For the full-dataset run, the session ID is `all_sessions`:

```text
model.pt
metrics.json
test_predictions.csv
test_probabilities.csv
training_history.csv
report.txt
```

The predictions file contains top-1 and top-5 guesses for each held-out clip. The probabilities file contains one probability per candidate key for each held-out clip.

To visualize the full-dataset acoustic CNN, run:

```bash
python scripts/visualize_acoustic_model.py --session all_sessions
open "models/acoustic_cnn/all_sessions/model_visualization.html"
```

The unified viewer detects whether the model folder contains the logistic baseline or the optimized CNN. For the CNN it shows model structure, training history, a confusion matrix, and held-out prediction probabilities. For the logistic baseline it also shows per-key learned weight heatmaps.

To visualize the logistic baseline instead:

```bash
python scripts/visualize_acoustic_model.py --kind baseline --session <session_id>
open "models/acoustic_baseline/<session_id>/model_visualization.html"
```

## Evaluating Automatic Segmentation

The acoustic CNN is trained on already-isolated key clips. Before using it on raw audio, evaluate whether the project can find individual keypresses from audio alone:

```bash
source .venv/bin/activate
python scripts/evaluate_segmentation.py --all-sessions --predict-cnn --extract-clips
```

This compares audio-detected peaks against the known keydown times from the data collector, writes a segmentation report, saves a match CSV, and creates detected-peak clips under `data/processed/detected_clips/`.

Current full-dataset result:

```text
No current result yet. Collect fresh Keychron V6 data, then run this evaluation after retraining.
```

This means the CNN performs well when the detector finds the correct click. The remaining live-app problem is mostly missed or extra detected keypresses.

## Training The Neural Segmenter

The two-network version of the project adds a neural segmenter before the acoustic CNN:

```text
raw multi-key audio
-> neural segmenter predicts keypress centers
-> fixed clips around those centers
-> acoustic CNN predicts each key
-> decoded string
```

Train the neural segmenter from every aligned trial:

```bash
source .venv/bin/activate
python scripts/train_neural_segmenter.py --all-sessions
```

This uses the existing aligned keydown timestamps as supervised labels. Outputs are written under:

```text
models/neural_segmenter/all_sessions/
```

Current held-out result:

```text
No current result yet. The neural segmenter needs to be retrained on fresh Keychron V6 trials.
```

To segment one WAV file into individual clips:

```bash
python scripts/run_neural_segmenter.py --audio "path/to/typing.wav" --expected-keys 5
```

That writes detected `.wav` clips plus a `clip_manifest.csv` under `data/processed/neural_segments/manual/`.

## Running The Acoustic Demo

After collecting Keychron V6 data and training the full-dataset CNN and neural segmenter, launch the local acoustic-only demo:

```bash
source .venv/bin/activate
python scripts/run_acoustic_demo.py
```

The demo records a short audio clip, uses `models/neural_segmenter/all_sessions/model.pt` when available to detect keystroke clips, classifies each detected clip with `models/acoustic_cnn/all_sessions/model.pt`, and shows the predicted string plus per-key probabilities. It also saves each decode as an inference run:

```text
data/raw/inference_runs/<run_id>/recording.wav
data/processed/inference_runs/<run_id>/clips/
data/processed/inference_runs/<run_id>/clip_manifest.csv
data/metadata/inference_runs/<run_id>/metadata.json
```

The browser shows audio controls for the raw recording and each generated single-key clip so you can audit what the neural segmenter produced.

This is still less reliable than the oracle training pipeline because it must detect keypresses from raw audio first. Use synthetic text only, start with short phrases, and set `Expected Keys` to the number of keys you plan to press. The live detector uses that value as a hard cap so small echoes and key-release sounds do not turn a short recording into many extra guesses.

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
optimized acoustic-only CNN
timing features
acoustic-only vs acoustic+timing comparison
```

## Ethics

Use only consenting participants and synthetic prompts. Do not collect real passwords or private text. This project should be used to measure leakage, understand limitations, and evaluate defenses.

## License

This project is available under the MIT License. See [LICENSE](LICENSE).
