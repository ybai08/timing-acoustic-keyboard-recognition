# Project Commands

This file is the quick command reference for the project. Run commands from the main project folder:

```bash
cd "/Users/yangbai/Desktop/Computer Science/Projects/Timing Acoustic Keyboard Recognition"
source .venv/bin/activate
```

## Setup And Health Checks

Check whether the environment and expected folders look usable:

```bash
python scripts/check_setup.py
```

Run the full test suite:

```bash
python -m pytest
```

Run the project test helper:

```bash
scripts/run_tests.sh
```

Check that Python files compile:

```bash
python -m compileall -q src scripts
```

## Collect New Data

Launch the browser-based data collection app:

```bash
python scripts/collect_trials.py
```

What it creates:

```text
data/raw/sessions/<session_id>/trial_001.wav
data/raw/sessions/<session_id>/trial_001_events.csv
data/raw/sessions/<session_id>/trial_001_metadata.json
```

## Align Key Events To Audio

Align the latest raw session:

```bash
python scripts/align_trials.py
```

Align a specific raw session:

```bash
python scripts/align_trials.py --session session_20260514_185847
```

Change the key clip window used during alignment:

```bash
python scripts/align_trials.py --pre-ms 20 --post-ms 45
```

Close keypresses can make the full fixed windows overlap. Alignment records both the fixed window and a smaller neighbor-aware keep region, split at the midpoint between adjacent keydowns. The extraction command keeps clip length fixed, but silences audio outside that keep region so very fast keypairs are not duplicated across both clips.

What it creates:

```text
data/metadata/alignment/<session_id>/trial_001_alignment.json
data/metadata/alignment/<session_id>/alignment_report.txt
```

## Extract Single-Key Audio Clips

Extract clips from the latest aligned session:

```bash
python scripts/extract_clips.py
```

Extract clips from a specific aligned session:

```bash
python scripts/extract_clips.py --session session_20260514_185847
```

What it creates:

```text
data/processed/clips/<session_id>/trial_001/trial_001_event_000_keyh_h.wav
data/processed/clips/<session_id>/clip_manifest.csv
data/processed/clips/<session_id>/clip_extraction_report.txt
```

The manifest includes the fixed clip window plus `isolation_*` columns. If `overlap_adjusted_left` or `overlap_adjusted_right` is `True`, that side of the clip was silenced because a neighboring keypress was very close.

## Generate Spectrograms And Preview

Generate spectrograms for the latest extracted clip session:

```bash
python scripts/generate_spectrograms.py
```

Generate spectrograms for a specific clip session:

```bash
python scripts/generate_spectrograms.py --session session_20260514_185847
```

Generate a preview that includes every clip:

```bash
python scripts/generate_spectrograms.py --session session_20260514_185847 --preview-count 0
```

Generate a smaller preview with only the first 20 clips:

```bash
python scripts/generate_spectrograms.py --session session_20260514_185847 --preview-count 20
```

Open the preview in your browser:

```bash
open "data/processed/spectrograms/session_20260514_185847/spectrogram_preview.html"
```

The preview page lets you choose a trial, see every key pressed in that trial, and inspect waveform plus log-mel spectrogram views for each key. The yellow vertical line marks the logged keydown position inside the extracted clip.

What it creates:

```text
data/processed/spectrograms/<session_id>/trial_001/trial_001_event_000_keyh_h_logmel.npz
data/processed/spectrograms/<session_id>/spectrogram_manifest.csv
data/processed/spectrograms/<session_id>/spectrogram_report.txt
data/processed/spectrograms/<session_id>/spectrogram_preview.html
```

## Usual Flow After Recording More Trials

Use this after you add new raw trials:

```bash
python scripts/align_trials.py --session session_20260514_185847
python scripts/extract_clips.py --session session_20260514_185847
python scripts/generate_spectrograms.py --session session_20260514_185847 --preview-count 0
python scripts/train_acoustic_baseline.py --session session_20260514_185847
python scripts/train_acoustic_cnn.py --session session_20260514_185847
open "data/processed/spectrograms/session_20260514_185847/spectrogram_preview.html"
```

Replace `session_20260514_185847` with your actual session folder name.

## Train The Logistic Acoustic Baseline

Train the first acoustic-only baseline on the latest spectrogram session:

```bash
python scripts/train_acoustic_baseline.py
```

Train on a specific spectrogram session:

```bash
python scripts/train_acoustic_baseline.py --session session_20260514_185847
```

Train from a specific manifest:

```bash
python scripts/train_acoustic_baseline.py --spectrogram-manifest "data/processed/spectrograms/session_20260514_185847/spectrogram_manifest.csv"
```

Change the test split size:

```bash
python scripts/train_acoustic_baseline.py --session session_20260514_185847 --test-size 0.25
```

What it creates:

```text
models/acoustic_baseline/<session_id>/model.joblib
models/acoustic_baseline/<session_id>/metrics.json
models/acoustic_baseline/<session_id>/test_predictions.csv
models/acoustic_baseline/<session_id>/test_probabilities.csv
models/acoustic_baseline/<session_id>/report.txt
```

The first acoustic baseline is logistic regression on flattened normalized log-mel spectrograms. It is intentionally simple: the purpose is to get a real acoustic-only measuring stick before building a neural network. The `test_predictions.csv` file gives top-1 and top-5 guesses for each held-out clip. The `test_probabilities.csv` file gives one probability per candidate key for each held-out clip.

## Train The Optimized Acoustic CNN

Install the ML dependency once:

```bash
python -m pip install -r requirements-ml.txt
```

Train the optimized acoustic-only CNN on the latest spectrogram session:

```bash
python scripts/train_acoustic_cnn.py
```

Train on a specific spectrogram session:

```bash
python scripts/train_acoustic_cnn.py --session session_20260514_185847
```

Useful tuning flags:

```bash
python scripts/train_acoustic_cnn.py --session session_20260514_185847 --epochs 220 --patience 35 --batch-size 64
python scripts/train_acoustic_cnn.py --session session_20260514_185847 --device cpu
```

What it creates:

```text
models/acoustic_cnn/<session_id>/model.pt
models/acoustic_cnn/<session_id>/metrics.json
models/acoustic_cnn/<session_id>/test_predictions.csv
models/acoustic_cnn/<session_id>/test_probabilities.csv
models/acoustic_cnn/<session_id>/training_history.csv
models/acoustic_cnn/<session_id>/report.txt
```

The optimized acoustic model is a compact ResNet-style CNN over the `64 x 10` spectrogram image. It uses class-balanced loss, light SpecAugment-style masking, small noise augmentation, AdamW, validation, and early stopping. This stays acoustic-only; it does not use timing or fusion features.

## Visualize The Logistic Acoustic Baseline

Generate a browser-based model visualization for the latest acoustic baseline:

```bash
python scripts/visualize_acoustic_model.py
```

Generate it for a specific session:

```bash
python scripts/visualize_acoustic_model.py --session session_20260514_185847
```

Open the visualization:

```bash
open "models/acoustic_baseline/session_20260514_185847/model_visualization.html"
```

This viewer is for the logistic-regression baseline, not the CNN. Its structure is:

```text
64 x 10 log-mel spectrogram
-> 640 flattened input features
-> StandardScaler
-> logistic regression
-> one probability for each key class
```

The visualization shows this structure, a per-key learned weight heatmap, a confusion matrix, and held-out prediction probabilities.

For the optimized CNN, inspect these files instead:

```text
models/acoustic_cnn/<session_id>/metrics.json
models/acoustic_cnn/<session_id>/training_history.csv
models/acoustic_cnn/<session_id>/test_predictions.csv
models/acoustic_cnn/<session_id>/report.txt
```
