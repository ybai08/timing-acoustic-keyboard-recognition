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
python scripts/train_acoustic_cnn.py --all-sessions
open "data/processed/spectrograms/session_20260514_185847/spectrogram_preview.html"
python scripts/visualize_acoustic_model.py --session all_sessions
open "models/acoustic_cnn/all_sessions/model_visualization.html"
python scripts/evaluate_segmentation.py --all-sessions --predict-cnn --extract-clips
python scripts/train_neural_segmenter.py --all-sessions
python scripts/run_acoustic_demo.py
```

Replace `session_20260514_185847` with your actual session folder name.
The `--all-sessions` command retrains the main optimized CNN on every processed spectrogram session. Train a single-session baseline only when you want a comparison against one recording batch.

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

Train the baseline on every processed spectrogram session:

```bash
python scripts/train_acoustic_baseline.py --all-sessions
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

Train the main optimized acoustic-only CNN on the entire processed dataset:

```bash
python scripts/train_acoustic_cnn.py --all-sessions
```

This first creates:

```text
data/processed/spectrograms/all_sessions/spectrogram_manifest.csv
```

Then it trains into:

```text
models/acoustic_cnn/all_sessions/
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

The optimized acoustic model is a compact ResNet-style CNN over the `64 x 10` spectrogram image. It uses class-balanced loss, SpecAugment-style masking, noise augmentation, mixup, AdamW, validation, and early stopping. This stays acoustic-only; it does not use timing or fusion features.

## Visualize Acoustic Models

Generate a browser-based visualization for the latest optimized acoustic CNN:

```bash
python scripts/visualize_acoustic_model.py
```

Generate it for a specific CNN session:

```bash
python scripts/visualize_acoustic_model.py --session session_20260514_230910
```

Generate it for the full-dataset CNN:

```bash
python scripts/visualize_acoustic_model.py --session all_sessions
```

Open the main full-dataset CNN visualization:

```bash
open "models/acoustic_cnn/all_sessions/model_visualization.html"
```

The unified viewer detects the model type. For the optimized CNN, it shows architecture, training history, a confusion matrix, and held-out prediction probabilities.

Generate the logistic-regression baseline visualization instead:

```bash
python scripts/visualize_acoustic_model.py --kind baseline --session session_20260514_230910
open "models/acoustic_baseline/session_20260514_230910/model_visualization.html"
```

The logistic baseline structure is:

```text
64 x 10 log-mel spectrogram
-> 640 flattened input features
-> StandardScaler
-> logistic regression
-> one probability for each key class
```

For the logistic baseline, the viewer also shows per-key learned weight heatmaps.

## Evaluate Automatic Segmentation

Evaluate whether raw-audio peak detection finds the same keypresses as the digital key log:

```bash
python scripts/evaluate_segmentation.py --all-sessions
```

Tune detector parameters against every aligned session:

```bash
python scripts/evaluate_segmentation.py --all-sessions --tune
```

Evaluate the tuned/default detector, run the full-dataset CNN on matched detected clips, and save detected clips:

```bash
python scripts/evaluate_segmentation.py --all-sessions --predict-cnn --extract-clips
```

What it creates:

```text
data/metadata/segmentation/all_sessions/segmentation_report.txt
data/metadata/segmentation/all_sessions/segmentation_report.json
data/metadata/segmentation/all_sessions/segmentation_matches.csv
data/metadata/segmentation/all_sessions/segmentation_tuning.json
data/processed/detected_clips/all_sessions_detected/clip_manifest.csv
```

This is the bridge between the oracle training pipeline and the live app. It measures detection separately from classification:

```text
raw audio
-> detected peaks
-> match peaks against known keydown times
-> extract detected clips
-> classify matched detected clips with the CNN
```

## Train The Neural Segmenter

Train NN #1 in the two-network pipeline:

```bash
python scripts/train_neural_segmenter.py --all-sessions
```

What it learns:

```text
raw phrase audio window
-> probability that a keypress is centered in that window
```

What it creates:

```text
models/neural_segmenter/all_sessions/model.pt
models/neural_segmenter/all_sessions/metrics.json
models/neural_segmenter/all_sessions/training_history.csv
models/neural_segmenter/all_sessions/test_event_predictions.csv
models/neural_segmenter/all_sessions/report.txt
```

Current held-out neural segmenter result:

```text
No current result yet. The old MacBook-keyboard models were deleted; retrain after collecting Keychron V6 trials.
```

Use the trained segmenter to turn one raw WAV file into individual clips:

```bash
python scripts/run_neural_segmenter.py --audio "path/to/typing.wav" --expected-keys 5
```

This writes detected clips and a manifest here by default:

```text
data/processed/neural_segments/manual/
```

## Run The Acoustic CNN Demo

Launch the local browser app that records a short typing audio clip, segments it, and decodes it with the full-dataset CNN after models have been retrained:

```bash
python scripts/run_acoustic_demo.py
```

Launch without opening a browser automatically:

```bash
python scripts/run_acoustic_demo.py --no-open
```

Use a different model folder:

```bash
python scripts/run_acoustic_demo.py --model-dir "models/acoustic_cnn/all_sessions"
```

What it does:

```text
browser recording
-> neural segmenter if trained, otherwise heuristic peak detection
-> fixed windows around detected peaks
-> log-mel spectrograms
-> optimized acoustic CNN
-> predicted string and per-key probabilities
```

Each decode is saved as:

```text
data/raw/inference_runs/<run_id>/recording.wav
data/processed/inference_runs/<run_id>/clips/
data/processed/inference_runs/<run_id>/clip_manifest.csv
data/metadata/inference_runs/<run_id>/metadata.json
```

The browser displays the raw recording and every generated single-key clip with audio controls.

This demo is acoustic-only. It does not use the true typed text, browser key logs, timing features, or language-model correction.
For best first tests, keep the typed phrase short and set `Expected Keys` to the number of keys you plan to press. The live demo uses that as a hard cap before the CNN runs, which helps prevent one short recording from producing a long sequence of false key detections.
