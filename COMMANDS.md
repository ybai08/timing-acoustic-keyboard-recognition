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
open "data/processed/spectrograms/session_20260514_185847/spectrogram_preview.html"
```

Replace `session_20260514_185847` with your actual session folder name.

## Training

Training is the next project milestone and does not have a command yet. The current pipeline gets data into the right shape for training:

```text
raw audio + key logs
-> aligned key events
-> single-key WAV clips
-> normalized log-mel spectrogram arrays
```

The next command we should add will likely train the first acoustic baseline from `data/processed/spectrograms/<session_id>/spectrogram_manifest.csv`.
