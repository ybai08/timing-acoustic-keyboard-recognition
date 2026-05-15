# Master Todo: Timing + Acoustic Keyboard Recognition

Use this as the single continuous path through the project. Each step should be mostly complete before moving to the next one.

## Working Rule

- [ ] Only use consent-based data.
- [ ] Only use synthetic prompts, never real passwords or private text.
- [ ] Keep raw audio, processed data, and trained models out of Git.
- [ ] Keep one obvious launcher for each workflow; avoid duplicate scripts that do the same job.
- [ ] After each milestone, run tests and commit the working state.

## Continuous Project Path

### 1. Lock The Project Scope

- [x] Write the first-scope statement in `README.md`.
- [x] Confirm the first version uses one keyboard, one microphone, one quiet room, and known prompts.
- [x] Confirm the first experiment compares acoustic-only vs timing-only vs acoustic + timing.
- [x] Define what "success" means for the first version.

Done when: the project goal is clear enough that every later task supports the same experiment.

### 2. Finalize The Config File

- [x] Open `configs/default.yaml`.
- [x] Add project-wide settings for sample rate, audio window size, prompt types, random seed, and output folders.
- [x] Add acoustic settings for mel bands, FFT window size, hop length, and clip duration.
- [x] Add timing settings for dwell time, press-press latency, release-press latency, and release-release latency.
- [x] Add evaluation settings for train/test split and metrics.

Done when: scripts can read one config file instead of relying on hard-coded experiment values.

### 3. Make The Setup Check Useful

- [x] Update `scripts/check_setup.py` so it checks the project folders exist.
- [x] Make it check that `configs/default.yaml` can be loaded.
- [x] Make it check required packages from `requirements.txt`.
- [x] Make it print a simple "ready for data collection" message when setup is valid.

Done when: a new user can run one command and know whether the project environment is ready.

### 4. Create Prompt Lists

- [x] Create a prompt file for isolated keys.
- [x] Create a prompt file for short English phrases.
- [x] Create a prompt file for random character strings.
- [x] Create a prompt file for synthetic password-like strings.
- [x] Add a short note explaining that prompts are synthetic and consent-based.

Done when: the data collector can load prompts from files instead of using manually typed examples.

### 5. Build The Typing Event Collector

- [x] Create a simple browser-based local typing collector.
- [x] Show one prompt at a time.
- [x] Record each `keydown` event.
- [x] Record each `keyup` event.
- [x] Record typed characters.
- [x] Save participant ID, trial ID, prompt text, typed text, and timestamps.
- [x] Save each trial as structured metadata beside its matching raw audio and event log.
- [x] Keep `scripts/collect_trials.py` as the single canonical collector launcher.

Done when: you can type a prompt and get a clean event log with keydown and keyup timestamps.

### 6. Add Audio Recording Or Audio File Linking

- [x] Decide whether the first version records audio directly or links externally recorded audio.
- [x] Add an `audio_file_path` field to each trial metadata record.
- [x] Store raw local recordings under `data/raw/`.
- [x] Confirm raw audio is ignored by Git.
- [x] Add a simple recording checklist for microphone position, room notes, keyboard model, and sample rate.
- [x] Let the browser collector choose and save the recording input device.

Done when: every trial has matching keyboard events and an audio file path.

### 7. Add Audio/Event Alignment

- [x] Decide whether the first alignment method uses a beep marker or shared local timestamps.
- [x] Create an alignment script that maps keydown timestamps to audio time.
- [x] Save alignment metadata for each trial.
- [x] Add a small visual or text report showing keydown times against audio windows.

Done when: the project can reliably locate where each keypress should appear in the audio file.

### 8. Implement Oracle Keystroke Extraction

- [x] Create `src/keyboard_fusion/preprocessing.py`.
- [x] Load one trial's audio and metadata.
- [x] For each keydown timestamp, cut a fixed window around the keypress.
- [x] Save extracted clips under `data/processed/`.
- [x] Save clip metadata with key label, trial ID, start time, and end time.
- [x] Add tests for clip window calculations.

Done when: known keydown timestamps can produce labeled single-key audio clips.

### 9. Build Log-Mel Spectrogram Generation

- [ ] Create `src/keyboard_fusion/spectrograms.py`.
- [ ] Convert one extracted clip into a log-mel spectrogram.
- [ ] Normalize spectrogram values.
- [ ] Save spectrogram arrays or images under `data/processed/`.
- [ ] Add a script to generate spectrograms for all clips.
- [ ] Add a quick notebook or report view to inspect a few spectrograms.

Done when: every extracted keystroke clip can become a model-ready spectrogram.

### 10. Train The First Acoustic Baseline

- [ ] Create `src/keyboard_fusion/acoustic_model.py`.
- [ ] Start with a small CNN or simple classical baseline before using larger models.
- [ ] Train on isolated-key spectrograms.
- [ ] Output probability scores for every possible key.
- [ ] Save model outputs for each test clip.
- [ ] Report top-1 and top-5 key accuracy.

Done when: the project has an acoustic-only baseline with saved per-key probabilities.

### 11. Expand Timing Feature Extraction

- [ ] Review `src/keyboard_fusion/timing_features.py`.
- [ ] Keep dwell time.
- [ ] Keep press-press latency.
- [ ] Keep release-press latency.
- [ ] Keep release-release latency.
- [ ] Add pause and burst indicators.
- [ ] Add keyboard layout distance features.
- [ ] Add same-hand and same-finger transition features.
- [ ] Add tests for each timing feature.

Done when: each typed sequence has timing features that describe both rhythm and keyboard movement.

### 12. Train A Timing-Only Baseline

- [ ] Create `src/keyboard_fusion/timing_model.py`.
- [ ] Start with a simple model such as logistic regression, random forest, or gradient boosting.
- [ ] Test timing-only user identification or transition prediction first.
- [ ] Test timing-only string ranking if enough data exists.
- [ ] Save timing scores in a format the fusion decoder can use.

Done when: timing has a measurable baseline, even if it cannot identify exact isolated keys well.

### 13. Build The First Fusion Decoder

- [ ] Create `src/keyboard_fusion/fusion_decoder.py`.
- [ ] Load acoustic probabilities for each keystroke.
- [ ] Load timing features between consecutive keystrokes.
- [ ] Implement a simple scoring function for candidate sequences.
- [ ] Implement beam search.
- [ ] Output top-k candidate strings.
- [ ] Keep the first decoder language-model-free.

Done when: the project can decode a typed sequence using acoustic evidence plus timing evidence.

### 14. Evaluate Acoustic-Only vs Fusion

- [ ] Create `src/keyboard_fusion/evaluation.py`.
- [ ] Evaluate acoustic-only character accuracy.
- [ ] Evaluate acoustic-only top-k accuracy.
- [ ] Evaluate acoustic + timing character accuracy.
- [ ] Evaluate acoustic + timing top-k accuracy.
- [ ] Evaluate edit distance and character error rate.
- [ ] Compare the systems on phrases, random strings, and synthetic password-like strings separately.

Done when: there is a clear table showing whether timing improves acoustic recovery.

### 15. Study Ambiguous Acoustic Cases

- [ ] Find cases where acoustic predictions are uncertain.
- [ ] Find cases where the true key is in the acoustic model's top 5 but not top 1.
- [ ] Check whether timing helps choose the correct key in those cases.
- [ ] Pay special attention to nearby keys and same-row neighbors.
- [ ] Write a short analysis in `reports/`.

Done when: the project can explain where timing helps, not just whether it helps overall.

### 16. Add Automatic Keystroke Segmentation

- [ ] Create `src/keyboard_fusion/segmentation.py`.
- [ ] Compute short-time audio energy.
- [ ] Detect keystroke peaks.
- [ ] Match detected peaks to known key events for evaluation.
- [ ] Extract clips from detected peaks.
- [ ] Compare oracle segmentation against automatic segmentation.

Done when: the project can measure how much performance drops when true timestamps are not used for clipping.

### 17. Add Better Acoustic Models

- [ ] Compare the simple CNN against a stronger model.
- [ ] Try a ResNet-style model or CoAtNet-style model.
- [ ] Keep the same train/test splits.
- [ ] Save the same probability output format.
- [ ] Compare accuracy, training time, and confusion patterns.

Done when: the project knows whether a stronger acoustic model changes the value of timing fusion.

### 18. Add A Language Model As A Separate Experiment

- [ ] Keep the main fusion experiment language-model-free.
- [ ] Add a separate optional language score for phrase prompts.
- [ ] Evaluate acoustic + timing + language model separately.
- [ ] Do not use language model results to claim timing alone helped.

Done when: language-model gains are measured separately from timing gains.

### 19. Run Ablation Studies

- [ ] Run fusion without dwell time.
- [ ] Run fusion without flight timing.
- [ ] Run fusion without keyboard distance.
- [ ] Run fusion without same-hand features.
- [ ] Run fusion without acoustic confidence scores.
- [ ] Compare each ablation to the full fusion model.

Done when: the project can identify which timing features actually matter.

### 20. Expand Data Collection

- [ ] Add more trials for the first participant.
- [ ] Add more consenting participants.
- [ ] Add a second recording session.
- [ ] Add another microphone position.
- [ ] Add another keyboard only after the first setup is stable.
- [ ] Keep metadata consistent across every new collection.

Done when: results can be tested beyond one tiny pilot dataset.

### 21. Test Generalization

- [ ] Train and test on the same user.
- [ ] Train on one session and test on another session.
- [ ] Train on some users and test on an unseen user.
- [ ] Train on one microphone position and test on another.
- [ ] Train on one keyboard only after enough data exists, then test transfer carefully.

Done when: the project reports what transfers and what stays setup-specific.

### 22. Test Defenses

- [ ] Test microphone distance changes.
- [ ] Test background noise.
- [ ] Test fake keystroke sounds.
- [ ] Test keyboard dampening.
- [ ] Test faster and slower typing styles.
- [ ] Compare acoustic-only and fusion performance under each defense.

Done when: the project includes practical mitigation results, not only attack results.

### 23. Write The Final Report

- [ ] Describe the research question.
- [ ] Describe the consent-based data collection setup.
- [ ] Describe the acoustic model.
- [ ] Describe the timing features.
- [ ] Describe the fusion decoder.
- [ ] Report acoustic-only, timing-only, fusion, and optional language-model results.
- [ ] Include ablation results.
- [ ] Include limitations.
- [ ] Include defenses.
- [ ] Include ethical boundaries.

Done when: someone can read the report and understand what was tested, what worked, what failed, and what should be tried next.

### 24. Package The Project

- [ ] Update `README.md` with setup instructions.
- [ ] Update `README.md` with the project workflow.
- [ ] Make sure all scripts have clear names.
- [ ] Check for stale duplicate scripts, generated package metadata, and outdated documentation.
- [ ] Make sure raw data is not tracked.
- [ ] Run tests.
- [ ] Commit the final working state.
- [ ] Push to GitHub.

Done when: the repository is clean, reproducible, and ready to share privately.

## Current Next Step

- [ ] Start with Step 9: build log-mel spectrogram generation from extracted clips.
