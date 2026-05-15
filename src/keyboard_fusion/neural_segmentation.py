from __future__ import annotations

import csv
import json
import math
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from keyboard_fusion.acoustic_cnn import resolve_device
from keyboard_fusion.paths import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from keyboard_fusion.segmentation import DetectedPeak, extract_fixed_window
from keyboard_fusion.spectrograms import read_wav_mono_float


DEFAULT_SEGMENTER_DIR = MODELS_DIR / "neural_segmenter" / "all_sessions"
DEFAULT_SEGMENTER_SESSION_ID = "all_sessions"

HISTORY_COLUMNS = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"]
EVENT_COLUMNS = [
    "session_id",
    "trial_id",
    "audio_path",
    "true_key_count",
    "detected_count",
    "matched_count",
    "false_positive_count",
    "false_negative_count",
    "precision",
    "recall",
    "f1",
]
CLIP_MANIFEST_COLUMNS = [
    "clip_id",
    "clip_audio_path",
    "source_audio_path",
    "detected_index",
    "detected_time_seconds",
    "detected_sample_index",
    "peak_probability",
    "sample_rate",
    "window_start_seconds",
    "window_end_seconds",
]


@dataclass(frozen=True)
class SegmenterTrial:
    alignment_path: Path
    session_id: str
    trial_id: str
    audio_path: Path
    sample_rate: int
    samples: np.ndarray
    event_samples: np.ndarray

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.samples.size) / float(self.sample_rate)


@dataclass(frozen=True)
class WindowExample:
    trial_index: int
    center_sample: int
    label: int


@dataclass(frozen=True)
class SegmenterTrainingOutputs:
    output_dir: Path
    model_path: Path
    metrics_path: Path
    history_path: Path
    event_predictions_path: Path
    report_path: Path
    metrics: dict[str, Any]


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for the neural segmenter. Install it with "
            "`python -m pip install -r requirements-ml.txt`."
        ) from exc
    return torch, nn, F, DataLoader


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate and target_rate must be positive")

    duration_seconds = samples.size / float(source_rate)
    target_count = max(1, int(round(duration_seconds * target_rate)))
    source_positions = np.arange(samples.size, dtype=np.float32) / float(source_rate)
    target_positions = np.arange(target_count, dtype=np.float32) / float(target_rate)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def alignment_audio_path(alignment: dict[str, Any], raw_sessions_dir: Path | None = None) -> Path:
    raw_root = raw_sessions_dir or RAW_DATA_DIR / "sessions"
    return raw_root / str(alignment["session_id"]) / str(alignment["audio_file_path"])


def load_alignment(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_samples_from_alignment(alignment: dict[str, Any], sample_rate: int) -> np.ndarray:
    samples: list[int] = []
    for event in alignment.get("keydown_alignments", []):
        if not event.get("within_audio", True):
            continue
        audio_time = float(event["audio_time_seconds"])
        sample_index = int(round(audio_time * sample_rate))
        if sample_index >= 0:
            samples.append(sample_index)
    return np.array(sorted(samples), dtype=np.int64)


def load_segmenter_trials(
    alignment_paths: list[Path],
    target_sample_rate: int = 48000,
    raw_sessions_dir: Path | None = None,
) -> list[SegmenterTrial]:
    trials: list[SegmenterTrial] = []
    for alignment_path in sorted(alignment_paths):
        alignment = load_alignment(alignment_path)
        audio_path = alignment_audio_path(alignment, raw_sessions_dir=raw_sessions_dir)
        sample_rate, samples = read_wav_mono_float(audio_path)
        if target_sample_rate and sample_rate != target_sample_rate:
            samples = resample_linear(samples, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        event_samples = event_samples_from_alignment(alignment, sample_rate)
        event_samples = event_samples[event_samples < samples.size]
        if event_samples.size == 0:
            continue
        trials.append(
            SegmenterTrial(
                alignment_path=alignment_path,
                session_id=str(alignment["session_id"]),
                trial_id=str(alignment["trial_id"]),
                audio_path=audio_path,
                sample_rate=sample_rate,
                samples=samples.astype(np.float32),
                event_samples=event_samples,
            )
        )
    if not trials:
        raise ValueError("No usable aligned trials were found for neural segmentation training.")
    return trials


def milliseconds_to_samples(sample_rate: int, milliseconds: float) -> int:
    return max(1, int(round(sample_rate * milliseconds / 1000.0)))


def extract_window_features(samples: np.ndarray, center_sample: int, window_samples: int) -> np.ndarray:
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")

    pre_samples = window_samples // 2
    start = int(center_sample) - pre_samples
    end = start + window_samples
    window = np.zeros(window_samples, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(int(samples.size), end)
    if source_end > source_start:
        target_start = source_start - start
        target_end = target_start + (source_end - source_start)
        window[target_start:target_end] = samples[source_start:source_end].astype(np.float32)

    centered = window - float(np.mean(window))
    raw_std = float(np.std(centered))
    if raw_std > 1e-8:
        centered = centered / raw_std

    transient = np.abs(np.diff(window, prepend=window[0])).astype(np.float32)
    transient = transient - float(np.mean(transient))
    transient_std = float(np.std(transient))
    if transient_std > 1e-8:
        transient = transient / transient_std

    return np.stack([centered, transient]).astype(np.float32)


def far_from_events(center_sample: int, event_samples: np.ndarray, exclusion_samples: int) -> bool:
    if event_samples.size == 0:
        return True
    return bool(np.min(np.abs(event_samples - int(center_sample))) >= exclusion_samples)


def build_window_examples(
    trials: list[SegmenterTrial],
    trial_indices: list[int],
    random_seed: int,
    negative_ratio: float = 1.5,
    positive_jitters_per_event: int = 1,
    positive_jitter_ms: float = 3.0,
    negative_exclusion_ms: float = 45.0,
) -> list[WindowExample]:
    rng = random.Random(random_seed)
    examples: list[WindowExample] = []

    for trial_index in trial_indices:
        trial = trials[trial_index]
        jitter_samples = milliseconds_to_samples(trial.sample_rate, positive_jitter_ms)
        exclusion_samples = milliseconds_to_samples(trial.sample_rate, negative_exclusion_ms)
        positive_count = 0

        for event_sample in trial.event_samples:
            centers = [int(event_sample)]
            for _ in range(max(0, positive_jitters_per_event)):
                centers.append(int(event_sample) + rng.randint(-jitter_samples, jitter_samples))
            for center in centers:
                center = max(0, min(int(center), int(trial.samples.size) - 1))
                examples.append(WindowExample(trial_index=trial_index, center_sample=center, label=1))
                positive_count += 1

        negative_target = max(1, int(math.ceil(positive_count * negative_ratio)))
        attempts = 0
        negatives = 0
        while negatives < negative_target and attempts < negative_target * 200:
            attempts += 1
            center = rng.randint(0, max(0, int(trial.samples.size) - 1))
            if not far_from_events(center, trial.event_samples, exclusion_samples):
                continue
            examples.append(WindowExample(trial_index=trial_index, center_sample=center, label=0))
            negatives += 1

    rng.shuffle(examples)
    return examples


class KeystrokeWindowDataset:
    def __init__(self, trials: list[SegmenterTrial], examples: list[WindowExample], window_samples: int) -> None:
        self.trials = trials
        self.examples = examples
        self.window_samples = int(window_samples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        example = self.examples[index]
        trial = self.trials[example.trial_index]
        features = extract_window_features(trial.samples, example.center_sample, self.window_samples)
        return features, np.float32(example.label)


def build_neural_segmenter_model(window_samples: int, dropout: float = 0.25) -> Any:
    _, nn, _, _ = _require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=9, padding=4, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, kernel_size=9, padding=4, bias=False),
                nn.BatchNorm1d(channels),
            )
            self.activation = nn.GELU()

        def forward(self, x: Any) -> Any:
            return self.activation(x + self.net(x))

    class NeuralKeystrokeSegmenterCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(2, 32, stride=2),
                ResidualBlock(32),
                ConvBlock(32, 64, stride=2),
                ResidualBlock(64),
                ConvBlock(64, 96, stride=2),
                ResidualBlock(96),
                ConvBlock(96, 128, stride=2),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.LayerNorm(64),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x)).squeeze(1)

    model = NeuralKeystrokeSegmenterCNN()
    model.input_shape = [2, int(window_samples)]
    return model


def count_trainable_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def architecture_description(window_samples: int, sample_rate: int, trainable_parameters: int) -> dict[str, Any]:
    return {
        "model_name": "NeuralKeystrokeSegmenterCNN",
        "input_shape": [2, int(window_samples)],
        "input_seconds": round(float(window_samples) / float(sample_rate), 6) if sample_rate else 0.0,
        "output": "one logit: probability that a keypress is centered in the window",
        "trainable_parameters": trainable_parameters,
        "layers": [
            "2-channel waveform window: normalized raw audio + normalized transient channel",
            "Conv1D 2->32, BatchNorm, GELU",
            "Residual 32-channel temporal block",
            "Conv1D 32->64, BatchNorm, GELU",
            "Residual 64-channel temporal block",
            "Conv1D 64->96, BatchNorm, GELU",
            "Residual 96-channel temporal block",
            "Conv1D 96->128, BatchNorm, GELU",
            "Global average pool",
            "Dropout, Dense 64, GELU, LayerNorm, Dropout",
            "Dense binary keypress-center logit",
        ],
    }


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = probabilities >= threshold
    positives = labels.astype(bool)
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & ~positives))
    false_negative = int(np.sum(~predictions & positives))
    true_negative = int(np.sum(~predictions & ~positives))
    total = max(1, labels.size)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (true_positive + true_negative) / total
    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


def train_epoch(model: Any, loader: Any, optimizer: Any, loss_fn: Any, device: str) -> tuple[float, float]:
    torch, _, _, _ = _require_torch()
    model.train()
    losses: list[float] = []
    correct = 0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device=device, dtype=torch.float32)
        y_batch = y_batch.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        predictions = torch.sigmoid(logits) >= 0.5
        correct += int((predictions == (y_batch >= 0.5)).sum().detach().cpu())
        total += int(y_batch.numel())
    return float(np.mean(losses)) if losses else 0.0, correct / max(1, total)


def evaluate_window_model(model: Any, loader: Any, loss_fn: Any, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    torch, _, _, _ = _require_torch()
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device, dtype=torch.float32)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            losses.append(float(loss.detach().cpu()))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y_batch.detach().cpu().numpy())

    if not probabilities:
        return 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return float(np.mean(losses)), np.concatenate(probabilities), np.concatenate(labels)


def predict_center_probabilities(
    model: Any,
    samples: np.ndarray,
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
    device: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _, _ = _require_torch()
    if samples.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    centers = np.arange(0, int(samples.size), max(1, int(hop_samples)), dtype=np.int64)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start : start + batch_size]
            batch = np.stack(
                [extract_window_features(samples, int(center), window_samples) for center in batch_centers]
            ).astype(np.float32)
            tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            logits = model(tensor)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return centers, np.concatenate(probabilities).astype(np.float32)


def probabilities_to_peaks(
    centers: np.ndarray,
    probabilities: np.ndarray,
    sample_rate: int,
    threshold: float,
    min_gap_ms: float,
    max_peaks: int,
) -> list[DetectedPeak]:
    if centers.size == 0 or probabilities.size == 0 or max_peaks <= 0:
        return []

    candidate_indices = np.flatnonzero(probabilities >= float(threshold))
    if candidate_indices.size == 0:
        return []

    min_gap_samples = milliseconds_to_samples(sample_rate, min_gap_ms)
    selected: list[int] = []
    for probability_index in sorted(candidate_indices, key=lambda index: float(probabilities[index]), reverse=True):
        center = int(centers[probability_index])
        if all(abs(center - int(centers[kept])) >= min_gap_samples for kept in selected):
            selected.append(int(probability_index))
        if len(selected) >= max_peaks:
            break

    peaks: list[DetectedPeak] = []
    for probability_index in sorted(selected, key=lambda index: int(centers[index])):
        center = int(centers[probability_index])
        probability = float(probabilities[probability_index])
        peaks.append(
            DetectedPeak(
                sample_index=center,
                time_seconds=round(center / float(sample_rate), 6),
                strength=probability,
                threshold_ratio=round(probability / max(float(threshold), 1e-12), 6),
            )
        )
    return peaks


def match_peaks_to_events(
    true_samples: np.ndarray,
    peaks: list[DetectedPeak],
    sample_rate: int,
    tolerance_ms: float,
) -> tuple[int, int, int]:
    tolerance_samples = milliseconds_to_samples(sample_rate, tolerance_ms)
    candidates: list[tuple[int, int, int]] = []
    for truth_index, truth_sample in enumerate(true_samples):
        for peak_index, peak in enumerate(peaks):
            error = abs(int(peak.sample_index) - int(truth_sample))
            if error <= tolerance_samples:
                candidates.append((error, truth_index, peak_index))
    candidates.sort(key=lambda item: item[0])

    used_truth: set[int] = set()
    used_peaks: set[int] = set()
    for _, truth_index, peak_index in candidates:
        if truth_index in used_truth or peak_index in used_peaks:
            continue
        used_truth.add(truth_index)
        used_peaks.add(peak_index)

    matched = len(used_truth)
    false_positive = len(peaks) - len(used_peaks)
    false_negative = len(true_samples) - len(used_truth)
    return matched, false_positive, false_negative


def event_metrics(matched: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = matched / max(1, matched + false_positive)
    recall = matched / max(1, matched + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


def evaluate_event_detection(
    model: Any,
    trials: list[SegmenterTrial],
    trial_indices: list[int],
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
    threshold: float,
    min_gap_ms: float,
    max_peak_multiplier: float,
    tolerance_ms: float,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total_matched = 0
    total_false_positive = 0
    total_false_negative = 0
    total_true = 0
    total_detected = 0

    for trial_index in trial_indices:
        trial = trials[trial_index]
        max_peaks = max(1, int(math.ceil(len(trial.event_samples) * max_peak_multiplier)))
        centers, probabilities = predict_center_probabilities(
            model=model,
            samples=trial.samples,
            sample_rate=sample_rate,
            window_samples=window_samples,
            hop_samples=hop_samples,
            device=device,
            batch_size=batch_size,
        )
        peaks = probabilities_to_peaks(
            centers=centers,
            probabilities=probabilities,
            sample_rate=sample_rate,
            threshold=threshold,
            min_gap_ms=min_gap_ms,
            max_peaks=max_peaks,
        )
        matched, false_positive, false_negative = match_peaks_to_events(
            true_samples=trial.event_samples,
            peaks=peaks,
            sample_rate=sample_rate,
            tolerance_ms=tolerance_ms,
        )
        metrics = event_metrics(matched, false_positive, false_negative)
        total_matched += matched
        total_false_positive += false_positive
        total_false_negative += false_negative
        total_true += int(len(trial.event_samples))
        total_detected += int(len(peaks))
        rows.append(
            {
                "session_id": trial.session_id,
                "trial_id": trial.trial_id,
                "audio_path": str(trial.audio_path),
                "true_key_count": int(len(trial.event_samples)),
                "detected_count": int(len(peaks)),
                "matched_count": matched,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
                **metrics,
            }
        )

    summary = {
        "true_key_count": total_true,
        "detected_count": total_detected,
        "matched_count": total_matched,
        "false_positive_count": total_false_positive,
        "false_negative_count": total_false_negative,
        **event_metrics(total_matched, total_false_positive, total_false_negative),
    }
    return summary, rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    write_csv_rows(path, history, HISTORY_COLUMNS)


def build_text_report(metrics: dict[str, Any]) -> str:
    lines = [
        "Neural Keystroke Segmenter Report",
        "=================================",
        "",
        f"Session ID: {metrics['session_id']}",
        f"Device: {metrics['device']}",
        f"Trials: train={metrics['train_trial_count']}, validation={metrics['validation_trial_count']}, test={metrics['test_trial_count']}",
        f"Window examples: train={metrics['train_window_count']}, validation={metrics['validation_window_count']}, test={metrics['test_window_count']}",
        f"Best epoch: {metrics['best_epoch']} / {metrics['epochs_ran']}",
        f"Parameters: {metrics['architecture']['trainable_parameters']:,}",
        "",
        "Window Classification",
        "---------------------",
        f"Accuracy:  {metrics['test_window_metrics']['accuracy']:.3f}",
        f"Precision: {metrics['test_window_metrics']['precision']:.3f}",
        f"Recall:    {metrics['test_window_metrics']['recall']:.3f}",
        f"F1:        {metrics['test_window_metrics']['f1']:.3f}",
        "",
        "Event Detection On Held-Out Trials",
        "----------------------------------",
        f"True keydowns:    {metrics['test_event_metrics']['true_key_count']}",
        f"Detected peaks:   {metrics['test_event_metrics']['detected_count']}",
        f"Matched peaks:    {metrics['test_event_metrics']['matched_count']}",
        f"False positives:  {metrics['test_event_metrics']['false_positive_count']}",
        f"False negatives:  {metrics['test_event_metrics']['false_negative_count']}",
        f"Precision:        {metrics['test_event_metrics']['precision']:.3f}",
        f"Recall:           {metrics['test_event_metrics']['recall']:.3f}",
        f"F1:               {metrics['test_event_metrics']['f1']:.3f}",
    ]
    return "\n".join(lines) + "\n"


def split_trial_indices(
    trials: list[SegmenterTrial],
    test_size: float,
    validation_size: float,
    random_seed: int,
) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(len(trials)))
    if len(indices) < 3:
        return indices, [], []

    train_validation, test = train_test_split(
        indices,
        test_size=max(1, int(round(len(indices) * test_size))),
        random_state=random_seed,
        shuffle=True,
    )
    if len(train_validation) < 3 or validation_size <= 0:
        return sorted(train_validation), [], sorted(test)

    validation_count = max(1, int(round(len(train_validation) * validation_size)))
    validation_count = min(validation_count, len(train_validation) - 1)
    train, validation = train_test_split(
        train_validation,
        test_size=validation_count,
        random_state=random_seed + 1,
        shuffle=True,
    )
    return sorted(train), sorted(validation), sorted(test)


def train_neural_segmenter(
    alignment_paths: list[Path],
    output_root: Path | None = None,
    output_session_id: str = DEFAULT_SEGMENTER_SESSION_ID,
    target_sample_rate: int = 24000,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
    epochs: int = 35,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 8,
    window_ms: float = 75.0,
    hop_ms: float = 5.0,
    min_gap_ms: float = 75.0,
    threshold: float = 0.5,
    tolerance_ms: float = 35.0,
    max_peak_multiplier: float = 1.0,
    negative_ratio: float = 1.5,
    positive_jitters_per_event: int = 1,
    positive_jitter_ms: float = 3.0,
    negative_exclusion_ms: float = 45.0,
    dropout: float = 0.25,
    device: str = "auto",
) -> SegmenterTrainingOutputs:
    torch, _, _, DataLoader = _require_torch()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    requested_device = device
    device = resolve_device(requested_device)
    trials = load_segmenter_trials(alignment_paths, target_sample_rate=target_sample_rate)
    train_indices, validation_indices, test_indices = split_trial_indices(
        trials=trials,
        test_size=test_size,
        validation_size=validation_size,
        random_seed=random_seed,
    )

    train_examples = build_window_examples(
        trials=trials,
        trial_indices=train_indices,
        random_seed=random_seed,
        negative_ratio=negative_ratio,
        positive_jitters_per_event=positive_jitters_per_event,
        positive_jitter_ms=positive_jitter_ms,
        negative_exclusion_ms=negative_exclusion_ms,
    )
    validation_examples = build_window_examples(
        trials=trials,
        trial_indices=validation_indices,
        random_seed=random_seed + 1,
        negative_ratio=negative_ratio,
        positive_jitters_per_event=positive_jitters_per_event,
        positive_jitter_ms=positive_jitter_ms,
        negative_exclusion_ms=negative_exclusion_ms,
    ) if validation_indices else []
    test_examples = build_window_examples(
        trials=trials,
        trial_indices=test_indices,
        random_seed=random_seed + 2,
        negative_ratio=negative_ratio,
        positive_jitters_per_event=positive_jitters_per_event,
        positive_jitter_ms=positive_jitter_ms,
        negative_exclusion_ms=negative_exclusion_ms,
    ) if test_indices else []

    if not train_examples:
        raise ValueError("No training windows were generated for the neural segmenter.")

    window_samples = milliseconds_to_samples(target_sample_rate, window_ms)
    hop_samples = milliseconds_to_samples(target_sample_rate, hop_ms)
    train_loader = DataLoader(
        KeystrokeWindowDataset(trials, train_examples, window_samples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        KeystrokeWindowDataset(trials, validation_examples or train_examples, window_samples),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        KeystrokeWindowDataset(trials, test_examples or validation_examples or train_examples, window_samples),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = build_neural_segmenter_model(window_samples=window_samples, dropout=dropout).to(device)
    labels = np.array([example.label for example in train_examples], dtype=np.float32)
    positive_count = float(np.sum(labels))
    negative_count = float(len(labels) - positive_count)
    pos_weight = torch.tensor([negative_count / max(1.0, positive_count)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.08)

    best_state: dict[str, Any] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        validation_loss, validation_probabilities, validation_labels = evaluate_window_model(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
        )
        validation_accuracy = binary_metrics(validation_probabilities, validation_labels, threshold=threshold)["accuracy"]
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 9),
                "train_accuracy": round(train_accuracy, 9),
                "validation_loss": round(validation_loss, 9),
                "validation_accuracy": round(validation_accuracy, 9),
                "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 12),
            }
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed_seconds = time.perf_counter() - start_time
    test_loss, test_probabilities, test_labels = evaluate_window_model(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )
    test_window_metrics = binary_metrics(test_probabilities, test_labels, threshold=threshold)
    test_event_metrics, event_rows = evaluate_event_detection(
        model=model,
        trials=trials,
        trial_indices=test_indices or validation_indices or train_indices,
        sample_rate=target_sample_rate,
        window_samples=window_samples,
        hop_samples=hop_samples,
        threshold=threshold,
        min_gap_ms=min_gap_ms,
        max_peak_multiplier=max_peak_multiplier,
        tolerance_ms=tolerance_ms,
        device=device,
        batch_size=batch_size,
    )

    output_base = output_root or MODELS_DIR / "neural_segmenter"
    output_dir = output_base / output_session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "training_history.csv"
    event_predictions_path = output_dir / "test_event_predictions.csv"
    report_path = output_dir / "report.txt"

    training_config = {
        "target_sample_rate": target_sample_rate,
        "test_size": test_size,
        "validation_size": validation_size,
        "random_seed": random_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "window_ms": window_ms,
        "window_samples": window_samples,
        "hop_ms": hop_ms,
        "hop_samples": hop_samples,
        "min_gap_ms": min_gap_ms,
        "threshold": threshold,
        "tolerance_ms": tolerance_ms,
        "max_peak_multiplier": max_peak_multiplier,
        "negative_ratio": negative_ratio,
        "positive_jitters_per_event": positive_jitters_per_event,
        "positive_jitter_ms": positive_jitter_ms,
        "negative_exclusion_ms": negative_exclusion_ms,
        "dropout": dropout,
        "requested_device": requested_device,
        "device": device,
    }
    architecture = architecture_description(
        window_samples=window_samples,
        sample_rate=target_sample_rate,
        trainable_parameters=count_trainable_parameters(model),
    )
    metrics = {
        "model_type": "neural_keystroke_segmenter",
        "session_id": output_session_id,
        "device": device,
        "trial_count": len(trials),
        "train_trial_count": len(train_indices),
        "validation_trial_count": len(validation_indices),
        "test_trial_count": len(test_indices),
        "train_window_count": len(train_examples),
        "validation_window_count": len(validation_examples),
        "test_window_count": len(test_examples),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "test_loss": round(float(test_loss), 9),
        "test_window_metrics": test_window_metrics,
        "test_event_metrics": test_event_metrics,
        "training_config": training_config,
        "architecture": architecture,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": training_config,
            "metrics": metrics,
            "architecture": architecture,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_history(history_path, history)
    write_csv_rows(event_predictions_path, event_rows, EVENT_COLUMNS)
    report_path.write_text(build_text_report(metrics), encoding="utf-8")

    return SegmenterTrainingOutputs(
        output_dir=output_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        history_path=history_path,
        event_predictions_path=event_predictions_path,
        report_path=report_path,
        metrics=metrics,
    )


class NeuralSegmenterPredictor:
    def __init__(self, model_dir: Path = DEFAULT_SEGMENTER_DIR, device: str = "auto") -> None:
        torch, _, _, _ = _require_torch()
        self.torch = torch
        self.model_dir = Path(model_dir)
        checkpoint_path = self.model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing neural segmenter checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.training_config = checkpoint.get("training_config", {})
        self.sample_rate = int(self.training_config.get("target_sample_rate", 24000))
        self.window_samples = int(self.training_config.get("window_samples", milliseconds_to_samples(self.sample_rate, 75.0)))
        self.hop_samples = int(self.training_config.get("hop_samples", milliseconds_to_samples(self.sample_rate, 5.0)))
        self.threshold = float(self.training_config.get("threshold", 0.5))
        self.min_gap_ms = float(self.training_config.get("min_gap_ms", 45.0))
        self.device = resolve_device(device)
        self.model = build_neural_segmenter_model(
            window_samples=self.window_samples,
            dropout=float(self.training_config.get("dropout", 0.25)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def detect_peaks(
        self,
        samples: np.ndarray,
        sample_rate: int,
        threshold: float | None = None,
        min_gap_ms: float | None = None,
        max_peaks: int = 80,
        batch_size: int = 256,
    ) -> list[DetectedPeak]:
        model_samples = resample_linear(samples, sample_rate, self.sample_rate)
        centers, probabilities = predict_center_probabilities(
            model=self.model,
            samples=model_samples,
            sample_rate=self.sample_rate,
            window_samples=self.window_samples,
            hop_samples=self.hop_samples,
            device=self.device,
            batch_size=batch_size,
        )
        return probabilities_to_peaks(
            centers=centers,
            probabilities=probabilities,
            sample_rate=self.sample_rate,
            threshold=self.threshold if threshold is None else threshold,
            min_gap_ms=self.min_gap_ms if min_gap_ms is None else min_gap_ms,
            max_peaks=max_peaks,
        )


def write_wav_mono_float(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def extract_detected_clips(
    samples: np.ndarray,
    sample_rate: int,
    peaks: list[DetectedPeak],
    output_dir: Path,
    source_audio_path: Path,
    pre_ms: float = 20.0,
    post_ms: float = 45.0,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, peak in enumerate(peaks, start=1):
        clip = extract_fixed_window(
            samples=samples,
            center_sample=peak.sample_index,
            sample_rate=sample_rate,
            pre_ms=pre_ms,
            post_ms=post_ms,
        )
        clip_id = f"detected_event_{index:03d}"
        clip_path = output_dir / f"{clip_id}.wav"
        write_wav_mono_float(clip_path, clip, sample_rate)
        start_seconds = max(0.0, peak.time_seconds - pre_ms / 1000.0)
        end_seconds = peak.time_seconds + post_ms / 1000.0
        records.append(
            {
                "clip_id": clip_id,
                "clip_audio_path": str(clip_path),
                "source_audio_path": str(source_audio_path),
                "detected_index": index,
                "detected_time_seconds": peak.time_seconds,
                "detected_sample_index": peak.sample_index,
                "peak_probability": round(float(peak.strength), 9),
                "sample_rate": sample_rate,
                "window_start_seconds": round(start_seconds, 9),
                "window_end_seconds": round(end_seconds, 9),
            }
        )
    write_csv_rows(output_dir / "clip_manifest.csv", records, CLIP_MANIFEST_COLUMNS)
    return records


def segment_audio_file_to_clips(
    audio_path: Path,
    output_dir: Path,
    model_dir: Path = DEFAULT_SEGMENTER_DIR,
    expected_keys: int | None = None,
    threshold: float | None = None,
    pre_ms: float = 20.0,
    post_ms: float = 45.0,
    device: str = "auto",
) -> tuple[list[DetectedPeak], list[dict[str, Any]]]:
    sample_rate, samples = read_wav_mono_float(audio_path)
    predictor = NeuralSegmenterPredictor(model_dir=model_dir, device=device)
    max_peaks = int(expected_keys) if expected_keys is not None else 80
    peaks = predictor.detect_peaks(
        samples=samples,
        sample_rate=sample_rate,
        threshold=threshold,
        max_peaks=max_peaks,
    )
    model_samples = resample_linear(samples, sample_rate, predictor.sample_rate)
    records = extract_detected_clips(
        samples=model_samples,
        sample_rate=predictor.sample_rate,
        peaks=peaks,
        output_dir=output_dir,
        source_audio_path=audio_path,
        pre_ms=pre_ms,
        post_ms=post_ms,
    )
    return peaks, records


__all__ = [
    "DEFAULT_SEGMENTER_DIR",
    "DEFAULT_SEGMENTER_SESSION_ID",
    "NeuralSegmenterPredictor",
    "SegmenterTrainingOutputs",
    "SegmenterTrial",
    "WindowExample",
    "build_neural_segmenter_model",
    "build_window_examples",
    "extract_detected_clips",
    "extract_window_features",
    "load_segmenter_trials",
    "probabilities_to_peaks",
    "segment_audio_file_to_clips",
    "train_neural_segmenter",
]
