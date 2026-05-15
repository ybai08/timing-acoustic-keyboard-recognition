from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from keyboard_fusion.acoustic_cnn import build_acoustic_cnn_model, resolve_device
from keyboard_fusion.paths import MODELS_DIR
from keyboard_fusion.segmentation import DetectedPeak, detect_keystroke_peaks, extract_fixed_window
from keyboard_fusion.spectrograms import log_mel_spectrogram, normalize_spectrogram


DEFAULT_MODEL_DIR = MODELS_DIR / "acoustic_cnn" / "all_sessions"


@dataclass(frozen=True)
class PredictionResult:
    predicted_text: str
    events: list[dict[str, Any]]
    detected_count: int
    audio_seconds: float
    sample_rate: int
    model_dir: str
    class_count: int
    segmentation_method: str


def read_wav_bytes_mono_float(raw_wav: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(raw_wav), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV audio is supported.")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return sample_rate, samples / 32768.0


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


def fit_spectrogram_shape(spectrogram: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = expected_shape
    output = np.zeros((rows, cols), dtype=np.float32)
    copy_rows = min(rows, spectrogram.shape[0])
    copy_cols = min(cols, spectrogram.shape[1])
    output[:copy_rows, :copy_cols] = spectrogram[:copy_rows, :copy_cols]
    return output


def key_to_text(key: str) -> str:
    if key == "Space":
        return " "
    if len(key) == 1:
        return key
    return ""


def display_key(key: str) -> str:
    return "Space" if key == "Space" else key


class AcousticCNNPredictor:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        checkpoint_path = self.model_dir / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing acoustic CNN checkpoint: {checkpoint_path}")

        import torch

        self.torch = torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.class_names = [str(value) for value in checkpoint["class_names"]]
        self.feature_shape = tuple(int(value) for value in checkpoint["feature_shape"])
        if len(self.feature_shape) != 2:
            raise ValueError(f"Expected 2D CNN feature shape, got {self.feature_shape}")

        training_config = checkpoint.get("training_config", {})
        dropout = float(training_config.get("dropout", 0.4))
        self.device = resolve_device(device)
        self.model = build_acoustic_cnn_model(
            self.feature_shape,
            class_count=len(self.class_names),
            dropout=dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.normalization_mean = float(checkpoint.get("normalization_mean", 0.0))
        self.normalization_std = float(checkpoint.get("normalization_std", 1.0)) or 1.0

    def spectrogram_from_clip(
        self,
        clip_samples: np.ndarray,
        sample_rate: int,
        mel_bands: int,
        fft_window_size: int,
        hop_length: int,
    ) -> np.ndarray:
        raw_log_mel = log_mel_spectrogram(
            samples=clip_samples,
            sample_rate=sample_rate,
            mel_bands=mel_bands,
            fft_window_size=fft_window_size,
            hop_length=hop_length,
        )
        normalized, _, _ = normalize_spectrogram(raw_log_mel)
        fitted = fit_spectrogram_shape(normalized, self.feature_shape)
        return ((fitted - self.normalization_mean) / self.normalization_std).astype(np.float32)

    def predict_spectrograms(self, spectrograms: list[np.ndarray], top_k: int = 5) -> list[list[dict[str, Any]]]:
        if not spectrograms:
            return []

        x = np.stack(spectrograms).astype(np.float32)
        tensor = self.torch.tensor(x[:, None, :, :], dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(tensor), dim=1).cpu().numpy()

        top_count = min(max(1, int(top_k)), len(self.class_names))
        results: list[list[dict[str, Any]]] = []
        for row in probabilities:
            order = np.argsort(row)[::-1][:top_count]
            results.append(
                [
                    {
                        "key": display_key(self.class_names[index]),
                        "probability": round(float(row[index]), 6),
                    }
                    for index in order
                ]
            )
        return results

    def predict_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
        target_sample_rate: int,
        mel_bands: int,
        fft_window_size: int,
        hop_length: int,
        pre_ms: float,
        post_ms: float,
        sensitivity: float = 1.0,
        min_gap_ms: float = 38.0,
        max_events: int = 80,
        top_k: int = 5,
        peaks: list[DetectedPeak] | None = None,
        segmentation_method: str = "heuristic_detector",
    ) -> PredictionResult:
        model_samples = resample_linear(samples, sample_rate, target_sample_rate)
        if peaks is None:
            peaks = detect_keystroke_peaks(
                model_samples,
                sample_rate=target_sample_rate,
                sensitivity=sensitivity,
                min_gap_ms=min_gap_ms,
                max_peaks=max_events,
            )
        else:
            peaks = [
                DetectedPeak(
                    sample_index=int(round(peak.time_seconds * target_sample_rate)),
                    time_seconds=round(float(peak.time_seconds), 6),
                    strength=float(peak.strength),
                    threshold_ratio=float(peak.threshold_ratio),
                )
                for peak in peaks[:max_events]
            ]

        spectrograms: list[np.ndarray] = []
        for peak in peaks:
            clip = extract_fixed_window(
                model_samples,
                center_sample=peak.sample_index,
                sample_rate=target_sample_rate,
                pre_ms=pre_ms,
                post_ms=post_ms,
            )
            spectrograms.append(
                self.spectrogram_from_clip(
                    clip_samples=clip,
                    sample_rate=target_sample_rate,
                    mel_bands=mel_bands,
                    fft_window_size=fft_window_size,
                    hop_length=hop_length,
                )
            )

        top_predictions = self.predict_spectrograms(spectrograms, top_k=top_k)
        events = build_event_payload(peaks, top_predictions)
        predicted_text = "".join(key_to_text(event["top"][0]["key"]) for event in events if event.get("top"))
        return PredictionResult(
            predicted_text=predicted_text,
            events=events,
            detected_count=len(events),
            audio_seconds=round(float(samples.size) / float(sample_rate), 6) if sample_rate else 0.0,
            sample_rate=target_sample_rate,
            model_dir=str(self.model_dir),
            class_count=len(self.class_names),
            segmentation_method=segmentation_method,
        )


def build_event_payload(
    peaks: list[DetectedPeak],
    top_predictions: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, (peak, top) in enumerate(zip(peaks, top_predictions), start=1):
        predicted_key = top[0]["key"] if top else ""
        events.append(
            {
                "index": index,
                "time_seconds": peak.time_seconds,
                "sample_index": peak.sample_index,
                "strength": round(float(peak.strength), 9),
                "threshold_ratio": peak.threshold_ratio,
                "predicted_key": predicted_key,
                "confidence": top[0]["probability"] if top else 0.0,
                "top": top,
            }
        )
    return events
