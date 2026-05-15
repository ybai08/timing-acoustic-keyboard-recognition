from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectedPeak:
    sample_index: int
    time_seconds: float
    strength: float
    threshold_ratio: float


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.astype(np.float32)
    if values.size == 0:
        return values.astype(np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def click_envelope(samples: np.ndarray, sample_rate: int, smooth_ms: float = 2.5) -> np.ndarray:
    """Build a click-focused amplitude envelope from a mono audio signal."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if samples.size == 0:
        return np.array([], dtype=np.float32)

    centered = samples.astype(np.float32) - float(np.mean(samples))
    transient = np.abs(np.diff(centered, prepend=centered[0])).astype(np.float32)
    window_size = max(1, int(round(sample_rate * smooth_ms / 1000.0)))
    return moving_average(transient, window_size)


def robust_threshold(envelope: np.ndarray, sensitivity: float = 1.0) -> float:
    if envelope.size == 0:
        return 0.0
    safe_sensitivity = max(0.25, min(float(sensitivity), 4.0))
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    robust_sigma = mad * 1.4826
    quantile_floor = float(np.quantile(envelope, 0.86))
    return max(median + (7.0 / safe_sensitivity) * robust_sigma, quantile_floor)


def local_peak_candidates(envelope: np.ndarray, threshold: float) -> np.ndarray:
    if envelope.size < 3:
        return np.array([], dtype=int)

    middle = envelope[1:-1]
    candidates = np.flatnonzero(
        (middle >= envelope[:-2])
        & (middle > envelope[2:])
        & (middle >= threshold)
    ) + 1
    if candidates.size:
        return candidates.astype(int)

    above = envelope >= threshold
    starts = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    ends = np.flatnonzero(above & np.concatenate((~above[1:], [True]))) + 1
    run_peaks = [start + int(np.argmax(envelope[start:end])) for start, end in zip(starts, ends)]
    return np.array(run_peaks, dtype=int)


def suppress_close_peaks(
    candidates: np.ndarray,
    envelope: np.ndarray,
    sample_rate: int,
    min_gap_ms: float,
    max_peaks: int,
) -> list[int]:
    if candidates.size == 0:
        return []

    min_gap_samples = max(1, int(round(sample_rate * min_gap_ms / 1000.0)))
    selected: list[int] = []
    for index in sorted((int(value) for value in candidates), key=lambda item: float(envelope[item]), reverse=True):
        if all(abs(index - kept) >= min_gap_samples for kept in selected):
            selected.append(index)
        if len(selected) >= max_peaks:
            break
    return sorted(selected)


def detect_keystroke_peaks(
    samples: np.ndarray,
    sample_rate: int,
    sensitivity: float = 1.0,
    min_gap_ms: float = 38.0,
    max_peaks: int = 80,
) -> list[DetectedPeak]:
    """Detect likely keystroke click peaks from audio alone.

    This is intentionally a simple first-pass segmenter. It is good enough for
    a local demo, but later evaluation should compare it against oracle
    keydown timestamps before treating its output as research-grade.
    """
    if max_peaks <= 0:
        return []

    envelope = click_envelope(samples, sample_rate)
    if envelope.size == 0 or float(np.max(envelope)) < 1e-8:
        return []
    threshold = robust_threshold(envelope, sensitivity=sensitivity)
    candidates = local_peak_candidates(envelope, threshold)
    selected = suppress_close_peaks(
        candidates=candidates,
        envelope=envelope,
        sample_rate=sample_rate,
        min_gap_ms=min_gap_ms,
        max_peaks=max_peaks,
    )

    peaks: list[DetectedPeak] = []
    for sample_index in selected:
        strength = float(envelope[sample_index])
        peaks.append(
            DetectedPeak(
                sample_index=int(sample_index),
                time_seconds=round(float(sample_index) / float(sample_rate), 6),
                strength=strength,
                threshold_ratio=round(strength / max(threshold, 1e-12), 6),
            )
        )
    return peaks


def extract_fixed_window(
    samples: np.ndarray,
    center_sample: int,
    sample_rate: int,
    pre_ms: float,
    post_ms: float,
) -> np.ndarray:
    pre_samples = int(round(sample_rate * pre_ms / 1000.0))
    post_samples = int(round(sample_rate * post_ms / 1000.0))
    window_length = max(1, pre_samples + post_samples)
    start = int(center_sample) - pre_samples
    end = start + window_length

    output = np.zeros(window_length, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(int(samples.size), end)
    if source_end <= source_start:
        return output

    target_start = source_start - start
    target_end = target_start + (source_end - source_start)
    output[target_start:target_end] = samples[source_start:source_end].astype(np.float32)
    return output
