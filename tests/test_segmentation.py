from __future__ import annotations

import numpy as np

from keyboard_fusion.segmentation import DetectedPeak, detect_keystroke_peaks, extract_fixed_window
from keyboard_fusion.segmentation_evaluation import match_peaks_to_oracle


def test_detect_keystroke_peaks_finds_synthetic_clicks() -> None:
    sample_rate = 1000
    samples = np.zeros(sample_rate, dtype=np.float32)
    samples[100] = 1.0
    samples[300] = -1.0
    samples[700] = 0.9

    peaks = detect_keystroke_peaks(
        samples,
        sample_rate=sample_rate,
        sensitivity=1.0,
        min_gap_ms=50,
        max_peaks=10,
    )

    assert [round(peak.time_seconds, 1) for peak in peaks] == [0.1, 0.3, 0.7]
    assert all(peak.threshold_ratio >= 1.0 for peak in peaks)


def test_detect_keystroke_peaks_ignores_silence() -> None:
    peaks = detect_keystroke_peaks(
        np.zeros(1000, dtype=np.float32),
        sample_rate=1000,
    )

    assert peaks == []


def test_detect_keystroke_peaks_respects_max_peaks() -> None:
    sample_rate = 1000
    samples = np.zeros(sample_rate, dtype=np.float32)
    samples[100] = 0.2
    samples[300] = 1.0
    samples[700] = 0.8

    peaks = detect_keystroke_peaks(
        samples,
        sample_rate=sample_rate,
        sensitivity=1.0,
        min_gap_ms=50,
        max_peaks=2,
    )

    assert [round(peak.time_seconds, 1) for peak in peaks] == [0.3, 0.7]


def test_extract_fixed_window_pads_edges() -> None:
    samples = np.arange(10, dtype=np.float32)

    window = extract_fixed_window(
        samples,
        center_sample=1,
        sample_rate=1000,
        pre_ms=5,
        post_ms=5,
    )

    assert len(window) == 10
    assert np.all(window[:4] == 0.0)
    assert np.array_equal(window[4:], np.arange(6, dtype=np.float32))


def test_match_peaks_to_oracle_is_one_to_one() -> None:
    truth = [
        {"event_index": 1, "key": "a", "audio_time_seconds": 0.100},
        {"event_index": 2, "key": "b", "audio_time_seconds": 0.200},
    ]
    detected = [
        # Both of these are close to the first oracle event; only the closest
        # should match, leaving the other as a false positive.
        DetectedPeak(sample_index=101, time_seconds=0.101, strength=1.0, threshold_ratio=2.0),
        DetectedPeak(sample_index=103, time_seconds=0.103, strength=0.8, threshold_ratio=1.8),
        DetectedPeak(sample_index=199, time_seconds=0.199, strength=0.9, threshold_ratio=1.9),
    ]

    matches, false_positives, false_negatives = match_peaks_to_oracle(
        truth,
        detected,
        tolerance_seconds=0.01,
    )

    assert len(matches) == 2
    assert len(false_positives) == 1
    assert false_negatives == []
