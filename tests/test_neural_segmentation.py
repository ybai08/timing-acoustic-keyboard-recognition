from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from keyboard_fusion.neural_segmentation import (
    SegmenterTrial,
    build_neural_segmenter_model,
    build_window_examples,
    extract_window_features,
    probabilities_to_peaks,
)


def test_extract_window_features_returns_two_channels() -> None:
    samples = np.zeros(100, dtype=np.float32)
    samples[50] = 1.0

    features = extract_window_features(samples, center_sample=50, window_samples=20)

    assert features.shape == (2, 20)
    assert features.dtype == np.float32
    assert np.max(features[1]) > 0.0


def test_probabilities_to_peaks_applies_threshold_gap_and_cap() -> None:
    centers = np.array([100, 120, 300, 700], dtype=np.int64)
    probabilities = np.array([0.9, 0.8, 0.4, 0.7], dtype=np.float32)

    peaks = probabilities_to_peaks(
        centers=centers,
        probabilities=probabilities,
        sample_rate=1000,
        threshold=0.5,
        min_gap_ms=50,
        max_peaks=2,
    )

    assert [peak.sample_index for peak in peaks] == [100, 700]


def test_build_window_examples_creates_positive_and_negative_examples() -> None:
    trial = SegmenterTrial(
        alignment_path=Path("trial_alignment.json"),
        session_id="session_test",
        trial_id="trial_001",
        audio_path=Path("trial.wav"),
        sample_rate=1000,
        samples=np.zeros(1000, dtype=np.float32),
        event_samples=np.array([100, 500], dtype=np.int64),
    )

    examples = build_window_examples(
        trials=[trial],
        trial_indices=[0],
        random_seed=1,
        negative_ratio=1.0,
        positive_jitters_per_event=0,
        negative_exclusion_ms=80,
    )

    labels = [example.label for example in examples]
    assert labels.count(1) == 2
    assert labels.count(0) == 2


def test_build_neural_segmenter_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    model = build_neural_segmenter_model(window_samples=128)
    x = torch.zeros((3, 2, 128), dtype=torch.float32)

    logits = model(x)

    assert tuple(logits.shape) == (3,)
