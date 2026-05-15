from __future__ import annotations

import csv

import numpy as np
import pytest

from keyboard_fusion.acoustic_cnn import split_train_validation_indices, train_acoustic_cnn


torch = pytest.importorskip("torch")


def write_test_spectrogram(path, label_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrogram = np.zeros((64, 10), dtype=np.float32)
    start = (label_index * 12) % 56
    spectrogram[start : start + 8, 3:7] = 3.0 + label_index
    spectrogram += np.random.default_rng(label_index).normal(0.0, 0.02, size=spectrogram.shape).astype(np.float32)
    np.savez_compressed(path, spectrogram=spectrogram)


def test_split_validation_keeps_single_example_labels_in_training() -> None:
    labels = np.array(["a", "a", "b", "b", "rare"])

    train_indices, validation_indices, rare_labels = split_train_validation_indices(
        labels=labels,
        validation_size=0.5,
        random_seed=42,
    )

    assert rare_labels == ["rare"]
    assert 4 in train_indices
    assert 4 not in validation_indices
    assert len(validation_indices) == 2


def test_train_acoustic_cnn_writes_model_outputs(tmp_path) -> None:
    rows = []
    labels = ["a", "b", "c"] * 6
    for index, label in enumerate(labels):
        spectrogram_path = tmp_path / "spectrograms" / f"clip_{index}.npz"
        write_test_spectrogram(spectrogram_path, label_index=ord(label) - ord("a"))
        rows.append(
            {
                "spectrogram_id": f"clip_{index}_logmel",
                "spectrogram_path": str(spectrogram_path),
                "clip_id": f"clip_{index}",
                "session_id": "session_test",
                "trial_id": f"trial_{index // 3:03d}",
                "event_index": str(index),
                "key": label,
            }
        )

    manifest_path = tmp_path / "spectrograms" / "session_test" / "spectrogram_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    outputs = train_acoustic_cnn(
        spectrogram_manifest_path=manifest_path,
        output_root=tmp_path / "models",
        test_size=0.25,
        validation_size=0.25,
        random_seed=42,
        epochs=2,
        batch_size=6,
        learning_rate=0.001,
        patience=4,
        device="cpu",
        mixup_alpha=0.0,
        frequency_mask_width=0,
        time_mask_width=0,
        noise_std=0.0,
    )

    assert outputs.model_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.predictions_path.exists()
    assert outputs.probabilities_path.exists()
    assert outputs.history_path.exists()
    assert outputs.report_path.exists()
    assert outputs.metrics["model_type"] == "acoustic_spectrogram_resnet"
    assert outputs.metrics["class_count"] == 3
    assert outputs.metrics["architecture"]["trainable_parameters"] > 0

    with outputs.predictions_path.open(newline="", encoding="utf-8") as file:
        prediction_rows = list(csv.DictReader(file))
    with outputs.probabilities_path.open(newline="", encoding="utf-8") as file:
        probability_rows = list(csv.DictReader(file))

    assert len(prediction_rows) == outputs.metrics["test_count"]
    assert len(probability_rows) == outputs.metrics["test_count"] * outputs.metrics["class_count"]
