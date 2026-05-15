from __future__ import annotations

import csv

import numpy as np

from keyboard_fusion.acoustic_model import split_train_test_indices, train_acoustic_baseline


def write_test_spectrogram(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrogram = np.full((4, 3), value, dtype=np.float32)
    np.savez_compressed(path, spectrogram=spectrogram)


def test_split_keeps_single_example_labels_in_training() -> None:
    labels = np.array(["a", "a", "b", "b", "rare"])

    train_indices, test_indices, rare_labels = split_train_test_indices(
        labels=labels,
        test_size=0.5,
        random_seed=42,
    )

    assert rare_labels == ["rare"]
    assert 4 in train_indices
    assert 4 not in test_indices
    assert len(test_indices) == 2


def test_train_acoustic_baseline_writes_model_outputs(tmp_path) -> None:
    rows = []
    labels = ["a", "a", "b", "b", "c", "c"]
    for index, label in enumerate(labels):
        spectrogram_path = tmp_path / "spectrograms" / f"clip_{index}.npz"
        write_test_spectrogram(spectrogram_path, value=float(index))
        rows.append(
            {
                "spectrogram_id": f"clip_{index}_logmel",
                "spectrogram_path": str(spectrogram_path),
                "clip_id": f"clip_{index}",
                "session_id": "session_test",
                "trial_id": f"trial_{index // 2:03d}",
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

    outputs = train_acoustic_baseline(
        spectrogram_manifest_path=manifest_path,
        output_root=tmp_path / "models",
        test_size=0.5,
        random_seed=42,
        max_iter=500,
    )

    assert outputs.model_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.predictions_path.exists()
    assert outputs.probabilities_path.exists()
    assert outputs.report_path.exists()
    assert outputs.metrics["train_count"] == 3
    assert outputs.metrics["test_count"] == 3
    assert outputs.metrics["class_count"] == 3

    with outputs.predictions_path.open(newline="", encoding="utf-8") as file:
        prediction_rows = list(csv.DictReader(file))
    with outputs.probabilities_path.open(newline="", encoding="utf-8") as file:
        probability_rows = list(csv.DictReader(file))

    assert len(prediction_rows) == outputs.metrics["test_count"]
    assert len(probability_rows) == outputs.metrics["test_count"] * outputs.metrics["class_count"]
    assert {"true_key", "predicted_key", "top1_key", "top5_key"} <= set(prediction_rows[0])
    assert {"candidate_key", "probability"} <= set(probability_rows[0])
