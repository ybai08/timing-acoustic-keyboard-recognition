from __future__ import annotations

import csv

import numpy as np
import pytest

from keyboard_fusion.acoustic_cnn import train_acoustic_cnn
from keyboard_fusion.acoustic_model import train_acoustic_baseline
from keyboard_fusion.acoustic_visualization import build_visualization_payload, generate_acoustic_visualization


def write_test_spectrogram(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrogram = np.full((4, 3), value, dtype=np.float32)
    np.savez_compressed(path, spectrogram=spectrogram)


def write_cnn_test_spectrogram(path, label_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectrogram = np.zeros((64, 10), dtype=np.float32)
    start = (label_index * 12) % 56
    spectrogram[start : start + 8, 3:7] = 3.0 + label_index
    np.savez_compressed(path, spectrogram=spectrogram)


def train_tiny_model(tmp_path):
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

    return train_acoustic_baseline(
        spectrogram_manifest_path=manifest_path,
        output_root=tmp_path / "models",
        test_size=0.5,
        random_seed=42,
        max_iter=500,
    )


def train_tiny_cnn(tmp_path):
    pytest.importorskip("torch")
    rows = []
    labels = ["a", "b", "c"] * 6
    for index, label in enumerate(labels):
        spectrogram_path = tmp_path / "cnn_spectrograms" / f"clip_{index}.npz"
        write_cnn_test_spectrogram(spectrogram_path, label_index=ord(label) - ord("a"))
        rows.append(
            {
                "spectrogram_id": f"clip_{index}_logmel",
                "spectrogram_path": str(spectrogram_path),
                "clip_id": f"clip_{index}",
                "session_id": "session_cnn_test",
                "trial_id": f"trial_{index // 3:03d}",
                "event_index": str(index),
                "key": label,
            }
        )

    manifest_path = tmp_path / "cnn_spectrograms" / "session_cnn_test" / "spectrogram_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return train_acoustic_cnn(
        spectrogram_manifest_path=manifest_path,
        output_root=tmp_path / "cnn_models",
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


def test_visualization_payload_describes_baseline_structure(tmp_path) -> None:
    outputs = train_tiny_model(tmp_path)

    payload = build_visualization_payload(outputs.output_dir)

    assert payload["architecture"]["isNeuralNetwork"] is False
    assert payload["architecture"]["hiddenNeurons"] == 0
    assert payload["architecture"]["inputFeatures"] == 12
    assert payload["architecture"]["outputClasses"] == 3
    assert payload["architecture"]["trainableParameters"] == 39
    assert len(payload["weightHeatmaps"]) == 3
    assert len(payload["predictions"]) == outputs.metrics["test_count"]


def test_visualization_payload_describes_cnn_structure(tmp_path) -> None:
    outputs = train_tiny_cnn(tmp_path)

    payload = build_visualization_payload(outputs.output_dir)

    assert payload["architecture"]["isNeuralNetwork"] is True
    assert payload["architecture"]["modelKind"] == "cnn"
    assert payload["architecture"]["outputClasses"] == 3
    assert payload["architecture"]["trainableParameters"] > 0
    assert payload["weightHeatmaps"] == []
    assert len(payload["history"]) == outputs.metrics["epochs_ran"]
    assert len(payload["predictions"]) == outputs.metrics["test_count"]


def test_generate_acoustic_visualization_writes_html(tmp_path) -> None:
    outputs = train_tiny_model(tmp_path)

    html_path = generate_acoustic_visualization(outputs.output_dir)

    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Acoustic Model Viewer" in html
    assert 'id="visualization-data"' in html
    assert "Per-Key Weight Heatmap" in html


def test_generate_cnn_visualization_writes_html(tmp_path) -> None:
    outputs = train_tiny_cnn(tmp_path)

    html_path = generate_acoustic_visualization(outputs.output_dir)

    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Acoustic Model Viewer" in html
    assert "CNN Training History" in html
