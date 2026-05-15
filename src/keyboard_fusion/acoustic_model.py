from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from keyboard_fusion.paths import MODELS_DIR, PROCESSED_DATA_DIR


PREDICTION_COLUMNS = [
    "clip_id",
    "spectrogram_path",
    "trial_id",
    "event_index",
    "true_key",
    "predicted_key",
    "correct_top1",
    "true_in_top5",
    "top1_key",
    "top1_probability",
    "top2_key",
    "top2_probability",
    "top3_key",
    "top3_probability",
    "top4_key",
    "top4_probability",
    "top5_key",
    "top5_probability",
]


PROBABILITY_COLUMNS = [
    "clip_id",
    "trial_id",
    "event_index",
    "true_key",
    "predicted_key",
    "candidate_key",
    "probability",
]


@dataclass(frozen=True)
class AcousticTrainingOutputs:
    output_dir: Path
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    probabilities_path: Path
    report_path: Path
    metrics: dict[str, Any]


def key_label(value: Any) -> str:
    key = str(value or "")
    if key == " ":
        return "Space"
    return key or "Unknown"


def load_spectrogram_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def find_latest_spectrogram_session(spectrogram_root: Path | None = None) -> Path:
    root = spectrogram_root or PROCESSED_DATA_DIR / "spectrograms"
    sessions = sorted(path for path in root.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No spectrogram sessions found under {root}")
    return sessions[-1]


def load_spectrogram_array(path: Path) -> np.ndarray:
    with np.load(path) as loaded:
        if "spectrogram" not in loaded:
            raise KeyError(f"{path} does not contain a 'spectrogram' array")
        return loaded["spectrogram"].astype(np.float32)


def load_feature_matrix(records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], tuple[int, ...]]:
    if not records:
        raise ValueError("No spectrogram records were provided.")

    features: list[np.ndarray] = []
    labels: list[str] = []
    expected_shape: tuple[int, ...] | None = None
    for record in records:
        spectrogram = load_spectrogram_array(Path(record["spectrogram_path"]))
        if expected_shape is None:
            expected_shape = tuple(int(size) for size in spectrogram.shape)
        if tuple(spectrogram.shape) != expected_shape:
            raise ValueError(
                "All spectrograms must have the same shape for this first baseline. "
                f"Expected {expected_shape}, got {spectrogram.shape} for {record['spectrogram_path']}"
            )
        features.append(spectrogram.reshape(-1))
        labels.append(key_label(record.get("key")))

    return np.vstack(features).astype(np.float32), np.array(labels), records, expected_shape or ()


def split_train_test_indices(
    labels: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a stratified split, keeping one-example labels in training only."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    label_counts = Counter(labels)
    rare_labels = sorted(label for label, count in label_counts.items() if count < 2)
    common_indices = np.array([index for index, label in enumerate(labels) if label not in rare_labels])
    rare_indices = np.array([index for index, label in enumerate(labels) if label in rare_labels])

    if len(common_indices) < 2:
        raise ValueError("Need at least two non-rare examples to create a test split.")

    common_labels = labels[common_indices]
    try:
        train_common, test_indices = train_test_split(
            common_indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=common_labels,
        )
    except ValueError:
        train_common, test_indices = train_test_split(
            common_indices,
            test_size=test_size,
            random_state=random_seed,
            shuffle=True,
        )

    train_indices = np.concatenate([np.array(train_common), rare_indices]).astype(int)
    test_indices = np.array(test_indices).astype(int)
    return np.sort(train_indices), np.sort(test_indices), rare_labels


def build_model(max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )


def ranked_predictions(classes: np.ndarray, probabilities: np.ndarray, top_k: int = 5) -> list[list[tuple[str, float]]]:
    rankings: list[list[tuple[str, float]]] = []
    limit = min(top_k, len(classes))
    for row in probabilities:
        order = np.argsort(row)[::-1][:limit]
        rankings.append([(str(classes[index]), float(row[index])) for index in order])
    return rankings


def build_prediction_records(
    records: list[dict[str, str]],
    test_indices: np.ndarray,
    classes: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rankings = ranked_predictions(classes, probabilities, top_k=5)
    prediction_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []

    for row_index, record_index in enumerate(test_indices):
        record = records[int(record_index)]
        true_key = key_label(record.get("key"))
        ranking = rankings[row_index]
        predicted_key = ranking[0][0]
        top_keys = [label for label, _ in ranking]

        prediction_row: dict[str, Any] = {
            "clip_id": record.get("clip_id", ""),
            "spectrogram_path": record.get("spectrogram_path", ""),
            "trial_id": record.get("trial_id", ""),
            "event_index": record.get("event_index", ""),
            "true_key": true_key,
            "predicted_key": predicted_key,
            "correct_top1": int(predicted_key == true_key),
            "true_in_top5": int(true_key in top_keys),
        }
        for rank in range(5):
            key_column = f"top{rank + 1}_key"
            probability_column = f"top{rank + 1}_probability"
            if rank < len(ranking):
                prediction_row[key_column] = ranking[rank][0]
                prediction_row[probability_column] = round(ranking[rank][1], 9)
            else:
                prediction_row[key_column] = ""
                prediction_row[probability_column] = ""
        prediction_rows.append(prediction_row)

        for class_index, candidate_key in enumerate(classes):
            probability_rows.append(
                {
                    "clip_id": record.get("clip_id", ""),
                    "trial_id": record.get("trial_id", ""),
                    "event_index": record.get("event_index", ""),
                    "true_key": true_key,
                    "predicted_key": predicted_key,
                    "candidate_key": str(candidate_key),
                    "probability": round(float(probabilities[row_index, class_index]), 9),
                }
            )

    return prediction_rows, probability_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def label_count_dict(labels: np.ndarray) -> dict[str, int]:
    return dict(sorted(Counter(str(label) for label in labels).items()))


def build_metrics(
    session_id: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    prediction_rows: list[dict[str, Any]],
    rare_labels: list[str],
    test_size: float,
    random_seed: int,
    feature_shape: tuple[int, ...],
) -> dict[str, Any]:
    test_count = len(y_test)
    top1_count = sum(int(row["correct_top1"]) for row in prediction_rows)
    top5_count = sum(int(row["true_in_top5"]) for row in prediction_rows)
    return {
        "session_id": session_id,
        "model_type": "logistic_regression_flattened_log_mel",
        "feature_shape": list(feature_shape),
        "train_count": int(len(y_train)),
        "test_count": int(test_count),
        "class_count": int(len(set(y_train))),
        "test_size": test_size,
        "random_seed": random_seed,
        "top1_accuracy": round(top1_count / test_count, 6) if test_count else 0.0,
        "top5_accuracy": round(top5_count / test_count, 6) if test_count else 0.0,
        "train_label_counts": label_count_dict(y_train),
        "test_label_counts": label_count_dict(y_test),
        "train_only_labels": rare_labels,
    }


def build_text_report(metrics: dict[str, Any], y_test: np.ndarray, y_pred: np.ndarray) -> str:
    lines = [
        "Acoustic Baseline Report",
        "========================",
        "",
        f"Session: {metrics['session_id']}",
        "Model: logistic regression on flattened normalized log-mel spectrograms",
        f"Train clips: {metrics['train_count']}",
        f"Test clips: {metrics['test_count']}",
        f"Classes seen during training: {metrics['class_count']}",
        f"Top-1 accuracy: {metrics['top1_accuracy']:.3f}",
        f"Top-5 accuracy: {metrics['top5_accuracy']:.3f}",
        "",
        "Important beginner note:",
        "This is a first acoustic-only sanity check, not the final neural network.",
        "Labels with only one example are kept in training so the split does not create impossible test labels.",
    ]
    if metrics["train_only_labels"]:
        lines.append(f"Train-only labels: {', '.join(metrics['train_only_labels'])}")

    lines.extend(
        [
            "",
            "Classification report:",
            classification_report(y_test, y_pred, zero_division=0),
        ]
    )
    return "\n".join(lines)


def train_acoustic_baseline(
    spectrogram_manifest_path: Path,
    output_root: Path | None = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    max_iter: int = 2000,
) -> AcousticTrainingOutputs:
    records = load_spectrogram_manifest(spectrogram_manifest_path)
    features, labels, records, feature_shape = load_feature_matrix(records)
    train_indices, test_indices, rare_labels = split_train_test_indices(
        labels=labels,
        test_size=test_size,
        random_seed=random_seed,
    )

    x_train = features[train_indices]
    x_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    model = build_model(max_iter=max_iter)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    classes = model.classes_
    y_pred = classes[np.argmax(probabilities, axis=1)]

    prediction_rows, probability_rows = build_prediction_records(
        records=records,
        test_indices=test_indices,
        classes=classes,
        probabilities=probabilities,
    )

    session_id = records[0].get("session_id", spectrogram_manifest_path.parent.name)
    output_base = output_root or MODELS_DIR / "acoustic_baseline"
    output_dir = output_base / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    probabilities_path = output_dir / "test_probabilities.csv"
    report_path = output_dir / "report.txt"

    metrics = build_metrics(
        session_id=session_id,
        y_train=y_train,
        y_test=y_test,
        prediction_rows=prediction_rows,
        rare_labels=rare_labels,
        test_size=test_size,
        random_seed=random_seed,
        feature_shape=feature_shape,
    )

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(predictions_path, prediction_rows, PREDICTION_COLUMNS)
    write_csv(probabilities_path, probability_rows, PROBABILITY_COLUMNS)
    report_path.write_text(build_text_report(metrics, y_test, y_pred), encoding="utf-8")

    return AcousticTrainingOutputs(
        output_dir=output_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        probabilities_path=probabilities_path,
        report_path=report_path,
        metrics=metrics,
    )
