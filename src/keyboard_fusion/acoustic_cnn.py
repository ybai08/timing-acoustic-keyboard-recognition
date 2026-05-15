from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keyboard_fusion.acoustic_model import (
    PREDICTION_COLUMNS,
    PROBABILITY_COLUMNS,
    build_prediction_records,
    find_latest_spectrogram_session,
    key_label,
    label_count_dict,
    load_spectrogram_array,
    load_spectrogram_manifest,
    split_train_test_indices,
    write_csv,
)
from keyboard_fusion.paths import MODELS_DIR


def _require_torch() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for the acoustic CNN. Install it with "
            "`python -m pip install -r requirements-ml.txt`."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


@dataclass(frozen=True)
class AcousticCNNTrainingOutputs:
    output_dir: Path
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    probabilities_path: Path
    history_path: Path
    report_path: Path
    metrics: dict[str, Any]


def load_spectrogram_tensor(records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if not records:
        raise ValueError("No spectrogram records were provided.")

    features: list[np.ndarray] = []
    labels: list[str] = []
    expected_shape: tuple[int, ...] | None = None
    for record in records:
        spectrogram = load_spectrogram_array(Path(record["spectrogram_path"]))
        if spectrogram.ndim != 2:
            raise ValueError(f"Expected a 2D spectrogram, got shape {spectrogram.shape}")
        if expected_shape is None:
            expected_shape = tuple(int(size) for size in spectrogram.shape)
        if tuple(spectrogram.shape) != expected_shape:
            raise ValueError(
                "All spectrograms must have the same shape for CNN training. "
                f"Expected {expected_shape}, got {spectrogram.shape} for {record['spectrogram_path']}"
            )
        features.append(spectrogram.astype(np.float32))
        labels.append(key_label(record.get("key")))

    return np.stack(features).astype(np.float32), np.array(labels), expected_shape or ()


def split_train_validation_indices(
    labels: np.ndarray,
    validation_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if validation_size <= 0:
        return np.arange(len(labels), dtype=int), np.array([], dtype=int), []
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    label_counts = Counter(labels)
    rare_labels = sorted(label for label, count in label_counts.items() if count < 2)
    common_indices = np.array([index for index, label in enumerate(labels) if label not in rare_labels])
    rare_indices = np.array([index for index, label in enumerate(labels) if label in rare_labels])

    if len(common_indices) < 2:
        return np.arange(len(labels), dtype=int), np.array([], dtype=int), rare_labels

    common_labels = labels[common_indices]
    try:
        train_common, validation_indices = train_test_split(
            common_indices,
            test_size=validation_size,
            random_state=random_seed,
            stratify=common_labels,
        )
    except ValueError:
        train_common, validation_indices = train_test_split(
            common_indices,
            test_size=validation_size,
            random_state=random_seed,
            shuffle=True,
        )

    train_indices = np.concatenate([np.array(train_common), rare_indices]).astype(int)
    validation_indices = np.array(validation_indices).astype(int)
    return np.sort(train_indices), np.sort(validation_indices), rare_labels


def encode_labels(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    class_to_index = {label: index for index, label in enumerate(class_names)}
    missing = sorted(set(str(label) for label in labels) - set(class_to_index))
    if missing:
        raise ValueError(f"Labels were not present during training: {', '.join(missing)}")
    return np.array([class_to_index[str(label)] for label in labels], dtype=np.int64)


def standardize_from_training(
    features: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(features[train_indices]))
    std = float(np.std(features[train_indices]))
    if std < 1e-8:
        return np.zeros_like(features, dtype=np.float32), mean, std
    return ((features - mean) / std).astype(np.float32), mean, std


def class_weight_tensor(y_train: np.ndarray, class_count: int, device: Any) -> Any:
    torch, _, _, _, _ = _require_torch()
    counts = np.bincount(y_train, minlength=class_count).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    beta = 0.99
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.35, 6.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def resolve_device(requested_device: str) -> str:
    torch, _, _, _, _ = _require_torch()
    if requested_device != "auto":
        return requested_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_acoustic_cnn_model(input_shape: tuple[int, int], class_count: int, dropout: float = 0.35) -> Any:
    _, nn, _, _, _ = _require_torch()

    class ConvNormAct(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] = (3, 3)) -> None:
            super().__init__()
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        def forward(self, x: Any) -> Any:
            return self.block(x)

    class SqueezeExcite(nn.Module):
        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            hidden = max(8, channels // reduction)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: Any) -> Any:
            return x * self.gate(x)

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, block_dropout: float = 0.08) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Dropout2d(block_dropout),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                SqueezeExcite(channels),
            )
            self.activation = nn.GELU()

        def forward(self, x: Any) -> Any:
            return self.activation(x + self.net(x))

    class AcousticSpectrogramResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                ConvNormAct(1, 32, kernel_size=(5, 3)),
                ResidualBlock(32),
                nn.MaxPool2d(kernel_size=(2, 1)),
                ConvNormAct(32, 64),
                ResidualBlock(64),
                nn.MaxPool2d(kernel_size=(2, 1)),
                ConvNormAct(64, 96),
                ResidualBlock(96),
                nn.MaxPool2d(kernel_size=(2, 2)),
                ConvNormAct(96, 128),
                ResidualBlock(128),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Dropout(dropout),
                nn.Linear(128, class_count),
            )

        def forward(self, x: Any) -> Any:
            return self.classifier(self.features(x))

    model = AcousticSpectrogramResNet()
    model.input_shape = input_shape
    return model


def count_trainable_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def architecture_description(input_shape: tuple[int, int], class_count: int, trainable_parameters: int) -> dict[str, Any]:
    return {
        "model_name": "AcousticSpectrogramResNet",
        "input_shape": [1, int(input_shape[0]), int(input_shape[1])],
        "output_classes": int(class_count),
        "trainable_parameters": trainable_parameters,
        "layers": [
            "Conv2D 1->32, kernel 5x3, BatchNorm, GELU",
            "Residual 32-channel block with squeeze-excite",
            "Frequency-only MaxPool",
            "Conv2D 32->64, BatchNorm, GELU",
            "Residual 64-channel block with squeeze-excite",
            "Frequency-only MaxPool",
            "Conv2D 64->96, BatchNorm, GELU",
            "Residual 96-channel block with squeeze-excite",
            "Frequency/time MaxPool",
            "Conv2D 96->128, BatchNorm, GELU",
            "Residual 128-channel block with squeeze-excite",
            "Global average pool",
            "Dropout, Dense 128, GELU, LayerNorm, Dropout",
            "Dense softmax logits for key classes",
        ],
    }


def augment_spectrogram_batch(
    x: Any,
    frequency_mask_width: int,
    time_mask_width: int,
    noise_std: float,
) -> Any:
    torch, _, _, _, _ = _require_torch()
    augmented = x.clone()
    batch_size, _, mel_bands, frames = augmented.shape

    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std

    if frequency_mask_width > 0:
        for index in range(batch_size):
            width = random.randint(0, min(frequency_mask_width, mel_bands))
            if width:
                start = random.randint(0, mel_bands - width)
                augmented[index, :, start : start + width, :] = 0.0

    if time_mask_width > 0:
        for index in range(batch_size):
            width = random.randint(0, min(time_mask_width, frames))
            if width:
                start = random.randint(0, frames - width)
                augmented[index, :, :, start : start + width] = 0.0

    return augmented


def weighted_soft_cross_entropy(logits: Any, targets: Any, class_weights: Any) -> Any:
    _, _, F, _, _ = _require_torch()
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs * class_weights.unsqueeze(0)).sum(dim=1).mean()


def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    class_weights: Any,
    device: str,
    mixup_alpha: float,
    frequency_mask_width: int,
    time_mask_width: int,
    noise_std: float,
    class_count: int,
) -> tuple[float, float]:
    torch, _, F, _, _ = _require_torch()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        x_batch = augment_spectrogram_batch(
            x_batch,
            frequency_mask_width=frequency_mask_width,
            time_mask_width=time_mask_width,
            noise_std=noise_std,
        )

        optimizer.zero_grad(set_to_none=True)
        if mixup_alpha > 0 and x_batch.shape[0] > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(x_batch.shape[0], device=device)
            mixed_x = lam * x_batch + (1.0 - lam) * x_batch[permutation]
            y_one_hot = F.one_hot(y_batch, num_classes=class_count).float()
            mixed_targets = lam * y_one_hot + (1.0 - lam) * y_one_hot[permutation]
            logits = model(mixed_x)
            loss = weighted_soft_cross_entropy(logits, mixed_targets, class_weights)
        else:
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch, weight=class_weights)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == y_batch).sum().item())
            total_count += int(y_batch.shape[0])
            total_loss += float(loss.item()) * int(y_batch.shape[0])

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def evaluate_model(model: Any, loader: Any, device: str, class_weights: Any | None = None) -> tuple[float, float, np.ndarray]:
    torch, _, F, _, _ = _require_torch()
    model.eval()
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch, weight=class_weights)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

            probabilities.append(probs.cpu().numpy())
            total_loss += float(loss.item()) * int(y_batch.shape[0])
            total_correct += int((predictions == y_batch).sum().item())
            total_count += int(y_batch.shape[0])

    if probabilities:
        stacked_probabilities = np.vstack(probabilities).astype(np.float32)
    else:
        stacked_probabilities = np.empty((0, 0), dtype=np.float32)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1), stacked_probabilities


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "learning_rate"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_metrics(
    session_id: str,
    model: Any,
    class_names: list[str],
    feature_shape: tuple[int, int],
    y_train_labels: np.ndarray,
    y_validation_labels: np.ndarray,
    y_test_labels: np.ndarray,
    prediction_rows: list[dict[str, Any]],
    train_only_labels: list[str],
    validation_only_labels: list[str],
    test_size: float,
    validation_size: float,
    random_seed: int,
    best_epoch: int,
    epochs_ran: int,
    elapsed_seconds: float,
    device: str,
    normalization_mean: float,
    normalization_std: float,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    test_count = len(y_test_labels)
    top1_count = sum(int(row["correct_top1"]) for row in prediction_rows)
    top5_count = sum(int(row["true_in_top5"]) for row in prediction_rows)
    trainable_parameters = count_trainable_parameters(model)
    return {
        "session_id": session_id,
        "model_type": "acoustic_spectrogram_resnet",
        "feature_shape": list(feature_shape),
        "architecture": architecture_description(feature_shape, len(class_names), trainable_parameters),
        "train_count": int(len(y_train_labels)),
        "validation_count": int(len(y_validation_labels)),
        "test_count": int(test_count),
        "class_count": int(len(class_names)),
        "test_size": test_size,
        "validation_size": validation_size,
        "random_seed": random_seed,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "device": device,
        "normalization_mean": round(normalization_mean, 9),
        "normalization_std": round(normalization_std, 9),
        "training_config": training_config,
        "top1_accuracy": round(top1_count / test_count, 6) if test_count else 0.0,
        "top5_accuracy": round(top5_count / test_count, 6) if test_count else 0.0,
        "train_label_counts": label_count_dict(y_train_labels),
        "validation_label_counts": label_count_dict(y_validation_labels),
        "test_label_counts": label_count_dict(y_test_labels),
        "train_only_labels": train_only_labels,
        "validation_rare_labels_kept_in_training": validation_only_labels,
        "classes": class_names,
    }


def build_text_report(metrics: dict[str, Any], y_test: np.ndarray, y_pred: np.ndarray) -> str:
    architecture = metrics["architecture"]
    training_config = metrics.get("training_config", {})
    lines = [
        "Optimized Acoustic CNN Report",
        "=============================",
        "",
        f"Session: {metrics['session_id']}",
        "Model: compact ResNet-style CNN on normalized log-mel spectrograms",
        f"Input shape: {architecture['input_shape']}",
        f"Trainable parameters: {architecture['trainable_parameters']:,}",
        f"Device: {metrics['device']}",
        f"Train clips: {metrics['train_count']}",
        f"Validation clips: {metrics['validation_count']}",
        f"Test clips: {metrics['test_count']}",
        f"Classes seen during training: {metrics['class_count']}",
        f"Best epoch: {metrics['best_epoch']} of {metrics['epochs_ran']}",
        f"Top-1 accuracy: {metrics['top1_accuracy']:.3f}",
        f"Top-5 accuracy: {metrics['top5_accuracy']:.3f}",
        "",
        "Architecture:",
    ]
    lines.extend(f"- {layer}" for layer in architecture["layers"])
    lines.extend(
        [
            "",
            "Training details:",
            "- AdamW optimizer",
            "- cosine learning-rate schedule",
            "- class-balanced loss",
            "- early stopping on validation loss",
        ]
    )
    if int(training_config.get("frequency_mask_width", 0)) > 0 or int(training_config.get("time_mask_width", 0)) > 0:
        lines.append(
            "- light SpecAugment-style masking "
            f"(frequency width {training_config.get('frequency_mask_width')}, "
            f"time width {training_config.get('time_mask_width')})"
        )
    if float(training_config.get("noise_std", 0.0)) > 0:
        lines.append(f"- small Gaussian noise augmentation ({training_config.get('noise_std')})")
    if float(training_config.get("mixup_alpha", 0.0)) > 0:
        lines.append(f"- mixup regularization (alpha {training_config.get('mixup_alpha')})")
    else:
        lines.append("- mixup disabled for this tuned acoustic-only run")
    if metrics["train_only_labels"]:
        lines.append(f"- Train-only labels: {', '.join(metrics['train_only_labels'])}")

    lines.extend(
        [
            "",
            "Classification report:",
            classification_report(y_test, y_pred, zero_division=0),
        ]
    )
    return "\n".join(lines)


def train_acoustic_cnn(
    spectrogram_manifest_path: Path,
    output_root: Path | None = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 35,
    dropout: float = 0.25,
    mixup_alpha: float = 0.0,
    frequency_mask_width: int = 4,
    time_mask_width: int = 1,
    noise_std: float = 0.01,
    device: str = "auto",
) -> AcousticCNNTrainingOutputs:
    torch, _, _, DataLoader, TensorDataset = _require_torch()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    requested_device = device
    device = resolve_device(requested_device)
    records = load_spectrogram_manifest(spectrogram_manifest_path)
    features, labels, feature_shape = load_spectrogram_tensor(records)

    train_full_indices, test_indices, train_only_labels = split_train_test_indices(
        labels=labels,
        test_size=test_size,
        random_seed=random_seed,
    )
    train_local_indices, validation_local_indices, validation_rare_labels = split_train_validation_indices(
        labels=labels[train_full_indices],
        validation_size=validation_size,
        random_seed=random_seed,
    )
    train_indices = train_full_indices[train_local_indices]
    validation_indices = train_full_indices[validation_local_indices]

    features, normalization_mean, normalization_std = standardize_from_training(features, train_indices)
    class_names = sorted(str(label) for label in set(labels[train_full_indices]))
    y_train = encode_labels(labels[train_indices], class_names)
    y_validation = encode_labels(labels[validation_indices], class_names) if len(validation_indices) else np.array([], dtype=np.int64)
    y_test = encode_labels(labels[test_indices], class_names)

    x_train = torch.tensor(features[train_indices, None, :, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_validation = torch.tensor(features[validation_indices, None, :, :], dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)
    x_test = torch.tensor(features[test_indices, None, :, :], dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    validation_loader = DataLoader(
        TensorDataset(x_validation, y_validation_tensor),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test_tensor),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = build_acoustic_cnn_model(feature_shape, class_count=len(class_names), dropout=dropout).to(device)
    class_weights = class_weight_tensor(y_train, len(class_names), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.05)

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
            class_weights=class_weights,
            device=device,
            mixup_alpha=mixup_alpha,
            frequency_mask_width=frequency_mask_width,
            time_mask_width=time_mask_width,
            noise_std=noise_std,
            class_count=len(class_names),
        )
        validation_loss, validation_accuracy, _ = evaluate_model(
            model,
            validation_loader if len(validation_indices) else train_loader,
            device=device,
            class_weights=None,
        )
        scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 9),
                "train_accuracy": round(train_accuracy, 9),
                "validation_loss": round(validation_loss, 9),
                "validation_accuracy": round(validation_accuracy, 9),
                "learning_rate": round(current_lr, 12),
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
    _, test_accuracy, probabilities = evaluate_model(model, test_loader, device=device, class_weights=None)
    classes = np.array(class_names)
    y_pred = classes[np.argmax(probabilities, axis=1)]

    prediction_rows, probability_rows = build_prediction_records(
        records=records,
        test_indices=test_indices,
        classes=classes,
        probabilities=probabilities,
    )

    session_id = records[0].get("session_id", spectrogram_manifest_path.parent.name)
    output_base = output_root or MODELS_DIR / "acoustic_cnn"
    output_dir = output_base / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    probabilities_path = output_dir / "test_probabilities.csv"
    history_path = output_dir / "training_history.csv"
    report_path = output_dir / "report.txt"
    training_config = {
        "test_size": test_size,
        "validation_size": validation_size,
        "random_seed": random_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "dropout": dropout,
        "mixup_alpha": mixup_alpha,
        "frequency_mask_width": frequency_mask_width,
        "time_mask_width": time_mask_width,
        "noise_std": noise_std,
        "requested_device": requested_device,
        "device": device,
    }

    metrics = build_metrics(
        session_id=session_id,
        model=model,
        class_names=class_names,
        feature_shape=feature_shape,
        y_train_labels=labels[train_indices],
        y_validation_labels=labels[validation_indices],
        y_test_labels=labels[test_indices],
        prediction_rows=prediction_rows,
        train_only_labels=train_only_labels,
        validation_only_labels=validation_rare_labels,
        test_size=test_size,
        validation_size=validation_size,
        random_seed=random_seed,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        elapsed_seconds=elapsed_seconds,
        device=device,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        training_config=training_config,
    )
    metrics["test_accuracy_internal"] = round(float(test_accuracy), 6)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "feature_shape": list(feature_shape),
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "training_config": training_config,
            "metrics": metrics,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(predictions_path, prediction_rows, PREDICTION_COLUMNS)
    write_csv(probabilities_path, probability_rows, PROBABILITY_COLUMNS)
    write_history(history_path, history)
    report_path.write_text(build_text_report(metrics, labels[test_indices], y_pred), encoding="utf-8")

    return AcousticCNNTrainingOutputs(
        output_dir=output_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        probabilities_path=probabilities_path,
        history_path=history_path,
        report_path=report_path,
        metrics=metrics,
    )


__all__ = [
    "AcousticCNNTrainingOutputs",
    "find_latest_spectrogram_session",
    "load_spectrogram_tensor",
    "split_train_validation_indices",
    "train_acoustic_cnn",
]
