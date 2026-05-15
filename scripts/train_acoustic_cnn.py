from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.acoustic_cnn import train_acoustic_cnn
from keyboard_fusion.acoustic_model import (
    COMBINED_SESSION_ID,
    build_combined_spectrogram_manifest,
    find_latest_spectrogram_session,
)
from keyboard_fusion.config import load_config
from keyboard_fusion.paths import MODELS_DIR, PROCESSED_DATA_DIR


def resolve_spectrogram_manifest(session: str | None, spectrogram_manifest: str | None) -> Path:
    if spectrogram_manifest:
        path = Path(spectrogram_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Could not find spectrogram manifest: {spectrogram_manifest}")
        return path

    spectrogram_root = PROCESSED_DATA_DIR / "spectrograms"
    if session:
        path = Path(session)
        if path.exists() and path.is_dir():
            return path / "spectrogram_manifest.csv"
        return spectrogram_root / session / "spectrogram_manifest.csv"

    return find_latest_spectrogram_session(spectrogram_root) / "spectrogram_manifest.csv"


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    experiment_config = config.get("experiment", {})

    parser = argparse.ArgumentParser(
        description="Train the optimized acoustic-only CNN from generated spectrograms."
    )
    parser.add_argument(
        "--session",
        help=(
            "Spectrogram session ID or folder path. Defaults to the latest folder under "
            "data/processed/spectrograms/."
        ),
    )
    parser.add_argument("--spectrogram-manifest", help="Path to a specific spectrogram_manifest.csv.")
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Build a combined manifest from every processed spectrogram session before training.",
    )
    parser.add_argument(
        "--combined-session-id",
        default=COMBINED_SESSION_ID,
        help="Folder/model name used when --all-sessions is enabled.",
    )
    parser.add_argument("--output-root", type=Path, default=MODELS_DIR / "acoustic_cnn")
    parser.add_argument("--test-size", type=float, default=float(experiment_config.get("test_size", 0.2)))
    parser.add_argument(
        "--validation-size",
        type=float,
        default=float(experiment_config.get("validation_size", 0.2)),
        help="Fraction of the training split used for early-stopping validation.",
    )
    parser.add_argument("--random-seed", type=int, default=int(experiment_config.get("random_seed", 42)))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--frequency-mask-width", type=int, default=8)
    parser.add_argument("--time-mask-width", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, mps, or cuda. auto prefers Apple MPS, then CUDA, then CPU.",
    )
    args = parser.parse_args(argv)

    combined_summary: dict[str, object] | None = None
    if args.all_sessions:
        if args.session or args.spectrogram_manifest:
            parser.error("--all-sessions cannot be combined with --session or --spectrogram-manifest")
        spectrogram_manifest_path, combined_summary = build_combined_spectrogram_manifest(
            combined_session_id=args.combined_session_id,
        )
    else:
        spectrogram_manifest_path = resolve_spectrogram_manifest(args.session, args.spectrogram_manifest)

    outputs = train_acoustic_cnn(
        spectrogram_manifest_path=spectrogram_manifest_path,
        output_root=args.output_root,
        output_session_id=args.combined_session_id if args.all_sessions else None,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_seed=args.random_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        dropout=args.dropout,
        mixup_alpha=args.mixup_alpha,
        frequency_mask_width=args.frequency_mask_width,
        time_mask_width=args.time_mask_width,
        noise_std=args.noise_std,
        device=args.device,
    )

    metrics = outputs.metrics
    if combined_summary:
        print(
            "Combined sessions: "
            f"{combined_summary['manifest_count']} manifests, {combined_summary['total_records']} spectrograms"
        )
        for session_id, count in combined_summary["session_counts"].items():
            print(f"- {session_id}: {count}")
    print(f"Spectrogram manifest: {spectrogram_manifest_path}")
    print(f"Output folder: {outputs.output_dir}")
    print(f"Model type: {metrics['model_type']}")
    print(f"Device: {metrics['device']}")
    print(f"Train clips: {metrics['train_count']}")
    print(f"Validation clips: {metrics['validation_count']}")
    print(f"Test clips: {metrics['test_count']}")
    print(f"Classes: {metrics['class_count']}")
    print(f"Parameters: {metrics['architecture']['trainable_parameters']:,}")
    print(f"Best epoch: {metrics['best_epoch']} / {metrics['epochs_ran']}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
    print(f"Top-5 accuracy: {metrics['top5_accuracy']:.3f}")
    if metrics["train_only_labels"]:
        print(f"Train-only labels with one example: {', '.join(metrics['train_only_labels'])}")
    print(f"Model: {outputs.model_path}")
    print(f"Predictions: {outputs.predictions_path}")
    print(f"Probabilities: {outputs.probabilities_path}")
    print(f"History: {outputs.history_path}")
    print(f"Metrics: {outputs.metrics_path}")
    print(f"Report: {outputs.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
