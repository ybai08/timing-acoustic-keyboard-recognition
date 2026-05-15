from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.acoustic_model import find_latest_spectrogram_session, train_acoustic_baseline
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
        description="Train the first acoustic-only baseline from generated spectrograms."
    )
    parser.add_argument(
        "--session",
        help=(
            "Spectrogram session ID or folder path. Defaults to the latest folder under "
            "data/processed/spectrograms/."
        ),
    )
    parser.add_argument("--spectrogram-manifest", help="Path to a specific spectrogram_manifest.csv.")
    parser.add_argument("--output-root", type=Path, default=MODELS_DIR / "acoustic_baseline")
    parser.add_argument("--test-size", type=float, default=float(experiment_config.get("test_size", 0.2)))
    parser.add_argument("--random-seed", type=int, default=int(experiment_config.get("random_seed", 42)))
    parser.add_argument("--max-iter", type=int, default=2000)
    args = parser.parse_args(argv)

    spectrogram_manifest_path = resolve_spectrogram_manifest(args.session, args.spectrogram_manifest)
    outputs = train_acoustic_baseline(
        spectrogram_manifest_path=spectrogram_manifest_path,
        output_root=args.output_root,
        test_size=args.test_size,
        random_seed=args.random_seed,
        max_iter=args.max_iter,
    )

    metrics = outputs.metrics
    print(f"Spectrogram manifest: {spectrogram_manifest_path}")
    print(f"Output folder: {outputs.output_dir}")
    print(f"Train clips: {metrics['train_count']}")
    print(f"Test clips: {metrics['test_count']}")
    print(f"Classes: {metrics['class_count']}")
    print(f"Top-1 accuracy: {metrics['top1_accuracy']:.3f}")
    print(f"Top-5 accuracy: {metrics['top5_accuracy']:.3f}")
    if metrics["train_only_labels"]:
        print(f"Train-only labels with one example: {', '.join(metrics['train_only_labels'])}")
    print(f"Model: {outputs.model_path}")
    print(f"Predictions: {outputs.predictions_path}")
    print(f"Probabilities: {outputs.probabilities_path}")
    print(f"Metrics: {outputs.metrics_path}")
    print(f"Report: {outputs.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
