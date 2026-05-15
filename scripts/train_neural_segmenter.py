from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_fusion.config import load_config
from keyboard_fusion.neural_segmentation import (
    DEFAULT_SEGMENTER_SESSION_ID,
    train_neural_segmenter,
)
from keyboard_fusion.paths import MODELS_DIR
from keyboard_fusion.segmentation_evaluation import find_alignment_paths


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    neural_segmenter_config = config.get("neural_segmenter", {})
    experiment_config = config.get("experiment", {})

    parser = argparse.ArgumentParser(
        description=(
            "Train the neural keystroke segmenter. This is NN #1 in the two-stage "
            "pipeline: raw phrase audio -> keypress clips."
        )
    )
    parser.add_argument("--session", help="Alignment session ID. Defaults to the latest aligned session.")
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Train from every alignment session under data/metadata/alignment/.",
    )
    parser.add_argument("--output-root", type=Path, default=MODELS_DIR / "neural_segmenter")
    parser.add_argument("--output-session-id", default=DEFAULT_SEGMENTER_SESSION_ID)
    parser.add_argument("--sample-rate", type=int, default=int(neural_segmenter_config.get("sample_rate", 24000)))
    parser.add_argument("--test-size", type=float, default=float(experiment_config.get("test_size", 0.2)))
    parser.add_argument("--validation-size", type=float, default=float(experiment_config.get("validation_size", 0.2)))
    parser.add_argument("--random-seed", type=int, default=int(experiment_config.get("random_seed", 42)))
    parser.add_argument("--epochs", type=int, default=int(neural_segmenter_config.get("epochs", 25)))
    parser.add_argument("--batch-size", type=int, default=int(neural_segmenter_config.get("batch_size", 128)))
    parser.add_argument("--learning-rate", type=float, default=float(neural_segmenter_config.get("learning_rate", 0.001)))
    parser.add_argument("--weight-decay", type=float, default=float(neural_segmenter_config.get("weight_decay", 0.01)))
    parser.add_argument("--patience", type=int, default=int(neural_segmenter_config.get("patience", 7)))
    parser.add_argument("--window-ms", type=float, default=float(neural_segmenter_config.get("window_ms", 75.0)))
    parser.add_argument("--hop-ms", type=float, default=float(neural_segmenter_config.get("hop_ms", 5.0)))
    parser.add_argument("--min-gap-ms", type=float, default=float(neural_segmenter_config.get("min_gap_ms", 75.0)))
    parser.add_argument("--threshold", type=float, default=float(neural_segmenter_config.get("threshold", 0.5)))
    parser.add_argument("--tolerance-ms", type=float, default=float(neural_segmenter_config.get("tolerance_ms", 35.0)))
    parser.add_argument(
        "--max-peak-multiplier",
        type=float,
        default=float(neural_segmenter_config.get("max_peak_multiplier", 1.0)),
    )
    parser.add_argument("--negative-ratio", type=float, default=float(neural_segmenter_config.get("negative_ratio", 1.5)))
    parser.add_argument(
        "--positive-jitters-per-event",
        type=int,
        default=int(neural_segmenter_config.get("positive_jitters_per_event", 1)),
    )
    parser.add_argument(
        "--positive-jitter-ms",
        type=float,
        default=float(neural_segmenter_config.get("positive_jitter_ms", 3.0)),
    )
    parser.add_argument(
        "--negative-exclusion-ms",
        type=float,
        default=float(neural_segmenter_config.get("negative_exclusion_ms", 45.0)),
    )
    parser.add_argument("--dropout", type=float, default=float(neural_segmenter_config.get("dropout", 0.25)))
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, mps, or cuda. auto prefers Apple MPS, then CUDA, then CPU.",
    )
    args = parser.parse_args(argv)

    if args.all_sessions and args.session:
        parser.error("--all-sessions cannot be combined with --session")

    output_session_id, alignment_paths = find_alignment_paths(
        session=args.session,
        all_sessions=args.all_sessions,
    )
    if args.all_sessions:
        output_session_id = args.output_session_id

    outputs = train_neural_segmenter(
        alignment_paths=alignment_paths,
        output_root=args.output_root,
        output_session_id=output_session_id,
        target_sample_rate=args.sample_rate,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_seed=args.random_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        min_gap_ms=args.min_gap_ms,
        threshold=args.threshold,
        tolerance_ms=args.tolerance_ms,
        max_peak_multiplier=args.max_peak_multiplier,
        negative_ratio=args.negative_ratio,
        positive_jitters_per_event=args.positive_jitters_per_event,
        positive_jitter_ms=args.positive_jitter_ms,
        negative_exclusion_ms=args.negative_exclusion_ms,
        dropout=args.dropout,
        device=args.device,
    )

    metrics = outputs.metrics
    event_metrics = metrics["test_event_metrics"]
    window_metrics = metrics["test_window_metrics"]
    print(f"Alignment files: {len(alignment_paths)}")
    print(f"Output folder: {outputs.output_dir}")
    print(f"Device: {metrics['device']}")
    print(f"Train windows: {metrics['train_window_count']}")
    print(f"Validation windows: {metrics['validation_window_count']}")
    print(f"Test windows: {metrics['test_window_count']}")
    print(f"Parameters: {metrics['architecture']['trainable_parameters']:,}")
    print(f"Best epoch: {metrics['best_epoch']} / {metrics['epochs_ran']}")
    print(f"Window F1: {window_metrics['f1']:.3f}")
    print(f"Event precision: {event_metrics['precision']:.3f}")
    print(f"Event recall: {event_metrics['recall']:.3f}")
    print(f"Event F1: {event_metrics['f1']:.3f}")
    print(f"Model: {outputs.model_path}")
    print(f"Metrics: {outputs.metrics_path}")
    print(f"Event predictions: {outputs.event_predictions_path}")
    print(f"Report: {outputs.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
