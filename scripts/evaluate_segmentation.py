from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keyboard_fusion.acoustic_inference import AcousticCNNPredictor, DEFAULT_MODEL_DIR
from keyboard_fusion.segmentation_evaluation import (
    SegmentationParameters,
    evaluate_alignment_paths,
    extract_detected_clips,
    find_alignment_paths,
    tune_parameters,
    write_evaluation_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate automatic audio-only keystroke segmentation against oracle key logs."
    )
    parser.add_argument("--session", help="Alignment session ID. Defaults to the latest aligned session.")
    parser.add_argument("--all-sessions", action="store_true", help="Evaluate every aligned session together.")
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--min-gap-ms", type=float, default=55.0)
    parser.add_argument("--tolerance-ms", type=float, default=35.0)
    parser.add_argument("--max-peak-multiplier", type=float, default=1.1)
    parser.add_argument("--pre-ms", type=float, default=20.0)
    parser.add_argument("--post-ms", type=float, default=45.0)
    parser.add_argument("--tune", action="store_true", help="Run a small parameter grid and use the best F1 settings.")
    parser.add_argument("--predict-cnn", action="store_true", help="Run the acoustic CNN on matched detected clips.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--extract-clips", action="store_true", help="Write detected-peak clips and a clip manifest.")
    parser.add_argument(
        "--include-false-positives",
        action="store_true",
        help="When extracting clips, include unmatched detected peaks as unlabeled false-positive clips.",
    )
    args = parser.parse_args(argv)

    if args.session and args.all_sessions:
        parser.error("--session and --all-sessions cannot be used together")

    output_session_id, alignment_paths = find_alignment_paths(
        session=args.session,
        all_sessions=args.all_sessions,
    )
    parameters = SegmentationParameters(
        sensitivity=args.sensitivity,
        min_gap_ms=args.min_gap_ms,
        tolerance_ms=args.tolerance_ms,
        max_peak_multiplier=args.max_peak_multiplier,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
    )

    if args.tune:
        tuning = tune_parameters(
            alignment_paths=alignment_paths,
            tolerance_ms=args.tolerance_ms,
        )
        best_parameters = tuning["best_parameters"]
        parameters = SegmentationParameters(
            sensitivity=float(best_parameters["sensitivity"]),
            min_gap_ms=float(best_parameters["min_gap_ms"]),
            tolerance_ms=float(best_parameters["tolerance_ms"]),
            max_peak_multiplier=float(best_parameters["max_peak_multiplier"]),
            pre_ms=args.pre_ms,
            post_ms=args.post_ms,
        )
        tuning_path = PROJECT_ROOT / "data" / "metadata" / "segmentation" / output_session_id / "segmentation_tuning.json"
        tuning_path.parent.mkdir(parents=True, exist_ok=True)
        tuning_path.write_text(json.dumps(tuning, indent=2), encoding="utf-8")
        print(f"Tuning: {tuning_path}")
        print(
            "Best parameters: "
            f"sensitivity={parameters.sensitivity}, min_gap_ms={parameters.min_gap_ms}, "
            f"max_peak_multiplier={parameters.max_peak_multiplier}"
        )

    predictor = AcousticCNNPredictor(args.model_dir) if args.predict_cnn else None
    evaluation = evaluate_alignment_paths(
        alignment_paths=alignment_paths,
        parameters=parameters,
        predictor=predictor,
    )
    report_json, report_txt, matches_csv = write_evaluation_outputs(
        evaluation=evaluation,
        output_session_id=output_session_id,
    )

    clip_manifest = None
    if args.extract_clips:
        _, clip_manifest = extract_detected_clips(
            trials=evaluation["trials"],
            output_session_id=f"{output_session_id}_detected",
            include_false_positives=args.include_false_positives,
            parameters=parameters,
        )

    summary = evaluation["summary"]
    print(f"Segmentation session: {output_session_id}")
    print(f"Trials: {summary['trial_count']}")
    print(f"True keydowns: {summary['true_key_count']}")
    print(f"Detected peaks: {summary['detected_key_count']}")
    print(f"Matched: {summary['matched_count']}")
    print(f"False positives: {summary['false_positive_count']}")
    print(f"False negatives: {summary['false_negative_count']}")
    print(f"Precision: {summary['precision']:.3f}")
    print(f"Recall: {summary['recall']:.3f}")
    print(f"F1: {summary['f1']:.3f}")
    print(f"Mean absolute timing error: {summary['mean_abs_error_ms']} ms")
    if summary.get("acoustic_prediction"):
        acoustic = summary["acoustic_prediction"]
        print(f"CNN top-1 on matched detected clips: {acoustic['top1_accuracy_on_matched_detections']:.3f}")
        print(f"CNN top-5 on matched detected clips: {acoustic['top5_accuracy_on_matched_detections']:.3f}")
    print(f"Report: {report_txt}")
    print(f"JSON: {report_json}")
    print(f"Matches: {matches_csv}")
    if clip_manifest:
        print(f"Detected clip manifest: {clip_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
