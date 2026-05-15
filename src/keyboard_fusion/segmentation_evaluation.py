from __future__ import annotations

import csv
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from keyboard_fusion.acoustic_inference import AcousticCNNPredictor
from keyboard_fusion.config import load_config
from keyboard_fusion.paths import METADATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from keyboard_fusion.preprocessing import MANIFEST_COLUMNS, safe_label
from keyboard_fusion.segmentation import DetectedPeak, detect_keystroke_peaks, extract_fixed_window
from keyboard_fusion.spectrograms import read_wav_mono_float


DEFAULT_SEGMENTATION_SESSION_ID = "all_sessions"

MATCH_COLUMNS = [
    "session_id",
    "trial_id",
    "status",
    "oracle_event_index",
    "oracle_key",
    "oracle_code",
    "oracle_time_seconds",
    "detected_index",
    "detected_time_seconds",
    "detected_sample_index",
    "error_ms",
    "peak_strength",
    "threshold_ratio",
    "predicted_key",
    "top1_probability",
    "true_in_top5",
]

DETECTED_CLIP_COLUMNS = MANIFEST_COLUMNS + [
    "source_session_id",
    "detection_status",
    "detected_index",
    "detected_time_seconds",
    "detected_sample_index",
    "oracle_event_index",
    "oracle_time_seconds",
    "segmentation_error_ms",
    "peak_strength",
    "threshold_ratio",
]


@dataclass(frozen=True)
class SegmentationParameters:
    sensitivity: float = 0.5
    min_gap_ms: float = 55.0
    tolerance_ms: float = 35.0
    max_peak_multiplier: float = 1.1
    pre_ms: float = 20.0
    post_ms: float = 45.0

    def as_dict(self) -> dict[str, float]:
        return {
            "sensitivity": self.sensitivity,
            "min_gap_ms": self.min_gap_ms,
            "tolerance_ms": self.tolerance_ms,
            "max_peak_multiplier": self.max_peak_multiplier,
            "pre_ms": self.pre_ms,
            "post_ms": self.post_ms,
        }


def load_alignment(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_events(alignment: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in alignment.get("keydown_alignments", []):
        if not item.get("within_audio", True):
            continue
        events.append(
            {
                "event_index": int(item["event_index"]),
                "key": "Space" if item.get("key") == " " else str(item.get("key", "")),
                "raw_key": item.get("key", ""),
                "char": item.get("char", ""),
                "code": item.get("code", ""),
                "audio_time_seconds": float(item["audio_time_seconds"]),
                "sample_index": int(item["sample_index"]),
                "keydown_time_seconds": float(item["keydown_time_seconds"]),
            }
        )
    return events


def match_peaks_to_oracle(
    truth: list[dict[str, Any]],
    detected: list[DetectedPeak],
    tolerance_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[float, int, int]] = []
    for oracle_index, event in enumerate(truth):
        oracle_time = float(event["audio_time_seconds"])
        for detected_index, peak in enumerate(detected):
            error = abs(float(peak.time_seconds) - oracle_time)
            if error <= tolerance_seconds:
                candidates.append((error, oracle_index, detected_index))

    candidates.sort(key=lambda item: item[0])
    used_oracle: set[int] = set()
    used_detected: set[int] = set()
    matches: list[dict[str, Any]] = []

    for error, oracle_index, detected_index in candidates:
        if oracle_index in used_oracle or detected_index in used_detected:
            continue
        used_oracle.add(oracle_index)
        used_detected.add(detected_index)
        event = truth[oracle_index]
        peak = detected[detected_index]
        matches.append(
            {
                "status": "matched",
                "oracle": event,
                "detected": peak,
                "detected_index": detected_index,
                "error_seconds": error,
                "error_ms": round(error * 1000.0, 3),
            }
        )

    false_negatives = [
        {
            "status": "false_negative",
            "oracle": event,
        }
        for index, event in enumerate(truth)
        if index not in used_oracle
    ]
    false_positives = [
        {
            "status": "false_positive",
            "detected": peak,
            "detected_index": index,
        }
        for index, peak in enumerate(detected)
        if index not in used_detected
    ]

    matches.sort(key=lambda item: item["oracle"]["audio_time_seconds"])
    false_negatives.sort(key=lambda item: item["oracle"]["audio_time_seconds"])
    false_positives.sort(key=lambda item: item["detected"].time_seconds)
    return matches, false_positives, false_negatives


def metrics_from_counts(matched: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    precision = matched / (matched + false_positives) if matched + false_positives else 0.0
    recall = matched / (matched + false_negatives) if matched + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def summarize_error_ms(errors: list[float]) -> dict[str, float | None]:
    if not errors:
        return {
            "mean_abs_error_ms": None,
            "median_abs_error_ms": None,
            "p95_abs_error_ms": None,
        }
    values = np.array(errors, dtype=np.float32)
    return {
        "mean_abs_error_ms": round(float(np.mean(values)), 3),
        "median_abs_error_ms": round(float(np.median(values)), 3),
        "p95_abs_error_ms": round(float(np.quantile(values, 0.95)), 3),
    }


def trial_summary(
    alignment: dict[str, Any],
    matches: list[dict[str, Any]],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
    truth_count: int,
    detected_count: int,
) -> dict[str, Any]:
    summary = {
        "session_id": alignment["session_id"],
        "trial_id": alignment["trial_id"],
        "prompt_text": alignment.get("prompt_text", ""),
        "typed_text": alignment.get("typed_text", ""),
        "true_key_count": truth_count,
        "detected_key_count": detected_count,
        "matched_count": len(matches),
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
    }
    summary.update(metrics_from_counts(len(matches), len(false_positives), len(false_negatives)))
    summary.update(summarize_error_ms([float(item["error_ms"]) for item in matches]))
    return summary


def evaluate_trial(
    alignment_path: Path,
    parameters: SegmentationParameters,
    raw_sessions_dir: Path | None = None,
) -> dict[str, Any]:
    alignment = load_alignment(alignment_path)
    raw_root = raw_sessions_dir or RAW_DATA_DIR / "sessions"
    audio_path = raw_root / str(alignment["session_id"]) / str(alignment["audio_file_path"])
    sample_rate, samples = read_wav_mono_float(audio_path)
    truth = oracle_events(alignment)
    max_peaks = max(1, int(math.ceil(len(truth) * parameters.max_peak_multiplier)))
    peaks = detect_keystroke_peaks(
        samples=samples,
        sample_rate=sample_rate,
        sensitivity=parameters.sensitivity,
        min_gap_ms=parameters.min_gap_ms,
        max_peaks=max_peaks,
    )
    matches, false_positives, false_negatives = match_peaks_to_oracle(
        truth=truth,
        detected=peaks,
        tolerance_seconds=parameters.tolerance_ms / 1000.0,
    )

    return {
        "alignment_path": str(alignment_path),
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "alignment": alignment,
        "samples": samples,
        "truth": truth,
        "peaks": peaks,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "summary": trial_summary(
            alignment=alignment,
            matches=matches,
            false_positives=false_positives,
            false_negatives=false_negatives,
            truth_count=len(truth),
            detected_count=len(peaks),
        ),
    }


def aggregate_summary(trials: list[dict[str, Any]], parameters: SegmentationParameters) -> dict[str, Any]:
    true_count = sum(int(trial["summary"]["true_key_count"]) for trial in trials)
    detected_count = sum(int(trial["summary"]["detected_key_count"]) for trial in trials)
    matched_count = sum(int(trial["summary"]["matched_count"]) for trial in trials)
    false_positive_count = sum(int(trial["summary"]["false_positive_count"]) for trial in trials)
    false_negative_count = sum(int(trial["summary"]["false_negative_count"]) for trial in trials)
    errors = [
        float(match["error_ms"])
        for trial in trials
        for match in trial["matches"]
    ]
    summary: dict[str, Any] = {
        "trial_count": len(trials),
        "true_key_count": true_count,
        "detected_key_count": detected_count,
        "matched_count": matched_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "parameters": parameters.as_dict(),
    }
    summary.update(metrics_from_counts(matched_count, false_positive_count, false_negative_count))
    summary.update(summarize_error_ms(errors))
    return summary


def attach_acoustic_predictions(
    trials: list[dict[str, Any]],
    predictor: AcousticCNNPredictor,
    parameters: SegmentationParameters,
) -> dict[str, Any]:
    config = load_config()
    audio_config = config.get("audio", {})
    feature_config = config.get("features", {})
    target_sample_rate = int(audio_config.get("sample_rate", 48000))
    mel_bands = int(feature_config.get("mel_bands", 64))
    fft_window_size = int(feature_config.get("fft_window_size", 1024))
    hop_length = int(feature_config.get("hop_length", 256))

    pending: list[tuple[dict[str, Any], np.ndarray]] = []
    for trial in trials:
        samples = trial["samples"]
        for match in trial["matches"]:
            sample_rate = int(trial["sample_rate"])
            peak = match["detected"]
            clip = extract_fixed_window(
                samples=samples,
                center_sample=peak.sample_index,
                sample_rate=sample_rate,
                pre_ms=parameters.pre_ms,
                post_ms=parameters.post_ms,
            )
            clip_sample_rate = sample_rate
            if sample_rate != target_sample_rate:
                from keyboard_fusion.acoustic_inference import resample_linear

                clip = resample_linear(clip, sample_rate, target_sample_rate)
                clip_sample_rate = target_sample_rate
            spectrogram = predictor.spectrogram_from_clip(
                clip_samples=clip,
                sample_rate=clip_sample_rate,
                mel_bands=mel_bands,
                fft_window_size=fft_window_size,
                hop_length=hop_length,
            )
            pending.append((match, spectrogram))

    top_predictions = predictor.predict_spectrograms([item[1] for item in pending], top_k=5)
    top1_count = 0
    top5_count = 0
    for (match, _), top in zip(pending, top_predictions):
        true_key = str(match["oracle"]["key"])
        predicted_key = top[0]["key"] if top else ""
        top_keys = [item["key"] for item in top]
        correct_top1 = int(predicted_key == true_key)
        true_in_top5 = int(true_key in top_keys)
        top1_count += correct_top1
        top5_count += true_in_top5
        match["acoustic_prediction"] = {
            "predicted_key": predicted_key,
            "top1_probability": top[0]["probability"] if top else 0.0,
            "correct_top1": correct_top1,
            "true_in_top5": true_in_top5,
            "top": top,
        }

    total = len(pending)
    return {
        "model_dir": str(predictor.model_dir),
        "matched_detected_clips": total,
        "top1_accuracy_on_matched_detections": round(top1_count / total, 6) if total else 0.0,
        "top5_accuracy_on_matched_detections": round(top5_count / total, 6) if total else 0.0,
    }


def evaluate_alignment_paths(
    alignment_paths: list[Path],
    parameters: SegmentationParameters,
    raw_sessions_dir: Path | None = None,
    predictor: AcousticCNNPredictor | None = None,
) -> dict[str, Any]:
    trials = [
        evaluate_trial(
            alignment_path=path,
            parameters=parameters,
            raw_sessions_dir=raw_sessions_dir,
        )
        for path in sorted(alignment_paths)
    ]
    summary = aggregate_summary(trials, parameters)
    acoustic_summary = None
    if predictor is not None:
        acoustic_summary = attach_acoustic_predictions(trials, predictor, parameters)
        summary["acoustic_prediction"] = acoustic_summary
    return {
        "summary": summary,
        "trials": trials,
        "acoustic_summary": acoustic_summary,
    }


def tune_parameters(
    alignment_paths: list[Path],
    sensitivities: list[float] | None = None,
    min_gap_values_ms: list[float] | None = None,
    max_peak_multipliers: list[float] | None = None,
    tolerance_ms: float = 35.0,
    raw_sessions_dir: Path | None = None,
) -> dict[str, Any]:
    sensitivities = sensitivities or [0.5, 0.6, 0.8, 1.0, 1.2]
    min_gap_values_ms = min_gap_values_ms or [38.0, 45.0, 55.0, 65.0]
    max_peak_multipliers = max_peak_multipliers or [1.0, 1.1, 1.2, 1.5]
    results: list[dict[str, Any]] = []

    for sensitivity in sensitivities:
        for min_gap_ms in min_gap_values_ms:
            for multiplier in max_peak_multipliers:
                parameters = SegmentationParameters(
                    sensitivity=sensitivity,
                    min_gap_ms=min_gap_ms,
                    tolerance_ms=tolerance_ms,
                    max_peak_multiplier=multiplier,
                )
                evaluation = evaluate_alignment_paths(
                    alignment_paths=alignment_paths,
                    parameters=parameters,
                    raw_sessions_dir=raw_sessions_dir,
                    predictor=None,
                )
                result = {
                    **evaluation["summary"],
                    "parameters": parameters.as_dict(),
                }
                results.append(result)

    def score(result: dict[str, Any]) -> tuple[float, float, float]:
        mean_error = result.get("mean_abs_error_ms")
        error_score = -float(mean_error if mean_error is not None else 999.0)
        return (float(result["f1"]), float(result["precision"]), error_score)

    best = max(results, key=score)
    return {
        "best_parameters": best["parameters"],
        "best_summary": best,
        "grid_results": sorted(results, key=score, reverse=True),
    }


def match_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        alignment = trial["alignment"]
        session_id = alignment["session_id"]
        trial_id = alignment["trial_id"]
        for match in trial["matches"]:
            oracle = match["oracle"]
            detected = match["detected"]
            acoustic = match.get("acoustic_prediction", {})
            rows.append(
                {
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "status": "matched",
                    "oracle_event_index": oracle["event_index"],
                    "oracle_key": oracle["key"],
                    "oracle_code": oracle["code"],
                    "oracle_time_seconds": oracle["audio_time_seconds"],
                    "detected_index": match["detected_index"],
                    "detected_time_seconds": detected.time_seconds,
                    "detected_sample_index": detected.sample_index,
                    "error_ms": match["error_ms"],
                    "peak_strength": round(float(detected.strength), 9),
                    "threshold_ratio": detected.threshold_ratio,
                    "predicted_key": acoustic.get("predicted_key", ""),
                    "top1_probability": acoustic.get("top1_probability", ""),
                    "true_in_top5": acoustic.get("true_in_top5", ""),
                }
            )
        for item in trial["false_positives"]:
            detected = item["detected"]
            rows.append(
                {
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "status": "false_positive",
                    "oracle_event_index": "",
                    "oracle_key": "",
                    "oracle_code": "",
                    "oracle_time_seconds": "",
                    "detected_index": item["detected_index"],
                    "detected_time_seconds": detected.time_seconds,
                    "detected_sample_index": detected.sample_index,
                    "error_ms": "",
                    "peak_strength": round(float(detected.strength), 9),
                    "threshold_ratio": detected.threshold_ratio,
                    "predicted_key": "",
                    "top1_probability": "",
                    "true_in_top5": "",
                }
            )
        for item in trial["false_negatives"]:
            oracle = item["oracle"]
            rows.append(
                {
                    "session_id": session_id,
                    "trial_id": trial_id,
                    "status": "false_negative",
                    "oracle_event_index": oracle["event_index"],
                    "oracle_key": oracle["key"],
                    "oracle_code": oracle["code"],
                    "oracle_time_seconds": oracle["audio_time_seconds"],
                    "detected_index": "",
                    "detected_time_seconds": "",
                    "detected_sample_index": "",
                    "error_ms": "",
                    "peak_strength": "",
                    "threshold_ratio": "",
                    "predicted_key": "",
                    "top1_probability": "",
                    "true_in_top5": "",
                }
            )
    return sorted(rows, key=lambda row: (str(row["session_id"]), str(row["trial_id"]), str(row["detected_time_seconds"])))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def detected_clip_record(
    trial: dict[str, Any],
    detection: dict[str, Any],
    output_session_id: str,
    clip_path: Path,
    clip_id: str,
    window_start_sample: int,
    window_end_sample: int,
) -> dict[str, Any]:
    alignment = trial["alignment"]
    sample_rate = int(trial["sample_rate"])
    status = detection["status"]
    oracle = detection.get("oracle", {})
    detected = detection["detected"]
    key = oracle.get("raw_key", "")
    error_ms = detection.get("error_ms", "")
    return {
        "clip_id": clip_id,
        "session_id": output_session_id,
        "trial_id": alignment["trial_id"],
        "event_index": detection.get("detected_index", ""),
        "key": key,
        "char": oracle.get("char", ""),
        "code": oracle.get("code", ""),
        "source_audio_path": trial["audio_path"],
        "clip_audio_path": str(clip_path),
        "keydown_time_seconds": oracle.get("keydown_time_seconds", detected.time_seconds),
        "audio_time_seconds": detected.time_seconds,
        "audio_start_offset_seconds": alignment.get("offset_estimate", {}).get("offset_seconds", ""),
        "window_start_seconds": round(window_start_sample / sample_rate, 9),
        "window_end_seconds": round(window_end_sample / sample_rate, 9),
        "window_duration_seconds": round((window_end_sample - window_start_sample) / sample_rate, 9),
        "isolation_start_seconds": round(window_start_sample / sample_rate, 9),
        "isolation_end_seconds": round(window_end_sample / sample_rate, 9),
        "isolation_duration_seconds": round((window_end_sample - window_start_sample) / sample_rate, 9),
        "sample_index": detected.sample_index,
        "window_start_sample": window_start_sample,
        "window_end_sample": window_end_sample,
        "isolation_start_sample": window_start_sample,
        "isolation_end_sample": window_end_sample,
        "overlap_adjusted_left": False,
        "overlap_adjusted_right": False,
        "previous_key_gap_seconds": "",
        "next_key_gap_seconds": "",
        "sample_rate": sample_rate,
        "channels": 1,
        "prompt_set": alignment.get("prompt_set", ""),
        "prompt_index": alignment.get("prompt_index", ""),
        "prompt_text": alignment.get("prompt_text", ""),
        "source_session_id": alignment["session_id"],
        "detection_status": status,
        "detected_index": detection.get("detected_index", ""),
        "detected_time_seconds": detected.time_seconds,
        "detected_sample_index": detected.sample_index,
        "oracle_event_index": oracle.get("event_index", ""),
        "oracle_time_seconds": oracle.get("audio_time_seconds", ""),
        "segmentation_error_ms": error_ms,
        "peak_strength": round(float(detected.strength), 9),
        "threshold_ratio": detected.threshold_ratio,
    }


def extract_detected_clips(
    trials: list[dict[str, Any]],
    output_session_id: str,
    output_root: Path | None = None,
    include_false_positives: bool = True,
    parameters: SegmentationParameters | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    parameters = parameters or SegmentationParameters()
    output_base = output_root or PROCESSED_DATA_DIR / "detected_clips"
    output_dir = output_base / output_session_id
    rows: list[dict[str, Any]] = []
    pre_samples_by_rate: dict[int, int] = {}
    post_samples_by_rate: dict[int, int] = {}

    for trial in trials:
        sample_rate = int(trial["sample_rate"])
        pre_samples = pre_samples_by_rate.setdefault(
            sample_rate,
            int(round(sample_rate * parameters.pre_ms / 1000.0)),
        )
        post_samples = post_samples_by_rate.setdefault(
            sample_rate,
            int(round(sample_rate * parameters.post_ms / 1000.0)),
        )
        detections: list[dict[str, Any]] = list(trial["matches"])
        if include_false_positives:
            detections.extend(trial["false_positives"])
        detections.sort(key=lambda item: item["detected"].sample_index)

        for local_index, detection in enumerate(detections):
            detected = detection["detected"]
            oracle = detection.get("oracle", {})
            key_label = safe_label(oracle.get("raw_key") or detection["status"])
            source_session_id = trial["alignment"]["session_id"]
            trial_id = trial["alignment"]["trial_id"]
            clip_id = f"{source_session_id}_{trial_id}_detected_{local_index:03d}_{key_label}"
            clip_path = output_dir / trial_id / f"{clip_id}.wav"
            clip = extract_fixed_window(
                trial["samples"],
                center_sample=detected.sample_index,
                sample_rate=sample_rate,
                pre_ms=parameters.pre_ms,
                post_ms=parameters.post_ms,
            )
            write_float_wav(clip_path, clip, sample_rate)
            window_start = max(0, int(detected.sample_index) - pre_samples)
            window_end = window_start + pre_samples + post_samples
            rows.append(
                detected_clip_record(
                    trial=trial,
                    detection=detection,
                    output_session_id=output_session_id,
                    clip_path=clip_path,
                    clip_id=clip_id,
                    window_start_sample=window_start,
                    window_end_sample=window_end,
                )
            )

    manifest_path = output_dir / "clip_manifest.csv"
    write_csv_rows(manifest_path, rows, DETECTED_CLIP_COLUMNS)
    return rows, manifest_path


def json_safe_trial(trial: dict[str, Any]) -> dict[str, Any]:
    safe = {
        "alignment_path": trial["alignment_path"],
        "audio_path": trial["audio_path"],
        "summary": trial["summary"],
        "matches": [],
        "false_positives": [],
        "false_negatives": [],
    }
    for match in trial["matches"]:
        safe["matches"].append(
            {
                "status": "matched",
                "oracle": match["oracle"],
                "detected_index": match["detected_index"],
                "detected": match["detected"].__dict__,
                "error_ms": match["error_ms"],
                "acoustic_prediction": match.get("acoustic_prediction"),
            }
        )
    for item in trial["false_positives"]:
        safe["false_positives"].append(
            {
                "status": "false_positive",
                "detected_index": item["detected_index"],
                "detected": item["detected"].__dict__,
            }
        )
    for item in trial["false_negatives"]:
        safe["false_negatives"].append(
            {
                "status": "false_negative",
                "oracle": item["oracle"],
            }
        )
    return safe


def build_text_report(evaluation: dict[str, Any], output_session_id: str) -> str:
    summary = evaluation["summary"]
    lines = [
        "Automatic Keystroke Segmentation Report",
        "========================================",
        "",
        f"Session: {output_session_id}",
        f"Trials: {summary['trial_count']}",
        f"True keydowns: {summary['true_key_count']}",
        f"Detected peaks: {summary['detected_key_count']}",
        f"Matched: {summary['matched_count']}",
        f"False positives: {summary['false_positive_count']}",
        f"False negatives: {summary['false_negative_count']}",
        f"Precision: {summary['precision']:.3f}",
        f"Recall: {summary['recall']:.3f}",
        f"F1: {summary['f1']:.3f}",
        f"Mean abs timing error: {summary['mean_abs_error_ms']} ms",
        f"Median abs timing error: {summary['median_abs_error_ms']} ms",
        f"P95 abs timing error: {summary['p95_abs_error_ms']} ms",
        "",
        "Parameters:",
    ]
    for key, value in summary["parameters"].items():
        lines.append(f"- {key}: {value}")

    acoustic = summary.get("acoustic_prediction")
    if acoustic:
        lines.extend(
            [
                "",
                "Acoustic CNN On Matched Detected Clips:",
                f"- Model: {acoustic['model_dir']}",
                f"- Matched detected clips: {acoustic['matched_detected_clips']}",
                f"- Top-1: {acoustic['top1_accuracy_on_matched_detections']:.3f}",
                f"- Top-5: {acoustic['top5_accuracy_on_matched_detections']:.3f}",
            ]
        )

    lines.extend(["", "Worst trials by F1:"])
    trial_summaries = sorted(
        (trial["summary"] for trial in evaluation["trials"]),
        key=lambda item: (float(item["f1"]), -int(item["true_key_count"])),
    )
    for item in trial_summaries[:20]:
        lines.append(
            f"- {item['session_id']} / {item['trial_id']}: "
            f"F1={item['f1']:.3f}, P={item['precision']:.3f}, R={item['recall']:.3f}, "
            f"true={item['true_key_count']}, detected={item['detected_key_count']}, "
            f"matched={item['matched_count']}"
        )
    return "\n".join(lines)


def write_evaluation_outputs(
    evaluation: dict[str, Any],
    output_session_id: str,
    output_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    output_base = output_root or METADATA_DIR / "segmentation"
    output_dir = output_base / output_session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = output_dir / "segmentation_report.json"
    report_txt_path = output_dir / "segmentation_report.txt"
    matches_csv_path = output_dir / "segmentation_matches.csv"
    payload = {
        "summary": evaluation["summary"],
        "trials": [json_safe_trial(trial) for trial in evaluation["trials"]],
    }
    report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_txt_path.write_text(build_text_report(evaluation, output_session_id), encoding="utf-8")
    write_csv_rows(matches_csv_path, match_rows(evaluation["trials"]), MATCH_COLUMNS)
    return report_json_path, report_txt_path, matches_csv_path


def find_alignment_paths(session: str | None = None, all_sessions: bool = False) -> tuple[str, list[Path]]:
    root = METADATA_DIR / "alignment"
    if all_sessions:
        paths = sorted(root.glob("session_*/trial_*_alignment.json"))
        if not paths:
            raise FileNotFoundError(f"No alignment files found under {root}")
        return DEFAULT_SEGMENTATION_SESSION_ID, paths
    if session:
        session_dir = root / session
    else:
        sessions = sorted(path for path in root.iterdir() if path.is_dir())
        if not sessions:
            raise FileNotFoundError(f"No alignment sessions found under {root}")
        session_dir = sessions[-1]
    paths = sorted(session_dir.glob("trial_*_alignment.json"))
    if not paths:
        raise FileNotFoundError(f"No alignment files found in {session_dir}")
    return session_dir.name, paths
