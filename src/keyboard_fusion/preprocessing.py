from __future__ import annotations

import csv
import json
import re
import wave
from pathlib import Path
from typing import Any

from keyboard_fusion.paths import METADATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


MANIFEST_COLUMNS = [
    "clip_id",
    "session_id",
    "trial_id",
    "event_index",
    "key",
    "char",
    "code",
    "source_audio_path",
    "clip_audio_path",
    "keydown_time_seconds",
    "window_start_seconds",
    "window_end_seconds",
    "window_duration_seconds",
    "window_start_sample",
    "window_end_sample",
    "sample_rate",
    "channels",
    "prompt_set",
    "prompt_index",
    "prompt_text",
]


def safe_label(value: Any) -> str:
    """Make a short filesystem-safe label for a key or character."""
    raw_text = str(value or "")
    if raw_text == " ":
        return "space"
    text = raw_text.strip()
    if not text:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    cleaned = cleaned.strip("_").lower()
    return cleaned or "unknown"


def clip_id_for_keydown(trial_id: str, keydown: dict[str, Any]) -> str:
    event_index = int(keydown["event_index"])
    code_label = safe_label(keydown.get("code"))
    key_label = safe_label(keydown.get("key"))
    return f"{trial_id}_event_{event_index:03d}_{code_label}_{key_label}"


def write_wav_clip(source_audio_path: Path, output_path: Path, start_sample: int, end_sample: int) -> int:
    """Copy a sample window from a source WAV into a new WAV clip."""
    if end_sample < start_sample:
        raise ValueError("end_sample must be greater than or equal to start_sample")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source_audio_path), "rb") as source:
        frame_count = source.getnframes()
        bounded_start = max(0, min(start_sample, frame_count))
        bounded_end = max(bounded_start, min(end_sample, frame_count))
        frames_to_read = bounded_end - bounded_start
        source.setpos(bounded_start)
        frames = source.readframes(frames_to_read)

        with wave.open(str(output_path), "wb") as target:
            target.setnchannels(source.getnchannels())
            target.setsampwidth(source.getsampwidth())
            target.setframerate(source.getframerate())
            target.writeframes(frames)

    return frames_to_read


def load_alignment(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_clip_record(
    alignment: dict[str, Any],
    keydown: dict[str, Any],
    clip_id: str,
    source_audio_path: Path,
    clip_audio_path: Path,
    frame_count: int,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "session_id": alignment["session_id"],
        "trial_id": alignment["trial_id"],
        "event_index": keydown["event_index"],
        "key": keydown.get("key", ""),
        "char": keydown.get("char", ""),
        "code": keydown.get("code", ""),
        "source_audio_path": str(source_audio_path),
        "clip_audio_path": str(clip_audio_path),
        "keydown_time_seconds": keydown["keydown_time_seconds"],
        "window_start_seconds": keydown["window_start_seconds"],
        "window_end_seconds": keydown["window_end_seconds"],
        "window_duration_seconds": round(frame_count / alignment["audio"]["sample_rate"], 9),
        "window_start_sample": keydown["window_start_sample"],
        "window_end_sample": keydown["window_end_sample"],
        "sample_rate": alignment["audio"]["sample_rate"],
        "channels": alignment["audio"]["channels"],
        "prompt_set": alignment["prompt_set"],
        "prompt_index": alignment["prompt_index"],
        "prompt_text": alignment["prompt_text"],
    }


def extract_trial_clips(
    alignment_path: Path,
    raw_sessions_dir: Path | None = None,
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract all aligned keydown windows for one trial."""
    alignment = load_alignment(alignment_path)
    session_id = str(alignment["session_id"])
    trial_id = str(alignment["trial_id"])
    raw_root = raw_sessions_dir or RAW_DATA_DIR / "sessions"
    output_base = output_root or PROCESSED_DATA_DIR / "clips"
    source_audio_path = raw_root / session_id / str(alignment["audio_file_path"])
    if not source_audio_path.exists():
        raise FileNotFoundError(f"Missing source audio: {source_audio_path}")

    trial_output_dir = output_base / session_id / trial_id
    records: list[dict[str, Any]] = []
    for keydown in alignment["keydown_alignments"]:
        clip_id = clip_id_for_keydown(trial_id, keydown)
        clip_audio_path = trial_output_dir / f"{clip_id}.wav"
        frame_count = write_wav_clip(
            source_audio_path=source_audio_path,
            output_path=clip_audio_path,
            start_sample=int(keydown["window_start_sample"]),
            end_sample=int(keydown["window_end_sample"]),
        )
        records.append(
            build_clip_record(
                alignment=alignment,
                keydown=keydown,
                clip_id=clip_id,
                source_audio_path=source_audio_path,
                clip_audio_path=clip_audio_path,
                frame_count=frame_count,
            )
        )
    return records


def write_clip_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in MANIFEST_COLUMNS})


def find_latest_alignment_session(alignment_root: Path | None = None) -> Path:
    root = alignment_root or METADATA_DIR / "alignment"
    sessions = sorted(path for path in root.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No alignment sessions found under {root}")
    return sessions[-1]


def build_clip_report(records: list[dict[str, Any]]) -> str:
    trial_counts: dict[str, int] = {}
    key_counts: dict[str, int] = {}
    for record in records:
        trial_counts[str(record["trial_id"])] = trial_counts.get(str(record["trial_id"]), 0) + 1
        key = str(record["key"] or "unknown")
        key_counts[key] = key_counts.get(key, 0) + 1

    lines = [
        "Oracle Keystroke Clip Extraction Report",
        "========================================",
        "",
        f"Total clips: {len(records)}",
        f"Trials: {len(trial_counts)}",
        f"Unique key labels: {len(key_counts)}",
        "",
        "Clips per trial:",
    ]
    for trial_id in sorted(trial_counts):
        lines.append(f"- {trial_id}: {trial_counts[trial_id]}")

    lines.extend(["", "Key label counts:"])
    for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0])):
        label = "Space" if key == " " else key
        lines.append(f"- {label}: {count}")
    return "\n".join(lines)


def extract_session_clips(
    alignment_session_dir: Path,
    raw_sessions_dir: Path | None = None,
    output_root: Path | None = None,
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Extract clips for every trial alignment JSON in a session folder."""
    output_base = output_root or PROCESSED_DATA_DIR / "clips"
    session_output_dir = output_base / alignment_session_dir.name
    records: list[dict[str, Any]] = []

    alignment_paths = sorted(alignment_session_dir.glob("trial_*_alignment.json"))
    if not alignment_paths:
        raise FileNotFoundError(f"No trial alignment files found in {alignment_session_dir}")

    for alignment_path in alignment_paths:
        records.extend(
            extract_trial_clips(
                alignment_path=alignment_path,
                raw_sessions_dir=raw_sessions_dir,
                output_root=output_base,
            )
        )

    manifest_path = session_output_dir / "clip_manifest.csv"
    report_path = session_output_dir / "clip_extraction_report.txt"
    write_clip_manifest(records, manifest_path)
    report_path.write_text(build_clip_report(records), encoding="utf-8")
    return records, manifest_path, report_path
