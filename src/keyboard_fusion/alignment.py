from __future__ import annotations

import csv
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyboard_fusion.paths import METADATA_DIR, RAW_DATA_DIR


ALIGNMENT_METHOD = "shared_browser_trial_clock_v1"


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    frame_count: int
    sample_width_bytes: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate


def read_wav_info(path: Path) -> WavInfo:
    """Read basic WAV information without loading the whole recording."""
    with wave.open(str(path), "rb") as wav_file:
        return WavInfo(
            sample_rate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            frame_count=wav_file.getnframes(),
            sample_width_bytes=wav_file.getsampwidth(),
        )


def load_events_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_metadata_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_repeat(event: dict[str, Any]) -> bool:
    value = event.get("repeat", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def keydown_events(events: list[dict[str, Any]], include_repeats: bool = False) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("event_type") == "keydown" and (include_repeats or not _is_repeat(event))
    ]


def align_keydown_event(
    event: dict[str, Any],
    sample_rate: int,
    frame_count: int,
    pre_keydown_ms: float,
    post_keydown_ms: float,
) -> dict[str, Any]:
    """Map one keydown event timestamp to a WAV sample index and extraction window."""
    keydown_time_seconds = _parse_float(event.get("trial_elapsed_seconds"))
    sample_index = int(round(keydown_time_seconds * sample_rate))
    pre_samples = int(round(sample_rate * pre_keydown_ms / 1000))
    post_samples = int(round(sample_rate * post_keydown_ms / 1000))

    raw_start = sample_index - pre_samples
    raw_end = sample_index + post_samples
    window_start_sample = min(max(raw_start, 0), frame_count)
    window_end_sample = min(max(raw_end, 0), frame_count)
    if window_end_sample < window_start_sample:
        window_end_sample = window_start_sample

    return {
        "event_index": _parse_int(event.get("event_index")),
        "key": event.get("key", ""),
        "char": event.get("char", ""),
        "code": event.get("code", ""),
        "keydown_time_seconds": round(keydown_time_seconds, 9),
        "sample_index": sample_index,
        "window_start_sample": window_start_sample,
        "window_end_sample": window_end_sample,
        "window_start_seconds": round(window_start_sample / sample_rate, 9),
        "window_end_seconds": round(window_end_sample / sample_rate, 9),
        "window_duration_seconds": round((window_end_sample - window_start_sample) / sample_rate, 9),
        "within_audio": 0 <= sample_index < frame_count,
        "clipped_left": raw_start < 0,
        "clipped_right": raw_end > frame_count,
    }


def align_trial(
    metadata_path: Path,
    pre_keydown_ms: float,
    post_keydown_ms: float,
) -> dict[str, Any]:
    """Build alignment metadata for one saved trial."""
    metadata = load_metadata_json(metadata_path)
    session_dir = metadata_path.parent
    audio_path = session_dir / str(metadata["audio_file_path"])
    events_path = session_dir / str(metadata["events_file_path"])
    wav_info = read_wav_info(audio_path)
    events = load_events_csv(events_path)
    keydowns = keydown_events(events)
    keydown_alignments = [
        align_keydown_event(
            event,
            wav_info.sample_rate,
            wav_info.frame_count,
            pre_keydown_ms,
            post_keydown_ms,
        )
        for event in keydowns
    ]
    outside_audio_count = sum(1 for item in keydown_alignments if not item["within_audio"])
    clipped_count = sum(1 for item in keydown_alignments if item["clipped_left"] or item["clipped_right"])
    keydown_times = [item["keydown_time_seconds"] for item in keydown_alignments]
    metadata_duration = _parse_float(metadata.get("duration_seconds"))

    return {
        "alignment_version": 1,
        "alignment_method": ALIGNMENT_METHOD,
        "alignment_assumption": (
            "WAV sample 0 is treated as approximately equal to trial_elapsed_seconds=0 "
            "from the browser Start Trial click. This is good enough for oracle-window "
            "experiments, but a beep or impulse marker can be added later for tighter "
            "audio-clock calibration."
        ),
        "session_id": metadata.get("session_id"),
        "trial_id": metadata.get("trial_id"),
        "participant_id": metadata.get("participant_id"),
        "prompt_set": metadata.get("prompt_set"),
        "prompt_index": metadata.get("prompt_index"),
        "prompt_text": metadata.get("prompt_text"),
        "typed_text": metadata.get("typed_text"),
        "typed_matches_prompt": metadata.get("typed_text") == metadata.get("prompt_text"),
        "audio_file_path": metadata.get("audio_file_path"),
        "events_file_path": metadata.get("events_file_path"),
        "audio_input_device": metadata.get("audio_input_device") or {},
        "audio": {
            "sample_rate": wav_info.sample_rate,
            "channels": wav_info.channels,
            "frame_count": wav_info.frame_count,
            "sample_width_bytes": wav_info.sample_width_bytes,
            "duration_seconds": round(wav_info.duration_seconds, 9),
        },
        "trial_timing": {
            "metadata_duration_seconds": metadata_duration,
            "wav_duration_seconds": round(wav_info.duration_seconds, 9),
            "metadata_minus_wav_duration_seconds": round(metadata_duration - wav_info.duration_seconds, 9),
            "first_keydown_seconds": min(keydown_times) if keydown_times else None,
            "last_keydown_seconds": max(keydown_times) if keydown_times else None,
        },
        "window": {
            "pre_keydown_ms": pre_keydown_ms,
            "post_keydown_ms": post_keydown_ms,
            "pre_keydown_samples": int(round(wav_info.sample_rate * pre_keydown_ms / 1000)),
            "post_keydown_samples": int(round(wav_info.sample_rate * post_keydown_ms / 1000)),
        },
        "event_counts": {
            "total_events": len(events),
            "keydown_events": len(keydowns),
            "aligned_keydown_events": len(keydown_alignments),
            "outside_audio_keydown_events": outside_audio_count,
            "clipped_window_events": clipped_count,
        },
        "keydown_alignments": keydown_alignments,
    }


def find_latest_session(raw_sessions_dir: Path | None = None) -> Path:
    sessions_dir = raw_sessions_dir or RAW_DATA_DIR / "sessions"
    sessions = sorted(path for path in sessions_dir.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No session directories found under {sessions_dir}")
    return sessions[-1]


def write_trial_alignment(alignment: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_id = str(alignment["trial_id"])
    output_path = output_dir / f"{trial_id}_alignment.json"
    output_path.write_text(json.dumps(alignment, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def build_session_report(alignments: list[dict[str, Any]]) -> str:
    lines = [
        "Audio/Event Alignment Report",
        "============================",
        "",
        f"Method: {ALIGNMENT_METHOD}",
        (
            "Assumption: WAV sample 0 approximately matches trial_elapsed_seconds=0 "
            "from the browser Start Trial click."
        ),
        "",
    ]

    for alignment in alignments:
        audio = alignment["audio"]
        timing = alignment["trial_timing"]
        counts = alignment["event_counts"]
        window = alignment["window"]
        lines.extend(
            [
                f"{alignment['trial_id']} ({alignment['prompt_set']} #{int(alignment['prompt_index']) + 1})",
                "-" * 72,
                f"Prompt: {alignment['prompt_text']}",
                f"Typed:  {alignment['typed_text']}",
                f"Typed matches prompt: {alignment['typed_matches_prompt']}",
                (
                    f"Audio: {audio['duration_seconds']:.3f}s, {audio['sample_rate']} Hz, "
                    f"{audio['channels']} channel(s), {audio['frame_count']} frames"
                ),
                (
                    "Browser duration minus WAV duration: "
                    f"{timing['metadata_minus_wav_duration_seconds']:.3f}s"
                ),
                (
                    f"Keydowns: {counts['aligned_keydown_events']} aligned, "
                    f"{counts['outside_audio_keydown_events']} outside audio, "
                    f"{counts['clipped_window_events']} clipped windows"
                ),
                (
                    f"Window: {window['pre_keydown_ms']} ms before to "
                    f"{window['post_keydown_ms']} ms after keydown"
                ),
                "idx  key      code        time_s    sample    window_s",
            ]
        )
        for item in alignment["keydown_alignments"]:
            key = item["key"] if item["key"] != " " else "Space"
            clipped = " clipped" if item["clipped_left"] or item["clipped_right"] else ""
            lines.append(
                f"{item['event_index']:>3}  {key:<8} {item['code']:<10} "
                f"{item['keydown_time_seconds']:>7.3f}  {item['sample_index']:>8}  "
                f"{item['window_start_seconds']:>7.3f}-{item['window_end_seconds']:<7.3f}"
                f"{clipped}"
            )
        lines.append("")

    return "\n".join(lines)


def align_session(
    session_dir: Path,
    pre_keydown_ms: float,
    post_keydown_ms: float,
    output_root: Path | None = None,
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Align all trials in a session and write JSON plus a text report."""
    output_base = output_root or METADATA_DIR / "alignment"
    output_dir = output_base / session_dir.name
    alignments = [
        align_trial(metadata_path, pre_keydown_ms, post_keydown_ms)
        for metadata_path in sorted(session_dir.glob("trial_*_metadata.json"))
    ]
    if not alignments:
        raise FileNotFoundError(f"No trial metadata files found in {session_dir}")

    for alignment in alignments:
        write_trial_alignment(alignment, output_dir)

    report_path = output_dir / "alignment_report.txt"
    report_path.write_text(build_session_report(alignments), encoding="utf-8")
    return alignments, output_dir, report_path
