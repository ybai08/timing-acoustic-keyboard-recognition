from __future__ import annotations

import csv
import json
import statistics
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keyboard_fusion.paths import METADATA_DIR, RAW_DATA_DIR


ALIGNMENT_METHOD = "audio_energy_offset_v2"


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


def read_wav_mono_samples(path: Path) -> tuple[WavInfo, list[int]]:
    """Read a WAV as mono integer samples for lightweight energy analysis."""
    with wave.open(str(path), "rb") as wav_file:
        wav_info = WavInfo(
            sample_rate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            frame_count=wav_file.getnframes(),
            sample_width_bytes=wav_file.getsampwidth(),
        )
        raw = wav_file.readframes(wav_info.frame_count)

    if wav_info.sample_width_bytes != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported for alignment offset estimation.")

    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if wav_info.channels == 1:
        return wav_info, list(values)

    mono_samples: list[int] = []
    for index in range(0, len(values), wav_info.channels):
        frame = values[index : index + wav_info.channels]
        mono_samples.append(int(sum(frame) / len(frame)))
    return wav_info, mono_samples


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


def build_energy_envelope(
    samples: list[int],
    sample_rate: int,
    hop_ms: float = 2.0,
    frame_ms: float = 8.0,
) -> tuple[list[float], int]:
    """Build a robust short-time absolute-energy envelope."""
    hop_samples = max(1, int(round(sample_rate * hop_ms / 1000)))
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000)))
    envelope: list[float] = []
    for start in range(0, len(samples), hop_samples):
        frame = samples[start : start + frame_samples]
        if not frame:
            break
        envelope.append(sum(abs(sample) for sample in frame) / len(frame))

    if not envelope:
        return [], hop_samples

    median = statistics.median(envelope)
    mad = statistics.median(abs(value - median) for value in envelope) or 1.0
    normalized = [max(0.0, (value - median) / mad) for value in envelope]
    return normalized, hop_samples


def score_audio_offset(
    envelope: list[float],
    hop_samples: int,
    sample_rate: int,
    keydown_times_seconds: list[float],
    offset_seconds: float,
    search_radius_ms: float = 25.0,
) -> float:
    """Score how well shifted keydown times land on audio energy peaks."""
    if not envelope:
        return float("-inf")

    radius_frames = max(1, int(round(sample_rate * search_radius_ms / 1000 / hop_samples)))
    scores: list[float] = []
    outside_count = 0

    for keydown_time in keydown_times_seconds:
        audio_time = keydown_time - offset_seconds
        if audio_time < 0:
            outside_count += 1
            continue
        center = int(round(audio_time * sample_rate / hop_samples))
        if center >= len(envelope):
            outside_count += 1
            continue
        start = max(0, center - radius_frames)
        end = min(len(envelope), center + radius_frames + 1)
        local_scores = [
            value * (1.0 - (abs(index - center) / (radius_frames + 1) * 0.5))
            for index, value in enumerate(envelope[start:end], start=start)
        ]
        scores.append(max(local_scores))

    if not scores:
        return float("-inf")
    return statistics.mean(scores) - (outside_count * 5.0)


def estimate_audio_start_offset(
    samples: list[int],
    sample_rate: int,
    keydown_times_seconds: list[float],
    metadata_minus_wav_duration_seconds: float,
    step_seconds: float = 0.001,
) -> dict[str, Any]:
    """Estimate how far the WAV clock starts after the browser trial clock.

    The browser key log uses `trial_elapsed_seconds`, but ScriptProcessor audio
    chunks can begin after that clock starts. This searches for the shift that
    makes keydown timestamps land on high-energy audio regions.
    """
    if not keydown_times_seconds:
        return {
            "mode": "audio_energy_search",
            "offset_seconds": 0.0,
            "score": 0.0,
            "search_start_seconds": 0.0,
            "search_end_seconds": 0.0,
            "step_seconds": step_seconds,
        }

    envelope, hop_samples = build_energy_envelope(samples, sample_rate)
    first_keydown = min(keydown_times_seconds)
    duration_based_guess = max(0.0, metadata_minus_wav_duration_seconds)
    search_end = min(
        max(0.0, first_keydown - 0.005),
        max(0.5, duration_based_guess + 0.75),
    )
    if search_end <= 0:
        search_end = min(0.25, max(keydown_times_seconds))

    best_offset = 0.0
    best_score = float("-inf")
    steps = max(1, int(round(search_end / step_seconds)))
    for step in range(steps + 1):
        offset = min(search_end, step * step_seconds)
        score = score_audio_offset(envelope, hop_samples, sample_rate, keydown_times_seconds, offset)
        if score > best_score:
            best_score = score
            best_offset = offset

    return {
        "mode": "audio_energy_search",
        "offset_seconds": round(best_offset, 9),
        "score": round(best_score, 6),
        "search_start_seconds": 0.0,
        "search_end_seconds": round(search_end, 9),
        "step_seconds": step_seconds,
        "metadata_minus_wav_duration_seconds": round(metadata_minus_wav_duration_seconds, 9),
    }


def align_keydown_event(
    event: dict[str, Any],
    sample_rate: int,
    frame_count: int,
    pre_keydown_ms: float,
    post_keydown_ms: float,
    audio_start_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    """Map one keydown event timestamp to a WAV sample index and extraction window."""
    keydown_time_seconds = _parse_float(event.get("trial_elapsed_seconds"))
    audio_time_seconds = keydown_time_seconds - audio_start_offset_seconds
    sample_index = int(round(audio_time_seconds * sample_rate))
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
        "audio_time_seconds": round(audio_time_seconds, 9),
        "audio_start_offset_seconds": round(audio_start_offset_seconds, 9),
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
    wav_info, samples = read_wav_mono_samples(audio_path)
    events = load_events_csv(events_path)
    keydowns = keydown_events(events)
    metadata_duration = _parse_float(metadata.get("duration_seconds"))
    metadata_minus_wav = metadata_duration - wav_info.duration_seconds
    keydown_trial_times = [_parse_float(event.get("trial_elapsed_seconds")) for event in keydowns]
    offset_estimate = estimate_audio_start_offset(
        samples=samples,
        sample_rate=wav_info.sample_rate,
        keydown_times_seconds=keydown_trial_times,
        metadata_minus_wav_duration_seconds=metadata_minus_wav,
    )
    audio_start_offset_seconds = float(offset_estimate["offset_seconds"])
    keydown_alignments = [
        align_keydown_event(
            event,
            wav_info.sample_rate,
            wav_info.frame_count,
            pre_keydown_ms,
            post_keydown_ms,
            audio_start_offset_seconds,
        )
        for event in keydowns
    ]
    outside_audio_count = sum(1 for item in keydown_alignments if not item["within_audio"])
    clipped_count = sum(1 for item in keydown_alignments if item["clipped_left"] or item["clipped_right"])
    next_key_overlap_count = sum(
        1
        for current, following in zip(keydown_alignments, keydown_alignments[1:])
        if following["audio_time_seconds"] < current["window_end_seconds"]
    )
    keydown_times = [item["keydown_time_seconds"] for item in keydown_alignments]

    return {
        "alignment_version": 2,
        "alignment_method": ALIGNMENT_METHOD,
        "alignment_assumption": (
            "WAV sample 0 can start after browser trial_elapsed_seconds=0. The alignment "
            "therefore estimates an audio_start_offset_seconds value by searching for the "
            "shift that places logged keydown times on high-energy audio regions. A beep "
            "or impulse marker can be added later for tighter calibration."
        ),
        "offset_estimate": offset_estimate,
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
            "metadata_minus_wav_duration_seconds": round(metadata_minus_wav, 9),
            "first_keydown_seconds": min(keydown_times) if keydown_times else None,
            "last_keydown_seconds": max(keydown_times) if keydown_times else None,
            "first_keydown_audio_seconds": min(
                (item["audio_time_seconds"] for item in keydown_alignments),
                default=None,
            ),
            "last_keydown_audio_seconds": max(
                (item["audio_time_seconds"] for item in keydown_alignments),
                default=None,
            ),
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
            "next_key_overlap_windows": next_key_overlap_count,
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
            "Assumption: WAV sample 0 may start after browser trial_elapsed_seconds=0; "
            "the script estimates that offset from audio energy."
        ),
        "",
    ]

    for alignment in alignments:
        audio = alignment["audio"]
        timing = alignment["trial_timing"]
        counts = alignment["event_counts"]
        window = alignment["window"]
        offset = alignment["offset_estimate"]
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
                    "Estimated audio start offset: "
                    f"{offset['offset_seconds']:.3f}s "
                    f"(score {offset['score']:.2f})"
                ),
                (
                    f"Keydowns: {counts['aligned_keydown_events']} aligned, "
                    f"{counts['outside_audio_keydown_events']} outside audio, "
                    f"{counts['clipped_window_events']} clipped windows, "
                    f"{counts['next_key_overlap_windows']} overlap next key"
                ),
                (
                    f"Window: {window['pre_keydown_ms']} ms before to "
                    f"{window['post_keydown_ms']} ms after keydown"
                ),
                "idx  key      code       trial_s  audio_s   sample    window_s",
            ]
        )
        for item in alignment["keydown_alignments"]:
            key = item["key"] if item["key"] != " " else "Space"
            clipped = " clipped" if item["clipped_left"] or item["clipped_right"] else ""
            lines.append(
                f"{item['event_index']:>3}  {key:<8} {item['code']:<10} "
                f"{item['keydown_time_seconds']:>7.3f}  {item['audio_time_seconds']:>7.3f}  "
                f"{item['sample_index']:>8}  "
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
