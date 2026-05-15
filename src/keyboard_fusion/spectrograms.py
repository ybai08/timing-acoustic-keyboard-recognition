from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any
import wave

import numpy as np

from keyboard_fusion.paths import PROCESSED_DATA_DIR


SPECTROGRAM_MANIFEST_COLUMNS = [
    "spectrogram_id",
    "spectrogram_path",
    "clip_id",
    "clip_audio_path",
    "session_id",
    "trial_id",
    "event_index",
    "key",
    "char",
    "code",
    "sample_rate",
    "mel_bands",
    "fft_window_size",
    "hop_length",
    "frames",
    "mean",
    "std",
    "prompt_set",
    "prompt_index",
    "prompt_text",
]


def hz_to_mel(frequency_hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def read_wav_mono_float(path: Path) -> tuple[int, np.ndarray]:
    """Read a 16-bit PCM WAV file into mono float32 samples in [-1, 1]."""
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported.")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0
    return sample_rate, samples


def frame_signal(samples: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    """Split a signal into overlapping frames, padding short clips as needed."""
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)))

    frame_count = 1 + int(np.ceil((len(samples) - frame_size) / hop_length))
    padded_length = frame_size + (frame_count - 1) * hop_length
    if len(samples) < padded_length:
        samples = np.pad(samples, (0, padded_length - len(samples)))

    frames = np.empty((frame_count, frame_size), dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_length
        frames[index] = samples[start : start + frame_size]
    return frames


def power_spectrogram(samples: np.ndarray, fft_window_size: int, hop_length: int) -> np.ndarray:
    frames = frame_signal(samples, fft_window_size, hop_length)
    window = np.hanning(fft_window_size).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, n=fft_window_size, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    return power.T


def mel_filterbank(
    sample_rate: int,
    fft_window_size: int,
    mel_bands: int,
    min_frequency_hz: float = 0.0,
    max_frequency_hz: float | None = None,
) -> np.ndarray:
    """Create triangular mel filters with shape (mel_bands, fft_bins)."""
    if mel_bands <= 0:
        raise ValueError("mel_bands must be positive")

    max_frequency_hz = max_frequency_hz or sample_rate / 2
    min_mel = hz_to_mel(min_frequency_hz)
    max_mel = hz_to_mel(max_frequency_hz)
    mel_points = np.linspace(min_mel, max_mel, mel_bands + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((fft_window_size + 1) * hz_points / sample_rate).astype(int)
    bin_count = fft_window_size // 2 + 1
    bin_indices = np.clip(bin_indices, 0, bin_count - 1)

    filters = np.zeros((mel_bands, bin_count), dtype=np.float32)
    for band in range(1, mel_bands + 1):
        left = bin_indices[band - 1]
        center = bin_indices[band]
        right = bin_indices[band + 1]

        if center == left:
            center = min(center + 1, bin_count - 1)
        if right == center:
            right = min(right + 1, bin_count)

        for bin_index in range(left, center):
            filters[band - 1, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, right):
            filters[band - 1, bin_index] = (right - bin_index) / max(right - center, 1)

    return filters


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    mel_bands: int,
    fft_window_size: int,
    hop_length: int,
) -> np.ndarray:
    power = power_spectrogram(samples, fft_window_size, hop_length)
    filters = mel_filterbank(sample_rate, fft_window_size, mel_bands)
    mel_power = filters @ power
    return np.log1p(mel_power).astype(np.float32)


def normalize_spectrogram(spectrogram: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(spectrogram))
    std = float(np.std(spectrogram))
    if std < 1e-8:
        return np.zeros_like(spectrogram, dtype=np.float32), mean, std
    normalized = ((spectrogram - mean) / std).astype(np.float32)
    return normalized, mean, std


def load_clip_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_spectrogram_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SPECTROGRAM_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in SPECTROGRAM_MANIFEST_COLUMNS})


def spectrogram_id_for_clip(clip_id: str) -> str:
    return f"{clip_id}_logmel"


def spectrogram_path_for_record(record: dict[str, str], output_root: Path) -> Path:
    session_id = record["session_id"]
    trial_id = record["trial_id"]
    spectrogram_id = spectrogram_id_for_clip(record["clip_id"])
    return output_root / session_id / trial_id / f"{spectrogram_id}.npz"


def generate_spectrogram_for_clip(
    record: dict[str, str],
    output_root: Path,
    mel_bands: int,
    fft_window_size: int,
    hop_length: int,
) -> tuple[dict[str, Any], np.ndarray]:
    clip_audio_path = Path(record["clip_audio_path"])
    sample_rate, samples = read_wav_mono_float(clip_audio_path)
    raw_log_mel = log_mel_spectrogram(
        samples=samples,
        sample_rate=sample_rate,
        mel_bands=mel_bands,
        fft_window_size=fft_window_size,
        hop_length=hop_length,
    )
    normalized, mean, std = normalize_spectrogram(raw_log_mel)

    output_path = spectrogram_path_for_record(record, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        spectrogram=normalized,
        log_mel=raw_log_mel,
        sample_rate=np.array(sample_rate),
        mel_bands=np.array(mel_bands),
        fft_window_size=np.array(fft_window_size),
        hop_length=np.array(hop_length),
    )

    spectrogram_record = {
        **record,
        "spectrogram_id": spectrogram_id_for_clip(record["clip_id"]),
        "spectrogram_path": str(output_path),
        "sample_rate": sample_rate,
        "mel_bands": mel_bands,
        "fft_window_size": fft_window_size,
        "hop_length": hop_length,
        "frames": normalized.shape[1],
        "mean": round(mean, 9),
        "std": round(std, 9),
    }
    return spectrogram_record, normalized


def spectrogram_to_svg(spectrogram: np.ndarray, width: int = 260, height: int = 130) -> str:
    """Render a compact inline SVG heatmap for quick visual inspection."""
    values = spectrogram.astype(np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        scaled = np.zeros_like(values)
    else:
        scaled = (values - minimum) / (maximum - minimum)

    rows, cols = scaled.shape
    cell_width = width / cols
    cell_height = height / rows
    rects: list[str] = []
    for row in range(rows):
        y = (rows - row - 1) * cell_height
        for col in range(cols):
            value = float(scaled[row, col])
            lightness = 12 + value * 72
            color = f"hsl(185 85% {lightness:.1f}%)"
            rects.append(
                f'<rect x="{col * cell_width:.2f}" y="{y:.2f}" '
                f'width="{cell_width + 0.2:.2f}" height="{cell_height + 0.2:.2f}" '
                f'fill="{color}" />'
            )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="log mel spectrogram">{"".join(rects)}</svg>'
    )


def build_preview_html(previews: list[tuple[dict[str, Any], np.ndarray]]) -> str:
    cards: list[str] = []
    for record, spectrogram in previews:
        key = "Space" if record.get("key") == " " else str(record.get("key", ""))
        title = html.escape(f"{record['clip_id']} | key={key}")
        subtitle = html.escape(f"{record['trial_id']} | frames={record['frames']} | shape={spectrogram.shape}")
        cards.append(
            "<section>"
            f"<h2>{title}</h2>"
            f"<p>{subtitle}</p>"
            f"{spectrogram_to_svg(spectrogram)}"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spectrogram Preview</title>
  <style>
    body {{
      background: #090b10;
      color: #edf3ff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
    }}
    main {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    h1 {{ font-size: 24px; margin: 0 0 18px; }}
    section {{
      background: #111722;
      border: 1px solid #263244;
      border-radius: 8px;
      padding: 12px;
    }}
    h2 {{ font-size: 13px; margin: 0 0 4px; overflow-wrap: anywhere; }}
    p {{ color: #9aa8bc; font-size: 12px; margin: 0 0 10px; }}
    svg {{ display: block; width: 100%; height: auto; background: #05070b; }}
  </style>
</head>
<body>
  <h1>Spectrogram Preview</h1>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def build_spectrogram_report(records: list[dict[str, Any]]) -> str:
    key_counts: dict[str, int] = {}
    frame_counts: dict[int, int] = {}
    for record in records:
        key = str(record.get("key") or "unknown")
        key_counts[key] = key_counts.get(key, 0) + 1
        frames = int(record["frames"])
        frame_counts[frames] = frame_counts.get(frames, 0) + 1

    lines = [
        "Log-Mel Spectrogram Generation Report",
        "=====================================",
        "",
        f"Total spectrograms: {len(records)}",
        f"Unique key labels: {len(key_counts)}",
        "",
        "Frame count distribution:",
    ]
    for frame_count in sorted(frame_counts):
        lines.append(f"- {frame_count} frames: {frame_counts[frame_count]}")

    lines.extend(["", "Key label counts:"])
    for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0])):
        label = "Space" if key == " " else key
        lines.append(f"- {label}: {count}")
    return "\n".join(lines)


def generate_session_spectrograms(
    clip_manifest_path: Path,
    output_root: Path | None = None,
    mel_bands: int = 64,
    fft_window_size: int = 1024,
    hop_length: int = 256,
    preview_count: int = 12,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    """Generate model-ready log-mel spectrogram arrays for every clip in a manifest."""
    clip_records = load_clip_manifest(clip_manifest_path)
    if not clip_records:
        raise ValueError(f"No clip records found in {clip_manifest_path}")

    session_id = clip_records[0]["session_id"]
    output_base = output_root or PROCESSED_DATA_DIR / "spectrograms"
    output_dir = output_base / session_id
    records: list[dict[str, Any]] = []
    previews: list[tuple[dict[str, Any], np.ndarray]] = []

    for clip_record in clip_records:
        spectrogram_record, spectrogram = generate_spectrogram_for_clip(
            record=clip_record,
            output_root=output_base,
            mel_bands=mel_bands,
            fft_window_size=fft_window_size,
            hop_length=hop_length,
        )
        records.append(spectrogram_record)
        if len(previews) < preview_count:
            previews.append((spectrogram_record, spectrogram))

    manifest_path = output_dir / "spectrogram_manifest.csv"
    report_path = output_dir / "spectrogram_report.txt"
    preview_path = output_dir / "spectrogram_preview.html"
    write_spectrogram_manifest(records, manifest_path)
    report_path.write_text(build_spectrogram_report(records), encoding="utf-8")
    preview_path.write_text(build_preview_html(previews), encoding="utf-8")
    return records, manifest_path, report_path, preview_path


def find_latest_clip_session(clips_root: Path | None = None) -> Path:
    root = clips_root or PROCESSED_DATA_DIR / "clips"
    sessions = sorted(path for path in root.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No clip sessions found under {root}")
    return sessions[-1]
