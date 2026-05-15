from __future__ import annotations

import csv
import json
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
    "keydown_time_seconds",
    "audio_time_seconds",
    "audio_start_offset_seconds",
    "window_start_seconds",
    "window_end_seconds",
    "keydown_position_in_clip",
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


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def keydown_position_in_clip(record: dict[str, Any]) -> float:
    """Return the keydown position inside the clip as a 0..1 fraction."""
    keydown_time = parse_float(record.get("audio_time_seconds"))
    if keydown_time == 0.0 and not record.get("audio_time_seconds"):
        sample_index = parse_float(record.get("sample_index"), -1.0)
        window_start_sample = parse_float(record.get("window_start_sample"), -1.0)
        window_end_sample = parse_float(record.get("window_end_sample"), -1.0)
        sample_duration = window_end_sample - window_start_sample
        if sample_index >= 0 and sample_duration > 0:
            return max(0.0, min(1.0, (sample_index - window_start_sample) / sample_duration))
        keydown_time = parse_float(record.get("keydown_time_seconds"))
    window_start = parse_float(record.get("window_start_seconds"))
    window_end = parse_float(record.get("window_end_seconds"))
    duration = window_end - window_start
    if duration <= 0:
        return 0.0
    return max(0.0, min(1.0, (keydown_time - window_start) / duration))


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
        "keydown_position_in_clip": round(keydown_position_in_clip(record), 9),
    }
    return spectrogram_record, normalized


def marker_line_svg(marker_fraction: float, width: int, height: int) -> str:
    x = max(0.0, min(1.0, marker_fraction)) * width
    return (
        f'<line x1="{x:.2f}" y1="0" x2="{x:.2f}" y2="{height}" '
        'stroke="#ffcc66" stroke-width="2" vector-effect="non-scaling-stroke" />'
    )


def waveform_to_svg(
    samples: np.ndarray,
    marker_fraction: float,
    width: int = 260,
    height: int = 72,
) -> str:
    """Render a compact waveform with a vertical keydown marker."""
    values = samples.astype(np.float32)
    if values.size == 0:
        values = np.zeros(1, dtype=np.float32)

    peak = float(np.max(np.abs(values))) or 1.0
    center_y = height / 2
    usable_height = height * 0.42
    columns = min(width, max(1, int(values.size)))
    lines: list[str] = [
        f'<line x1="0" y1="{center_y:.2f}" x2="{width}" y2="{center_y:.2f}" '
        'stroke="rgba(154,168,188,0.35)" stroke-width="1" />'
    ]

    for column in range(columns):
        start = int(column * values.size / columns)
        end = int((column + 1) * values.size / columns)
        chunk = values[start:max(end, start + 1)]
        minimum = float(np.min(chunk))
        maximum = float(np.max(chunk))
        x = column * width / columns
        y1 = center_y - (maximum / peak * usable_height)
        y2 = center_y - (minimum / peak * usable_height)
        lines.append(
            f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" '
            'stroke="#49a5ff" stroke-width="1" />'
        )

    lines.append(marker_line_svg(marker_fraction, width, height))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="waveform with keydown marker">{"".join(lines)}</svg>'
    )


def spectrogram_to_svg(
    spectrogram: np.ndarray,
    marker_fraction: float,
    width: int = 260,
    height: int = 130,
) -> str:
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

    rects.append(marker_line_svg(marker_fraction, width, height))
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="log mel spectrogram">{"".join(rects)}</svg>'
    )


def key_label_for_record(record: dict[str, Any]) -> str:
    key = str(record.get("key") or "")
    if key == " ":
        return "Space"
    return key or str(record.get("code") or "Unknown")


def event_index_for_record(record: dict[str, Any]) -> int:
    try:
        return int(record.get("event_index", 0))
    except (TypeError, ValueError):
        return 0


def spectrogram_preview_grid(spectrogram: np.ndarray) -> list[list[float]]:
    """Return a small normalized 0..1 grid that browser canvas can draw."""
    values = spectrogram.astype(np.float32)
    if values.size == 0:
        return [[0.0]]

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        scaled = np.zeros_like(values, dtype=np.float32)
    else:
        scaled = ((values - minimum) / (maximum - minimum)).astype(np.float32)
    return [[round(float(value), 4) for value in row] for row in scaled]


def waveform_preview_peaks(samples: np.ndarray, columns: int = 180) -> list[list[float]]:
    """Downsample a waveform into normalized min/max columns for preview drawing."""
    values = samples.astype(np.float32)
    if values.size == 0:
        return [[0.0, 0.0]]

    column_count = min(max(columns, 1), int(values.size))
    peak = float(np.max(np.abs(values))) or 1.0
    peaks: list[list[float]] = []
    for column in range(column_count):
        start = int(column * values.size / column_count)
        end = int((column + 1) * values.size / column_count)
        chunk = values[start:max(end, start + 1)]
        peaks.append(
            [
                round(float(np.min(chunk)) / peak, 4),
                round(float(np.max(chunk)) / peak, 4),
            ]
        )
    return peaks


def build_preview_payload(previews: list[tuple[dict[str, Any], np.ndarray, np.ndarray]]) -> dict[str, Any]:
    trials: dict[str, dict[str, Any]] = {}
    sorted_previews = sorted(previews, key=lambda item: (str(item[0].get("trial_id", "")), event_index_for_record(item[0])))

    for record, spectrogram, samples in sorted_previews:
        trial_id = str(record.get("trial_id") or "unknown_trial")
        trial = trials.setdefault(
            trial_id,
            {
                "trialId": trial_id,
                "promptSet": str(record.get("prompt_set") or ""),
                "promptIndex": str(record.get("prompt_index") or ""),
                "promptText": str(record.get("prompt_text") or ""),
                "items": [],
            },
        )

        marker_fraction = parse_float(record.get("keydown_position_in_clip"))
        window_duration_seconds = parse_float(record.get("window_duration_seconds"), 0.065)
        item = {
            "clipId": str(record.get("clip_id") or ""),
            "eventIndex": event_index_for_record(record),
            "key": key_label_for_record(record),
            "code": str(record.get("code") or ""),
            "char": str(record.get("char") or ""),
            "keydownMs": round(marker_fraction * window_duration_seconds * 1000, 2),
            "marker": round(marker_fraction, 5),
            "frames": int(spectrogram.shape[1]) if spectrogram.ndim == 2 else 0,
            "shape": [int(size) for size in spectrogram.shape],
            "waveform": waveform_preview_peaks(samples),
            "spectrogram": spectrogram_preview_grid(spectrogram),
        }
        trial["items"].append(item)

    trial_list = list(trials.values())
    for trial in trial_list:
        trial["keyCount"] = len(trial["items"])
        trial["keySequence"] = " ".join(str(item["key"]) for item in trial["items"])

    return {
        "trialCount": len(trial_list),
        "clipCount": sum(len(trial["items"]) for trial in trial_list),
        "trials": trial_list,
    }


def build_preview_html(previews: list[tuple[dict[str, Any], np.ndarray, np.ndarray]]) -> str:
    payload = build_preview_payload(previews)
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_json = (
        payload_json.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spectrogram Preview</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #090b10;
      --panel: #111722;
      --panel-strong: #151d2b;
      --line: #263244;
      --line-soft: rgba(154, 168, 188, 0.18);
      --text: #edf3ff;
      --muted: #9aa8bc;
      --accent: #49a5ff;
      --marker: #ffcc66;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      min-width: 320px;
    }}
    .shell {{
      display: grid;
      gap: 16px;
      padding: 24px;
    }}
    header {{
      align-items: end;
      display: flex;
      gap: 16px;
      justify-content: space-between;
    }}
    h1 {{
      font-size: clamp(24px, 3vw, 34px);
      line-height: 1.05;
      margin: 0;
    }}
    .summary {{
      color: var(--muted);
      font-size: 14px;
      margin: 8px 0 0;
    }}
    label {{
      color: var(--muted);
      display: grid;
      font-size: 12px;
      font-weight: 700;
      gap: 7px;
      min-width: min(340px, 100%);
      text-transform: uppercase;
    }}
    select {{
      appearance: none;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--text);
      font: inherit;
      font-size: 14px;
      min-height: 42px;
      padding: 12px;
    }}
    .trial-panel {{
      background: linear-gradient(180deg, rgba(73, 165, 255, 0.08), rgba(73, 165, 255, 0.02));
      border: 1px solid var(--line);
      border-radius: 8px;
      display: grid;
      gap: 14px;
      grid-template-columns: minmax(0, 1fr) auto;
      padding: 14px;
    }}
    .eyebrow {{
      color: var(--muted);
      display: block;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0;
      margin-bottom: 5px;
      text-transform: uppercase;
    }}
    .prompt-text {{
      font-size: 15px;
      line-height: 1.4;
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .stats {{
      align-items: center;
      display: flex;
      gap: 10px;
    }}
    .stat {{
      background: rgba(9, 11, 16, 0.46);
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      min-width: 92px;
      padding: 9px 10px;
    }}
    .stat strong {{
      display: block;
      font-size: 20px;
      line-height: 1;
    }}
    .stat span {{
      color: var(--muted);
      display: block;
      font-size: 11px;
      margin-top: 4px;
    }}
    .key-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .key-chip {{
      background: #151d2b;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      font-size: 12px;
      line-height: 1;
      min-width: 28px;
      padding: 7px 8px;
      text-align: center;
    }}
    .cards {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .card h2 {{
      font-size: 13px;
      line-height: 1.25;
      margin: 0 0 4px;
      overflow-wrap: anywhere;
    }}
    .card p {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      margin: 0 0 10px;
    }}
    .label-row {{
      color: var(--muted);
      display: flex;
      font-size: 11px;
      gap: 10px;
      justify-content: space-between;
      margin: 10px 0 5px;
    }}
    canvas {{
      background: #05070b;
      border: 1px solid var(--line-soft);
      display: block;
      width: 100%;
    }}
    .waveform {{ height: 76px; }}
    .spectrogram {{ height: 142px; }}
    .empty {{
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 20px;
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 16px; }}
      header {{ align-items: stretch; flex-direction: column; }}
      .trial-panel {{ grid-template-columns: 1fr; }}
      .stats {{ align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>Spectrogram Preview</h1>
        <p class="summary" id="summary">Loading preview data...</p>
      </div>
      <label for="trialSelect">
        Trial
        <select id="trialSelect"></select>
      </label>
    </header>

    <section class="trial-panel" aria-label="selected trial summary">
      <div>
        <span class="eyebrow">Prompt</span>
        <p class="prompt-text" id="promptText"></p>
      </div>
      <div class="stats">
        <div class="stat"><strong id="keyCount">0</strong><span>keys</span></div>
        <div class="stat"><strong id="trialNumber">-</strong><span>trial</span></div>
      </div>
    </section>

    <div class="key-strip" id="keyStrip" aria-label="keys pressed in the selected trial"></div>
    <main class="cards" id="cards"></main>
  </div>

  <script id="preview-data" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById("preview-data").textContent);
    const summary = document.getElementById("summary");
    const select = document.getElementById("trialSelect");
    const promptText = document.getElementById("promptText");
    const keyCount = document.getElementById("keyCount");
    const trialNumber = document.getElementById("trialNumber");
    const keyStrip = document.getElementById("keyStrip");
    const cards = document.getElementById("cards");
    let selectedIndex = 0;
    let resizeTimer = null;

    function drawMarker(ctx, marker, width, height) {{
      const x = Math.max(0, Math.min(1, marker || 0)) * width;
      ctx.strokeStyle = "#ffcc66";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }}

    function setupCanvas(canvas, height) {{
      const width = Math.max(260, Math.floor(canvas.clientWidth || 260));
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      return {{ ctx, width, height }};
    }}

    function drawWaveform(canvas, waveform, marker) {{
      const {{ ctx, width, height }} = setupCanvas(canvas, 76);
      ctx.fillStyle = "#05070b";
      ctx.fillRect(0, 0, width, height);
      const center = height / 2;
      const usable = height * 0.42;
      ctx.strokeStyle = "rgba(154, 168, 188, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, center);
      ctx.lineTo(width, center);
      ctx.stroke();
      ctx.strokeStyle = "#49a5ff";
      ctx.beginPath();
      waveform.forEach((pair, index) => {{
        const x = (index + 0.5) * width / waveform.length;
        const min = Number(pair[0]) || 0;
        const max = Number(pair[1]) || 0;
        ctx.moveTo(x, center - max * usable);
        ctx.lineTo(x, center - min * usable);
      }});
      ctx.stroke();
      drawMarker(ctx, marker, width, height);
    }}

    function heatColor(value) {{
      const v = Math.max(0, Math.min(1, Number(value) || 0));
      const lightness = 12 + v * 72;
      return `hsl(185 85% ${{lightness}}%)`;
    }}

    function drawSpectrogram(canvas, grid, marker) {{
      const {{ ctx, width, height }} = setupCanvas(canvas, 142);
      ctx.fillStyle = "#05070b";
      ctx.fillRect(0, 0, width, height);
      const rows = grid.length;
      const cols = rows ? grid[0].length : 0;
      if (!rows || !cols) {{
        drawMarker(ctx, marker, width, height);
        return;
      }}
      const cellWidth = width / cols;
      const cellHeight = height / rows;
      for (let row = 0; row < rows; row += 1) {{
        for (let col = 0; col < cols; col += 1) {{
          ctx.fillStyle = heatColor(grid[row][col]);
          ctx.fillRect(col * cellWidth, height - (row + 1) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
        }}
      }}
      drawMarker(ctx, marker, width, height);
    }}

    function makeText(tag, className, text) {{
      const element = document.createElement(tag);
      if (className) element.className = className;
      element.textContent = text;
      return element;
    }}

    function makeCard(item) {{
      const card = document.createElement("section");
      card.className = "card";
      card.appendChild(makeText("h2", "", `${{item.clipId}} | key=${{item.key}}`));
      card.appendChild(makeText("p", "", `event ${{item.eventIndex}} | keydown at ${{item.keydownMs.toFixed(1)}} ms | shape=(${{item.shape.join(", ")}})`));

      const waveLabel = document.createElement("div");
      waveLabel.className = "label-row";
      waveLabel.appendChild(makeText("span", "", "Waveform"));
      waveLabel.appendChild(makeText("span", "", "yellow line = logged keydown"));
      card.appendChild(waveLabel);

      const waveform = document.createElement("canvas");
      waveform.className = "waveform";
      waveform.dataset.kind = "waveform";
      card.appendChild(waveform);

      const specLabel = document.createElement("div");
      specLabel.className = "label-row";
      specLabel.appendChild(makeText("span", "", "Log-mel spectrogram"));
      specLabel.appendChild(makeText("span", "", "same keydown marker"));
      card.appendChild(specLabel);

      const spectrogram = document.createElement("canvas");
      spectrogram.className = "spectrogram";
      spectrogram.dataset.kind = "spectrogram";
      card.appendChild(spectrogram);
      return {{ card, waveform, spectrogram }};
    }}

    function renderTrial(index) {{
      selectedIndex = index;
      const trial = data.trials[index];
      cards.textContent = "";
      keyStrip.textContent = "";
      if (!trial) {{
        cards.appendChild(makeText("p", "empty", "No preview data is available."));
        return;
      }}

      promptText.textContent = trial.promptText || "No prompt text saved.";
      keyCount.textContent = String(trial.keyCount || trial.items.length);
      trialNumber.textContent = trial.trialId.replace("trial_", "");

      trial.items.forEach((item) => {{
        const chip = makeText("span", "key-chip", item.key);
        keyStrip.appendChild(chip);
      }});

      const canvases = [];
      trial.items.forEach((item) => {{
        const preview = makeCard(item);
        cards.appendChild(preview.card);
        canvases.push([item, preview.waveform, preview.spectrogram]);
      }});
      requestAnimationFrame(() => {{
        canvases.forEach(([item, waveform, spectrogram]) => {{
          drawWaveform(waveform, item.waveform, item.marker);
          drawSpectrogram(spectrogram, item.spectrogram, item.marker);
        }});
      }});
    }}

    function initialize() {{
      summary.textContent = `${{data.clipCount}} clips across ${{data.trialCount}} trials`;
      data.trials.forEach((trial, index) => {{
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${{trial.trialId}} - ${{trial.keyCount}} keys`;
        select.appendChild(option);
      }});
      select.addEventListener("change", () => renderTrial(Number(select.value)));
      renderTrial(0);
    }}

    window.addEventListener("resize", () => {{
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => renderTrial(selectedIndex), 120);
    }});

    initialize();
  </script>
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
    preview_count: int = 0,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    """Generate model-ready log-mel spectrogram arrays for every clip in a manifest."""
    clip_records = load_clip_manifest(clip_manifest_path)
    if not clip_records:
        raise ValueError(f"No clip records found in {clip_manifest_path}")

    session_id = clip_records[0]["session_id"]
    output_base = output_root or PROCESSED_DATA_DIR / "spectrograms"
    output_dir = output_base / session_id
    records: list[dict[str, Any]] = []
    previews: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []

    for clip_record in clip_records:
        spectrogram_record, spectrogram = generate_spectrogram_for_clip(
            record=clip_record,
            output_root=output_base,
            mel_bands=mel_bands,
            fft_window_size=fft_window_size,
            hop_length=hop_length,
        )
        records.append(spectrogram_record)
        if preview_count <= 0 or len(previews) < preview_count:
            _, samples = read_wav_mono_float(Path(spectrogram_record["clip_audio_path"]))
            previews.append((spectrogram_record, spectrogram, samples))

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
