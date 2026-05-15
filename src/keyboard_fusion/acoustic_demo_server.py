from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from keyboard_fusion.acoustic_inference import (
    AcousticCNNPredictor,
    DEFAULT_MODEL_DIR,
    read_wav_bytes_mono_float,
    resample_linear,
)
from keyboard_fusion.config import load_config
from keyboard_fusion.neural_segmentation import DEFAULT_SEGMENTER_DIR, NeuralSegmenterPredictor, write_wav_mono_float
from keyboard_fusion.paths import METADATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
from keyboard_fusion.segmentation import extract_fixed_window
from keyboard_fusion.web_collection_server import find_available_port


WEB_DIR = PROJECT_ROOT / "web"
INDEX_PATH = WEB_DIR / "acoustic_demo.html"
DEFAULT_EXPECTED_KEY_COUNT = 5
MAX_EXPECTED_KEY_COUNT = 120
RUN_ID_PATTERN = re.compile(r"^run_[0-9]{8}_[0-9]{6}_[0-9]{6}$")
CLIP_COLUMNS = [
    "clip_id",
    "clip_audio_path",
    "clip_url",
    "event_index",
    "predicted_key",
    "time_seconds",
    "sample_index",
    "confidence",
    "peak_strength",
    "window_start_seconds",
    "window_end_seconds",
]
STATIC_FILES = {
    "/acoustic_demo.css": (WEB_DIR / "acoustic_demo.css", "text/css; charset=utf-8"),
    "/acoustic_demo.js": (WEB_DIR / "acoustic_demo.js", "text/javascript; charset=utf-8"),
}


class AcousticDemoRequestHandler(BaseHTTPRequestHandler):
    server: "AcousticDemoServer"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(f"[acoustic-demo] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path in {"/", "/index.html"}:
            self._send_file(INDEX_PATH, "text/html; charset=utf-8")
            return
        if request_path in STATIC_FILES:
            path, content_type = STATIC_FILES[request_path]
            self._send_file(path, content_type)
            return
        if request_path == "/api/config":
            self._send_json(self.server.config_payload())
            return
        run_file = resolve_run_file_url(request_path)
        if run_file is not None:
            self._send_file(run_file, "audio/wav")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        request_path = urlparse(self.path).path
        if request_path in {"/", "/index.html"}:
            self._send_file(INDEX_PATH, "text/html; charset=utf-8", head_only=True)
            return
        if request_path in STATIC_FILES:
            path, content_type = STATIC_FILES[request_path]
            self._send_file(path, content_type, head_only=True)
            return
        run_file = resolve_run_file_url(request_path)
        if run_file is not None:
            self._send_file(run_file, "audio/wav", head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self._read_json()
            result = self._predict(payload)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"ok": True, **result})

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        audio_base64 = str(payload.get("audio_base64") or "")
        if not audio_base64:
            raise ValueError("Missing audio_base64.")

        raw_wav = base64.b64decode(audio_base64)
        sample_rate, samples = read_wav_bytes_mono_float(raw_wav)
        max_audio_seconds = float(self.server.max_audio_seconds)
        audio_seconds = samples.size / float(sample_rate)
        if audio_seconds > max_audio_seconds:
            raise ValueError(f"Audio clip is too long. Keep it under {max_audio_seconds:g} seconds.")

        audio_config = self.server.project_config.get("audio", {})
        feature_config = self.server.project_config.get("features", {})
        segmentation_config = self.server.project_config.get("segmentation", {})
        pre_ms = float(segmentation_config.get("pre_keydown_ms", 20))
        post_ms = float(segmentation_config.get("post_keydown_ms", 45))
        event_limit = parse_event_limit(payload)
        predictor = self.server.predictor()
        peaks = None
        segmentation_method = "heuristic_detector"
        neural_segmenter = self.server.neural_segmenter()
        if neural_segmenter is not None:
            peaks = neural_segmenter.detect_peaks(
                samples=samples,
                sample_rate=sample_rate,
                max_peaks=event_limit,
                batch_size=128,
            )
            segmentation_method = "neural_segmenter"
        result = predictor.predict_samples(
            samples=samples,
            sample_rate=sample_rate,
            target_sample_rate=int(audio_config.get("sample_rate", 48000)),
            mel_bands=int(feature_config.get("mel_bands", 64)),
            fft_window_size=int(feature_config.get("fft_window_size", 1024)),
            hop_length=int(feature_config.get("hop_length", 256)),
            pre_ms=pre_ms,
            post_ms=post_ms,
            sensitivity=float(payload.get("sensitivity") or 0.5),
            min_gap_ms=float(payload.get("min_gap_ms") or 55.0),
            max_events=event_limit,
            top_k=5,
            peaks=peaks,
            segmentation_method=segmentation_method,
        )
        run_payload = save_inference_run(
            raw_wav=raw_wav,
            source_sample_rate=sample_rate,
            source_samples=samples,
            target_sample_rate=result.sample_rate,
            events=result.events,
            request_payload={
                "expected_key_count": event_limit,
                "sensitivity": payload.get("sensitivity"),
                "segmentation_method": result.segmentation_method,
            },
            model_dir=self.server.model_dir,
            segmenter_dir=self.server.segmenter_dir,
            pre_ms=pre_ms,
            post_ms=post_ms,
        )
        return {
            "predicted_text": result.predicted_text,
            "events": result.events,
            "detected_count": result.detected_count,
            "event_limit": event_limit,
            "audio_seconds": result.audio_seconds,
            "sample_rate": result.sample_rate,
            "model_dir": result.model_dir,
            "class_count": result.class_count,
            "segmentation_method": result.segmentation_method,
            "run": run_payload,
        }

    def _send_file(self, path: Path, content_type: str, head_only: bool = False) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Missing file: {path}")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AcousticDemoServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        model_dir: Path = DEFAULT_MODEL_DIR,
        segmenter_dir: Path | None = DEFAULT_SEGMENTER_DIR,
        max_audio_seconds: float = 15.0,
    ) -> None:
        super().__init__(server_address, AcousticDemoRequestHandler)
        self.project_config = load_config()
        self.model_dir = Path(model_dir)
        self.segmenter_dir = Path(segmenter_dir) if segmenter_dir is not None else None
        self.max_audio_seconds = max_audio_seconds
        self._predictor: AcousticCNNPredictor | None = None
        self._neural_segmenter: NeuralSegmenterPredictor | None = None

    def predictor(self) -> AcousticCNNPredictor:
        if self._predictor is None:
            self._predictor = AcousticCNNPredictor(self.model_dir)
        return self._predictor

    def neural_segmenter(self) -> NeuralSegmenterPredictor | None:
        if self.segmenter_dir is None:
            return None
        if not (self.segmenter_dir / "model.pt").exists():
            return None
        if self._neural_segmenter is None:
            self._neural_segmenter = NeuralSegmenterPredictor(self.segmenter_dir)
        return self._neural_segmenter

    def config_payload(self) -> dict[str, Any]:
        metrics_path = self.model_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "model_dir": str(self.model_dir),
            "model_exists": (self.model_dir / "model.pt").exists(),
            "segmenter_dir": str(self.segmenter_dir) if self.segmenter_dir is not None else "",
            "segmenter_exists": bool(self.segmenter_dir and (self.segmenter_dir / "model.pt").exists()),
            "model_metrics": {
                "session_id": metrics.get("session_id", self.model_dir.name),
                "top1_accuracy": metrics.get("top1_accuracy"),
                "top5_accuracy": metrics.get("top5_accuracy"),
                "class_count": metrics.get("class_count"),
            },
            "max_audio_seconds": self.max_audio_seconds,
            "config": self.project_config,
        }


def parse_event_limit(payload: dict[str, Any]) -> int:
    raw_value = payload.get("expected_key_count", payload.get("max_events", DEFAULT_EXPECTED_KEY_COUNT))
    try:
        event_limit = int(raw_value)
    except (TypeError, ValueError):
        event_limit = DEFAULT_EXPECTED_KEY_COUNT
    return max(1, min(MAX_EXPECTED_KEY_COUNT, event_limit))


def make_run_id(now: datetime | None = None) -> str:
    timestamp = now or datetime.now()
    return timestamp.strftime("run_%Y%m%d_%H%M%S_%f")


def safe_clip_label(value: str) -> str:
    label = "space" if value == "Space" else value.lower()
    label = re.sub(r"[^a-z0-9_-]+", "_", label).strip("_")
    return label or "unknown"


def run_directories(run_id: str) -> dict[str, Path]:
    return {
        "raw": RAW_DATA_DIR / "inference_runs" / run_id,
        "processed": PROCESSED_DATA_DIR / "inference_runs" / run_id,
        "metadata": METADATA_DIR / "inference_runs" / run_id,
    }


def ensure_unique_run_id() -> str:
    run_id = make_run_id()
    while any(path.exists() for path in run_directories(run_id).values()):
        run_id = make_run_id()
    return run_id


def resolve_run_file_url(request_path: str) -> Path | None:
    parts = [unquote(part) for part in request_path.split("/") if part]
    if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "raw.wav":
        run_id = parts[2]
        if not RUN_ID_PATTERN.match(run_id):
            return None
        return RAW_DATA_DIR / "inference_runs" / run_id / "recording.wav"
    if len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "clips":
        run_id = parts[2]
        filename = parts[4]
        if not RUN_ID_PATTERN.match(run_id) or "/" in filename or ".." in filename:
            return None
        return PROCESSED_DATA_DIR / "inference_runs" / run_id / "clips" / filename
    return None


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def save_inference_run(
    raw_wav: bytes,
    source_sample_rate: int,
    source_samples: Any,
    target_sample_rate: int,
    events: list[dict[str, Any]],
    request_payload: dict[str, Any],
    model_dir: Path,
    segmenter_dir: Path | None,
    pre_ms: float,
    post_ms: float,
) -> dict[str, Any]:
    run_id = ensure_unique_run_id()
    directories = run_directories(run_id)
    raw_dir = directories["raw"]
    processed_dir = directories["processed"]
    metadata_dir = directories["metadata"]
    clips_dir = processed_dir / "clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_audio_path = raw_dir / "recording.wav"
    raw_audio_path.write_bytes(raw_wav)

    target_samples = resample_linear(source_samples, source_sample_rate, target_sample_rate)
    clip_rows: list[dict[str, Any]] = []
    for event in events:
        event_index = int(event.get("index", len(clip_rows) + 1))
        predicted_key = str(event.get("predicted_key") or "unknown")
        clip_id = f"event_{event_index:03d}_{safe_clip_label(predicted_key)}"
        clip_path = clips_dir / f"{clip_id}.wav"
        center_sample = int(event.get("sample_index", 0))
        clip = extract_fixed_window(
            samples=target_samples,
            center_sample=center_sample,
            sample_rate=target_sample_rate,
            pre_ms=pre_ms,
            post_ms=post_ms,
        )
        write_wav_mono_float(clip_path, clip, target_sample_rate)
        time_seconds = float(event.get("time_seconds", 0.0))
        clip_url = f"/api/runs/{run_id}/clips/{clip_path.name}"
        event["clip_id"] = clip_id
        event["clip_audio_path"] = str(clip_path)
        event["clip_url"] = clip_url
        clip_rows.append(
            {
                "clip_id": clip_id,
                "clip_audio_path": str(clip_path),
                "clip_url": clip_url,
                "event_index": event_index,
                "predicted_key": predicted_key,
                "time_seconds": round(time_seconds, 6),
                "sample_index": center_sample,
                "confidence": event.get("confidence", 0.0),
                "peak_strength": event.get("strength", 0.0),
                "window_start_seconds": round(max(0.0, time_seconds - pre_ms / 1000.0), 9),
                "window_end_seconds": round(time_seconds + post_ms / 1000.0, 9),
            }
        )

    clip_manifest_path = processed_dir / "clip_manifest.csv"
    write_csv_rows(clip_manifest_path, clip_rows, CLIP_COLUMNS)
    metadata_path = metadata_dir / "metadata.json"
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_audio_path": str(raw_audio_path),
        "raw_audio_url": f"/api/runs/{run_id}/raw.wav",
        "clips_dir": str(clips_dir),
        "clip_manifest_path": str(clip_manifest_path),
        "model_dir": str(model_dir),
        "segmenter_dir": str(segmenter_dir) if segmenter_dir else "",
        "source_sample_rate": source_sample_rate,
        "target_sample_rate": target_sample_rate,
        "request": request_payload,
        "events": events,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "raw_audio_path": str(raw_audio_path),
        "raw_audio_url": f"/api/runs/{run_id}/raw.wav",
        "clips_dir": str(clips_dir),
        "clip_manifest_path": str(clip_manifest_path),
        "metadata_path": str(metadata_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the browser-based acoustic CNN demo app.")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--segmenter-dir", type=Path, default=DEFAULT_SEGMENTER_DIR)
    parser.add_argument(
        "--no-neural-segmenter",
        action="store_true",
        help="Use the older heuristic peak detector even if a neural segmenter exists.",
    )
    parser.add_argument("--max-audio-seconds", type=float, default=15.0)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args(argv)

    port = find_available_port(args.port)
    server = AcousticDemoServer(
        ("127.0.0.1", port),
        model_dir=args.model_dir,
        segmenter_dir=None if args.no_neural_segmenter else args.segmenter_dir,
        max_audio_seconds=args.max_audio_seconds,
    )
    url = f"http://127.0.0.1:{port}"
    print(f"Acoustic CNN demo running at {url}", flush=True)
    print(f"Model: {args.model_dir}", flush=True)
    if args.no_neural_segmenter:
        print("Neural segmenter: disabled", flush=True)
    else:
        print(f"Neural segmenter: {args.segmenter_dir}", flush=True)
    print("Press Ctrl+C in this terminal to stop it.", flush=True)

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping acoustic CNN demo.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
