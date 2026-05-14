from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from keyboard_fusion.collection import (
    build_trial_paths,
    load_prompt_files,
    make_session_id,
    next_trial_id,
    sanitize_id,
    write_events_csv,
    write_metadata_json,
)
from keyboard_fusion.config import load_config
from keyboard_fusion.paths import PROJECT_ROOT, RAW_DATA_DIR


WEB_DIR = PROJECT_ROOT / "web"
INDEX_PATH = WEB_DIR / "collector.html"
STATIC_FILES = {
    "/collector.css": (WEB_DIR / "collector.css", "text/css; charset=utf-8"),
    "/collector.js": (WEB_DIR / "collector.js", "text/javascript; charset=utf-8"),
}


def find_available_port(preferred_port: int) -> int:
    """Return preferred_port if possible, otherwise ask the OS for a free port."""
    if preferred_port <= 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred_port))
            return preferred_port
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class CollectorRequestHandler(BaseHTTPRequestHandler):
    server: "CollectorServer"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(f"[collector] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_file(INDEX_PATH, "text/html; charset=utf-8")
            return
        if self.path in STATIC_FILES:
            path, content_type = STATIC_FILES[self.path]
            self._send_file(path, content_type)
            return
        if self.path == "/api/config":
            self._send_json(
                {
                    "config": self.server.project_config,
                    "prompt_sets": self.server.prompt_sets,
                    "default_session_id": make_session_id(),
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_headers_for_file(INDEX_PATH, "text/html; charset=utf-8")
            return
        if self.path in STATIC_FILES:
            path, content_type = STATIC_FILES[self.path]
            self._send_headers_for_file(path, content_type)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/save-trial":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            payload = self._read_json()
            saved = self._save_trial(payload)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"ok": True, **saved})

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _save_trial(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = sanitize_id(str(payload.get("session_id") or make_session_id()))
        participant_id = sanitize_id(str(payload.get("participant_id") or "p001"))
        session_dir = RAW_DATA_DIR / "sessions" / session_id
        trial_id = next_trial_id(session_dir)
        paths = build_trial_paths(session_id, trial_id)
        paths.session_dir.mkdir(parents=True, exist_ok=True)

        audio_base64 = str(payload.get("audio_base64") or "")
        if not audio_base64:
            raise ValueError("Missing audio_base64.")
        paths.audio_path.write_bytes(base64.b64decode(audio_base64))

        events = payload.get("events") or []
        if not isinstance(events, list):
            raise ValueError("events must be a list.")
        write_events_csv(paths.events_path, events)

        metadata = {
            "trial_id": trial_id,
            "session_id": session_id,
            "participant_id": participant_id,
            "prompt_set": payload.get("prompt_set"),
            "prompt_index": payload.get("prompt_index"),
            "prompt_text": payload.get("prompt_text"),
            "typed_text": payload.get("typed_text"),
            "started_at": payload.get("started_at"),
            "ended_at": payload.get("ended_at"),
            "duration_seconds": payload.get("duration_seconds"),
            "audio_file_path": paths.audio_path.name,
            "events_file_path": paths.events_path.name,
            "sample_rate": payload.get("sample_rate"),
            "channels": payload.get("channels", 1),
            "audio_input_device": payload.get("audio_input_device") or {},
            "event_count": len(events),
            "audio_frame_count": payload.get("audio_frame_count"),
            "keyboard": self.server.project_config.get("hardware", {}).get("keyboard", {}),
            "microphone": self.server.project_config.get("hardware", {}).get("microphone", {}),
            "environment": self.server.project_config.get("environment", {}),
            "collection_client": "browser",
            "notes": (
                "Event timestamps are browser performance times. "
                "trial_elapsed_seconds is relative to the Start Trial click."
            ),
        }
        write_metadata_json(paths.metadata_path, metadata)

        return {
            "trial_id": trial_id,
            "session_id": session_id,
            "session_dir": str(paths.session_dir),
            "audio_path": str(paths.audio_path),
            "events_path": str(paths.events_path),
            "metadata_path": str(paths.metadata_path),
        }

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Missing file: {path}")
            return
        body = path.read_bytes()
        self._send_headers(content_type, len(body))
        self.wfile.write(body)

    def _send_headers_for_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"Missing file: {path}")
            return
        self._send_headers(content_type, path.stat().st_size)

    def _send_headers(self, content_type: str, content_length: int) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CollectorServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, CollectorRequestHandler)
        self.project_config = load_config()
        self.prompt_sets = load_prompt_files()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the browser-based data collection app.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args(argv)

    port = find_available_port(args.port)
    server = CollectorServer(("127.0.0.1", port))
    url = f"http://127.0.0.1:{port}"
    print(f"Collector running at {url}", flush=True)
    print("Press Ctrl+C in this terminal to stop it.", flush=True)

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping collector.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
