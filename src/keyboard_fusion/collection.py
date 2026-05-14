from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from keyboard_fusion.paths import PROJECT_ROOT, RAW_DATA_DIR


TRIAL_ID_PATTERN = re.compile(r"trial_(\d{3})_metadata\.json$")


@dataclass(frozen=True)
class TrialPaths:
    session_dir: Path
    audio_path: Path
    events_path: Path
    metadata_path: Path


def sanitize_id(value: str) -> str:
    """Convert a user-entered ID into a simple filesystem-safe ID."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def make_session_id(now: datetime | None = None) -> str:
    """Create a stable session ID from the current local time."""
    now = now or datetime.now()
    return now.strftime("session_%Y%m%d_%H%M%S")


def next_trial_id(session_dir: Path) -> str:
    """Return the next trial ID for a session directory."""
    max_seen = 0
    if session_dir.exists():
        for path in session_dir.glob("trial_*_metadata.json"):
            match = TRIAL_ID_PATTERN.match(path.name)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
    return f"trial_{max_seen + 1:03d}"


def build_trial_paths(
    session_id: str,
    trial_id: str,
    raw_root: Path | None = None,
) -> TrialPaths:
    """Build the three raw files that belong to one trial."""
    base = raw_root or RAW_DATA_DIR / "sessions"
    session_dir = base / sanitize_id(session_id)
    return TrialPaths(
        session_dir=session_dir,
        audio_path=session_dir / f"{trial_id}.wav",
        events_path=session_dir / f"{trial_id}_events.csv",
        metadata_path=session_dir / f"{trial_id}_metadata.json",
    )


def load_prompt_files(prompt_dir: Path | None = None) -> dict[str, list[str]]:
    """Load prompt lists from text files.

    Each non-empty, non-comment line is one prompt. The dictionary key is the
    filename stem, such as "english_phrases".
    """
    directory = prompt_dir or PROJECT_ROOT / "prompts"
    prompt_sets: dict[str, list[str]] = {}
    if not directory.exists():
        return prompt_sets

    for path in sorted(directory.glob("*.txt")):
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if prompts:
            prompt_sets[path.stem] = prompts
    return prompt_sets


def write_events_csv(path: Path, events: Iterable[dict[str, Any]]) -> None:
    """Write key events to CSV in a stable column order."""
    rows = list(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "event_index",
        "event_type",
        "key",
        "char",
        "keysym",
        "code",
        "keycode",
        "location",
        "repeat",
        "timestamp_monotonic",
        "browser_time_ms",
        "trial_elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    """Write trial metadata as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
