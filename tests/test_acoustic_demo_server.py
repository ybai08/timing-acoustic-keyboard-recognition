from __future__ import annotations

from datetime import datetime

from keyboard_fusion.acoustic_demo_server import make_run_id, parse_event_limit, resolve_run_file_url, safe_clip_label


def test_parse_event_limit_prefers_expected_key_count() -> None:
    assert parse_event_limit({"expected_key_count": "5", "max_events": "80"}) == 5


def test_parse_event_limit_clamps_values() -> None:
    assert parse_event_limit({"expected_key_count": "0"}) == 1
    assert parse_event_limit({"expected_key_count": "999"}) == 120


def test_parse_event_limit_uses_safe_default() -> None:
    assert parse_event_limit({"expected_key_count": "not a number"}) == 5


def test_make_run_id_is_stable_and_safe() -> None:
    assert make_run_id(datetime(2026, 5, 15, 12, 34, 56, 123456)) == "run_20260515_123456_123456"


def test_safe_clip_label_handles_space_and_symbols() -> None:
    assert safe_clip_label("Space") == "space"
    assert safe_clip_label("Key/A?") == "key_a"
    assert safe_clip_label("") == "unknown"


def test_resolve_run_file_url_limits_to_run_files() -> None:
    raw = resolve_run_file_url("/api/runs/run_20260515_123456_123456/raw.wav")
    clip = resolve_run_file_url("/api/runs/run_20260515_123456_123456/clips/event_001_a.wav")
    bad = resolve_run_file_url("/api/runs/not_safe/clips/event_001_a.wav")

    assert raw is not None
    assert raw.name == "recording.wav"
    assert clip is not None
    assert clip.name == "event_001_a.wav"
    assert bad is None
