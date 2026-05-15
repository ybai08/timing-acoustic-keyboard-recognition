from __future__ import annotations

from datetime import datetime

import pytest

from keyboard_fusion.collection import (
    build_trial_paths,
    delete_trial_files,
    make_session_id,
    next_trial_id,
    sanitize_id,
    validate_trial_id,
)


def test_sanitize_id() -> None:
    assert sanitize_id(" session 001 ") == "session_001"
    assert sanitize_id("p:001") == "p_001"
    assert sanitize_id("   ") == "unknown"


def test_make_session_id() -> None:
    session_id = make_session_id(datetime(2026, 5, 14, 16, 30, 5))
    assert session_id == "session_20260514_163005"


def test_next_trial_id(tmp_path) -> None:
    assert next_trial_id(tmp_path) == "trial_001"
    (tmp_path / "trial_001_metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "trial_003_metadata.json").write_text("{}", encoding="utf-8")
    assert next_trial_id(tmp_path) == "trial_004"


def test_build_trial_paths(tmp_path) -> None:
    paths = build_trial_paths("session 001", "trial_001", raw_root=tmp_path)
    assert paths.session_dir == tmp_path / "session_001"
    assert paths.audio_path.name == "trial_001.wav"
    assert paths.events_path.name == "trial_001_events.csv"
    assert paths.metadata_path.name == "trial_001_metadata.json"


def test_validate_trial_id() -> None:
    assert validate_trial_id("trial_001") == "trial_001"
    with pytest.raises(ValueError):
        validate_trial_id("../trial_001")
    with pytest.raises(ValueError):
        validate_trial_id("trial_1")


def test_delete_trial_files(tmp_path) -> None:
    paths = build_trial_paths("session 001", "trial_001", raw_root=tmp_path)
    paths.session_dir.mkdir(parents=True)
    paths.audio_path.write_bytes(b"wav")
    paths.events_path.write_text("events", encoding="utf-8")
    paths.metadata_path.write_text("{}", encoding="utf-8")

    result = delete_trial_files("session 001", "trial_001", raw_root=tmp_path)

    assert result["session_id"] == "session_001"
    assert result["trial_id"] == "trial_001"
    assert len(result["deleted_paths"]) == 3
    assert not paths.audio_path.exists()
    assert not paths.events_path.exists()
    assert not paths.metadata_path.exists()


def test_delete_trial_files_requires_existing_trial(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        delete_trial_files("session_001", "trial_001", raw_root=tmp_path)
