from __future__ import annotations

import csv
import wave

from keyboard_fusion.preprocessing import clip_id_for_keydown, safe_label, write_clip_manifest, write_wav_clip


def write_test_wav(path, frame_count: int = 1000, sample_rate: int = 1000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            frames.extend(int(index % 100).to_bytes(2, byteorder="little", signed=True))
        wav_file.writeframes(bytes(frames))


def test_safe_label_handles_spaces_and_symbols() -> None:
    assert safe_label(" ") == "space"
    assert safe_label("KeyA") == "keya"
    assert safe_label(";") == "unknown"
    assert safe_label("ArrowLeft") == "arrowleft"


def test_clip_id_for_keydown_is_stable() -> None:
    keydown = {"event_index": 7, "code": "Space", "key": " "}

    assert clip_id_for_keydown("trial_003", keydown) == "trial_003_event_007_space_space"


def test_write_wav_clip_cuts_expected_frame_count(tmp_path) -> None:
    source_path = tmp_path / "source.wav"
    clip_path = tmp_path / "clip.wav"
    write_test_wav(source_path)

    frames_written = write_wav_clip(source_path, clip_path, start_sample=100, end_sample=350)

    assert frames_written == 250
    with wave.open(str(clip_path), "rb") as clip_file:
        assert clip_file.getframerate() == 1000
        assert clip_file.getnchannels() == 1
        assert clip_file.getnframes() == 250


def test_write_wav_clip_silences_outside_keep_region(tmp_path) -> None:
    source_path = tmp_path / "source.wav"
    clip_path = tmp_path / "clip.wav"
    write_test_wav(source_path, frame_count=100)

    frames_written = write_wav_clip(
        source_path,
        clip_path,
        start_sample=10,
        end_sample=30,
        keep_start_sample=15,
        keep_end_sample=25,
    )

    assert frames_written == 20
    with wave.open(str(clip_path), "rb") as clip_file:
        raw_frames = clip_file.readframes(clip_file.getnframes())

    samples = [
        int.from_bytes(raw_frames[index : index + 2], byteorder="little", signed=True)
        for index in range(0, len(raw_frames), 2)
    ]
    assert samples[:5] == [0, 0, 0, 0, 0]
    assert samples[5:15] == list(range(15, 25))
    assert samples[15:] == [0, 0, 0, 0, 0]


def test_write_wav_clip_clips_to_audio_bounds(tmp_path) -> None:
    source_path = tmp_path / "source.wav"
    clip_path = tmp_path / "clip.wav"
    write_test_wav(source_path, frame_count=100)

    frames_written = write_wav_clip(source_path, clip_path, start_sample=80, end_sample=130)

    assert frames_written == 20
    with wave.open(str(clip_path), "rb") as clip_file:
        assert clip_file.getnframes() == 20


def test_write_clip_manifest_uses_stable_columns(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_clip_manifest(
        [
            {
                "clip_id": "trial_001_event_000_keya_a",
                "session_id": "session_001",
                "trial_id": "trial_001",
                "event_index": 0,
                "key": "a",
                "char": "a",
                "code": "KeyA",
            }
        ],
        manifest_path,
    )

    with manifest_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["clip_id"] == "trial_001_event_000_keya_a"
    assert rows[0]["key"] == "a"
    assert "window_start_seconds" in rows[0]
