from __future__ import annotations

import csv
import wave

import numpy as np

from keyboard_fusion.spectrograms import (
    build_preview_payload,
    generate_session_spectrograms,
    keydown_position_in_clip,
    log_mel_spectrogram,
    mel_filterbank,
    normalize_spectrogram,
    read_wav_mono_float,
)


def write_test_wav(path, samples: np.ndarray, sample_rate: int = 8000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def test_mel_filterbank_has_expected_shape() -> None:
    filters = mel_filterbank(sample_rate=8000, fft_window_size=512, mel_bands=16)

    assert filters.shape == (16, 257)
    assert np.all(filters >= 0)
    assert np.any(filters > 0)


def test_log_mel_spectrogram_shape_and_normalization() -> None:
    sample_rate = 8000
    time = np.arange(0, 0.2, 1 / sample_rate)
    samples = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    spectrogram = log_mel_spectrogram(
        samples=samples,
        sample_rate=sample_rate,
        mel_bands=24,
        fft_window_size=256,
        hop_length=128,
    )
    normalized, mean, std = normalize_spectrogram(spectrogram)

    assert spectrogram.shape[0] == 24
    assert spectrogram.shape[1] > 1
    assert abs(float(np.mean(normalized))) < 1e-5
    assert 0.9 < float(np.std(normalized)) < 1.1
    assert mean >= 0
    assert std > 0


def test_read_wav_mono_float(tmp_path) -> None:
    wav_path = tmp_path / "clip.wav"
    samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    write_test_wav(wav_path, samples)

    sample_rate, loaded = read_wav_mono_float(wav_path)

    assert sample_rate == 8000
    assert loaded.shape == (3,)
    assert np.allclose(loaded, samples, atol=1e-4)


def test_keydown_position_uses_audio_time_not_browser_time() -> None:
    position = keydown_position_in_clip(
        {
            "keydown_time_seconds": "3.3364",
            "audio_time_seconds": "2.7664",
            "window_start_seconds": "2.7464",
            "window_end_seconds": "2.8114",
        }
    )

    assert 0.30 < position < 0.31


def test_preview_payload_groups_all_keys_by_trial() -> None:
    spectrogram = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    samples = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
    previews = [
        (
            {
                "clip_id": "trial_002_event_001_keyb_b",
                "trial_id": "trial_002",
                "event_index": "1",
                "key": "b",
                "code": "KeyB",
                "keydown_position_in_clip": "0.5",
                "window_duration_seconds": "0.1",
                "prompt_text": "ab",
            },
            spectrogram,
            samples,
        ),
        (
            {
                "clip_id": "trial_001_event_000_keya_a",
                "trial_id": "trial_001",
                "event_index": "0",
                "key": "a",
                "code": "KeyA",
                "keydown_position_in_clip": "0.25",
                "window_duration_seconds": "0.1",
                "prompt_text": "a",
            },
            spectrogram,
            samples,
        ),
    ]

    payload = build_preview_payload(previews)

    assert payload["trialCount"] == 2
    assert payload["clipCount"] == 2
    assert [trial["trialId"] for trial in payload["trials"]] == ["trial_001", "trial_002"]
    assert payload["trials"][0]["items"][0]["key"] == "a"
    assert payload["trials"][1]["items"][0]["keydownMs"] == 50.0


def test_generate_session_spectrograms_writes_outputs(tmp_path) -> None:
    session_dir = tmp_path / "clips" / "session_001" / "trial_001"
    session_dir.mkdir(parents=True)
    wav_path = session_dir / "trial_001_event_000_keya_a.wav"
    samples = np.zeros(800, dtype=np.float32)
    samples[200:230] = 0.8
    write_test_wav(wav_path, samples)

    manifest_path = tmp_path / "clips" / "session_001" / "clip_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "clip_id",
                "clip_audio_path",
                "session_id",
                "trial_id",
                "event_index",
                "key",
                "char",
                "code",
                "keydown_time_seconds",
                "audio_time_seconds",
                "window_start_seconds",
                "window_end_seconds",
                "window_duration_seconds",
                "prompt_set",
                "prompt_index",
                "prompt_text",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "trial_001_event_000_keya_a",
                "clip_audio_path": str(wav_path),
                "session_id": "session_001",
                "trial_id": "trial_001",
                "event_index": 0,
                "key": "a",
                "char": "a",
                "code": "KeyA",
                "keydown_time_seconds": "0.5",
                "audio_time_seconds": "0.5",
                "window_start_seconds": "0.48",
                "window_end_seconds": "0.545",
                "window_duration_seconds": "0.065",
                "prompt_set": "test",
                "prompt_index": 0,
                "prompt_text": "a",
            }
        )

    records, spectrogram_manifest_path, report_path, preview_path = generate_session_spectrograms(
        clip_manifest_path=manifest_path,
        output_root=tmp_path / "spectrograms",
        mel_bands=8,
        fft_window_size=128,
        hop_length=64,
        preview_count=1,
    )

    assert len(records) == 1
    assert spectrogram_manifest_path.exists()
    assert report_path.exists()
    assert preview_path.exists()
    preview_html = preview_path.read_text(encoding="utf-8")
    assert 'id="trialSelect"' in preview_html
    assert 'id="preview-data"' in preview_html
    assert "drawSpectrogram" in preview_html
    spectrogram_path = tmp_path / "spectrograms" / "session_001" / "trial_001" / "trial_001_event_000_keya_a_logmel.npz"
    assert spectrogram_path.exists()
    loaded = np.load(spectrogram_path)
    assert loaded["spectrogram"].shape[0] == 8
    assert 0.30 < float(records[0]["keydown_position_in_clip"]) < 0.31
