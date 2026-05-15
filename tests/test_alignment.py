from __future__ import annotations

from keyboard_fusion.alignment import align_keydown_event, estimate_audio_start_offset, keydown_events


def test_align_keydown_event_maps_time_to_sample_window() -> None:
    alignment = align_keydown_event(
        {"event_index": "3", "event_type": "keydown", "key": "j", "code": "KeyJ", "trial_elapsed_seconds": "1.234"},
        sample_rate=1000,
        frame_count=5000,
        pre_keydown_ms=50,
        post_keydown_ms=250,
    )

    assert alignment["event_index"] == 3
    assert alignment["sample_index"] == 1234
    assert alignment["window_start_sample"] == 1184
    assert alignment["window_end_sample"] == 1484
    assert alignment["within_audio"] is True
    assert alignment["clipped_left"] is False
    assert alignment["clipped_right"] is False


def test_align_keydown_event_clips_window_to_audio_bounds() -> None:
    alignment = align_keydown_event(
        {"event_index": "0", "event_type": "keydown", "key": "a", "code": "KeyA", "trial_elapsed_seconds": "0.020"},
        sample_rate=1000,
        frame_count=5000,
        pre_keydown_ms=50,
        post_keydown_ms=250,
    )

    assert alignment["sample_index"] == 20
    assert alignment["window_start_sample"] == 0
    assert alignment["window_end_sample"] == 270
    assert alignment["clipped_left"] is True


def test_align_keydown_event_subtracts_audio_start_offset() -> None:
    alignment = align_keydown_event(
        {"event_index": "1", "event_type": "keydown", "key": "a", "code": "KeyA", "trial_elapsed_seconds": "1.000"},
        sample_rate=1000,
        frame_count=5000,
        pre_keydown_ms=20,
        post_keydown_ms=80,
        audio_start_offset_seconds=0.250,
    )

    assert alignment["audio_time_seconds"] == 0.75
    assert alignment["sample_index"] == 750
    assert alignment["window_start_sample"] == 730
    assert alignment["window_end_sample"] == 830


def test_estimate_audio_start_offset_finds_synthetic_impulse_shift() -> None:
    sample_rate = 1000
    samples = [0] * 1500
    for impulse_center in [400, 800, 1200]:
        for index in range(impulse_center - 2, impulse_center + 3):
            samples[index] = 10000

    estimate = estimate_audio_start_offset(
        samples=samples,
        sample_rate=sample_rate,
        keydown_times_seconds=[0.5, 0.9, 1.3],
        metadata_minus_wav_duration_seconds=0.1,
    )

    assert abs(estimate["offset_seconds"] - 0.1) <= 0.005


def test_keydown_events_ignores_repeats_by_default() -> None:
    events = [
        {"event_type": "keydown", "key": "a", "repeat": "False"},
        {"event_type": "keydown", "key": "a", "repeat": "True"},
        {"event_type": "keyup", "key": "a", "repeat": "False"},
    ]

    assert [event["key"] for event in keydown_events(events)] == ["a"]
    assert [event["key"] for event in keydown_events(events, include_repeats=True)] == ["a", "a"]
