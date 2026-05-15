from __future__ import annotations

from keyboard_fusion.alignment import align_keydown_event, keydown_events


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


def test_keydown_events_ignores_repeats_by_default() -> None:
    events = [
        {"event_type": "keydown", "key": "a", "repeat": "False"},
        {"event_type": "keydown", "key": "a", "repeat": "True"},
        {"event_type": "keyup", "key": "a", "repeat": "False"},
    ]

    assert [event["key"] for event in keydown_events(events)] == ["a"]
    assert [event["key"] for event in keydown_events(events, include_repeats=True)] == ["a", "a"]
