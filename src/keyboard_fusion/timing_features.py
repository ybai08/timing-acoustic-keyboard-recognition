from __future__ import annotations

import pandas as pd


def add_basic_timing_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add beginner-friendly timing features to a key event table.

    Expected columns:
    - key
    - keydown_time
    - keyup_time

    Times should be in seconds on one shared timeline.
    """
    required = {"key", "keydown_time", "keyup_time"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = events.sort_values("keydown_time").reset_index(drop=True).copy()
    df["dwell_time"] = df["keyup_time"] - df["keydown_time"]
    df["next_key"] = df["key"].shift(-1)
    df["press_press_latency"] = df["keydown_time"].shift(-1) - df["keydown_time"]
    df["release_press_latency"] = df["keydown_time"].shift(-1) - df["keyup_time"]
    df["release_release_latency"] = df["keyup_time"].shift(-1) - df["keyup_time"]
    return df

