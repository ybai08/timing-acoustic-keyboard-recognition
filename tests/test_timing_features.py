from __future__ import annotations

import pandas as pd
import pytest

from keyboard_fusion.timing_features import add_basic_timing_features


def test_add_basic_timing_features() -> None:
    events = pd.DataFrame(
        {
            "key": ["a", "s"],
            "keydown_time": [1.0, 1.2],
            "keyup_time": [1.1, 1.3],
        }
    )

    result = add_basic_timing_features(events)

    assert result.loc[0, "dwell_time"] == pytest.approx(0.1)
    assert result.loc[0, "next_key"] == "s"
    assert round(result.loc[0, "press_press_latency"], 3) == 0.2
    assert round(result.loc[0, "release_press_latency"], 3) == 0.1
