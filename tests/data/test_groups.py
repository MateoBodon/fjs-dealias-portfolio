from __future__ import annotations

import pandas as pd
import pytest

from src.data.groups import VolStateConfig, group_dayofweek, group_volstate


def test_group_dayofweek_stub() -> None:
    frame = pd.DataFrame({"a": [0.0]}, index=pd.to_datetime(["2025-01-02"]))
    with pytest.raises(NotImplementedError):
        group_dayofweek(frame)


def test_group_volstate_stub() -> None:
    series = pd.Series([10.0, 12.0], index=pd.date_range("2025-01-01", periods=2, freq="D"))
    config = VolStateConfig()
    with pytest.raises(NotImplementedError):
        group_volstate(series, config=config)
