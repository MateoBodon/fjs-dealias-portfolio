from __future__ import annotations

import pandas as pd
import pytest

from src.data.groups import VolStateConfig, group_dayofweek, group_volstate


def test_group_dayofweek_assigns_expected_labels() -> None:
    index = pd.date_range("2025-01-06", periods=5, freq="B")
    frame = pd.DataFrame({"a": range(5)}, index=index)
    labels = group_dayofweek(frame)
    assert list(labels) == [1, 2, 3, 4, 5]
    assert labels.name == "day_of_week"


def test_group_dayofweek_rejects_weekends() -> None:
    frame = pd.DataFrame({"a": [0.0]}, index=pd.to_datetime(["2025-01-05"]))
    with pytest.raises(ValueError):
        group_dayofweek(frame)


def test_group_volstate_quantile_bins() -> None:
    series = pd.Series(
        [10.0, 12.0, 15.0, 30.0, 45.0, 50.0, 80.0, 120.0],
        index=pd.date_range("2025-01-06", periods=8, freq="B"),
    )
    labels = group_volstate(series, config=VolStateConfig(n_bins=4))
    assert labels.dtype.name == "category"
    assert set(labels.dropna().unique()) == {"low", "medium", "high", "crash"}
    assert labels.isna().sum() == 0


def test_group_volstate_requires_enough_samples() -> None:
    series = pd.Series([10.0, 12.0], index=pd.date_range("2025-01-06", periods=2, freq="B"))
    with pytest.raises(ValueError):
        group_volstate(series, config=VolStateConfig(n_bins=4))
