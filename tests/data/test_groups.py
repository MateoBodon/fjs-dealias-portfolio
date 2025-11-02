from __future__ import annotations

import numpy as np
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


def test_group_volstate_uses_vix_series() -> None:
    dates = pd.date_range("2025-01-06", periods=8, freq="B")
    returns = pd.DataFrame(
        {
            "A": np.linspace(0.01, 0.08, len(dates)),
            "B": np.linspace(0.005, 0.07, len(dates)),
        },
        index=dates,
    )
    vix = pd.Series([12.0, 13.0, 14.0, 18.0, 24.0, 32.0, 40.0, 65.0], index=dates)
    labels = group_volstate(returns, vix=vix)
    assert labels.index.equals(dates)
    assert labels.cat.categories.tolist() == ["low", "medium", "high", "crash"]
    assert labels.iloc[0] == "low"
    assert labels.iloc[-1] == "crash"


def test_group_volstate_fallback_realised_volatility() -> None:
    dates = pd.date_range("2025-01-06", periods=10, freq="B")
    returns = pd.DataFrame(
        {
            "A": np.linspace(0.01, 0.06, len(dates)),
            "B": np.linspace(0.012, 0.07, len(dates)),
            "C": np.linspace(0.02, 0.08, len(dates)),
        },
        index=dates,
    )
    labels = group_volstate(returns, config=VolStateConfig(n_bins=3, realized_span=3))
    assert labels.isna().sum() == 0
    assert labels.cat.categories.tolist() == ["low", "medium", "high"]
    assert labels.iloc[0] == "low"
    assert "high" in set(labels.astype(str))


def test_group_volstate_accepts_series_proxy() -> None:
    series = pd.Series(
        [10.0, 12.0, 15.0, 30.0, 45.0, 50.0, 80.0, 120.0],
        index=pd.date_range("2025-01-06", periods=8, freq="B"),
    )
    labels = group_volstate(series, config=VolStateConfig(n_bins=4))
    assert labels.dtype.name == "category"
    assert set(labels.dropna().unique()) <= {"low", "medium", "high", "crash"}
    assert labels.isna().sum() == 0


def test_group_volstate_handles_small_sample() -> None:
    series = pd.Series([10.0, 12.0], index=pd.date_range("2025-01-06", periods=2, freq="B"))
    labels = group_volstate(series, config=VolStateConfig(n_bins=4))
    assert labels.isna().sum() == 0
    assert labels.iloc[0] == "low"
