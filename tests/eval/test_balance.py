from __future__ import annotations

import numpy as np
import pandas as pd

from eval.balance import build_balanced_window


def test_build_balanced_window_trims_to_min_replicates() -> None:
    dates = pd.date_range("2024-01-01", periods=7, freq="B")
    frame = pd.DataFrame(
        {
            "asset_a": np.arange(7, dtype=float),
            "asset_b": np.arange(1, 8, dtype=float),
        },
        index=dates,
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 1], dtype=int)

    result = build_balanced_window(frame, labels, min_replicates=3)

    assert result.frame.shape[0] == 6
    assert result.frame.index.equals(dates[[0, 1, 2, 3, 4, 5]])
    assert set(result.frame.columns) == {"asset_a", "asset_b"}
    assert result.telemetry.reps_per_group == {0: 3, 1: 3}
    assert result.telemetry.rows_dropped == 1
    assert result.reason == "trimmed"


def test_build_balanced_window_intersects_assets_across_groups() -> None:
    dates = pd.date_range("2024-02-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "asset_a": np.arange(6, dtype=float),
            "asset_b": np.arange(10, 16, dtype=float),
        },
        index=dates,
    )
    frame.loc[dates[4], "asset_b"] = np.nan
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    result = build_balanced_window(frame, labels, min_replicates=3)

    assert set(result.frame.columns) == {"asset_a"}
    assert result.telemetry.assets_dropped == 1
    assert result.reason in {"trimmed", "ok"}


def test_build_balanced_window_flags_insufficient_replicates() -> None:
    dates = pd.date_range("2024-03-01", periods=4, freq="B")
    frame = pd.DataFrame(
        {
            "asset_a": np.linspace(0.0, 1.0, 4),
            "asset_b": np.linspace(1.0, 2.0, 4),
        },
        index=dates,
    )
    labels = np.array([0, 0, 1, 1], dtype=int)

    result = build_balanced_window(frame, labels, min_replicates=3)

    assert result.frame.shape[0] == 4
    assert result.telemetry.target_replicates == 2
    assert result.reason == "insufficient_reps"
