from __future__ import annotations

import numpy as np
import pandas as pd

from eval.clean import apply_nan_policy


def test_apply_nan_policy_drops_assets_above_threshold() -> None:
    dates = pd.date_range("2024-04-01", periods=5, freq="B")
    frame = pd.DataFrame(
        {
            "asset_a": np.linspace(0.0, 1.0, 5),
            "asset_b": [np.nan, 0.1, np.nan, 0.3, 0.4],
            "asset_c": [0.5, 0.6, 0.7, 0.8, 0.9],
        },
        index=dates,
    )
    labels = np.zeros(len(frame), dtype=int)

    result = apply_nan_policy(
        frame,
        labels,
        max_missing_asset=0.39,
        max_missing_group_row=1.0,
    )

    assert list(result.frame.columns) == ["asset_a", "asset_c"]
    assert result.telemetry.assets_dropped == 1
    assert result.telemetry.assets_retained == 2
    assert np.array_equal(result.labels, labels)


def test_apply_nan_policy_drops_rows_with_zero_tolerance() -> None:
    dates = pd.date_range("2024-05-01", periods=4, freq="B")
    frame = pd.DataFrame(
        {
            "asset_a": [0.1, 0.2, 0.3, 0.4],
            "asset_b": [0.5, np.nan, 0.7, 0.8],
        },
        index=dates,
    )
    labels = np.array([0, 0, 1, 1], dtype=int)

    result = apply_nan_policy(
        frame,
        labels,
        max_missing_asset=1.0,
        max_missing_group_row=0.0,
    )

    assert result.frame.shape[0] == 3
    assert np.isnan(result.frame.to_numpy()).sum() == 0
    assert result.labels.shape[0] == 3
    assert result.telemetry.rows_dropped == 1
