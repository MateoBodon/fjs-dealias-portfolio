from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines.factors import load_observed_factors


def _write_factor_csv(path: Path, scale: float = 1.0) -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    frame = pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": scale * np.linspace(0.1, 0.2, len(dates)),
            "SMB": scale * 0.05,
            "Mom": scale * 0.03,
            "RF": scale * 0.01,
        }
    )
    frame.to_csv(path, index=False)


def test_load_observed_factors_prefers_explicit_path(tmp_path: Path) -> None:
    factor_path = tmp_path / "factors_ff5_mom.csv"
    _write_factor_csv(factor_path, scale=1.0)

    factors = load_observed_factors(path=factor_path)
    assert list(factors.columns) == ["MKT", "SMB", "MOM", "RF"]
    assert np.isclose(factors.iloc[0, 0], 0.1)
    assert factors.index.is_monotonic_increasing

    # Cached subsequent call should produce the same values without mutating the cache
    factors_again = load_observed_factors(path=factor_path)
    pd.testing.assert_frame_equal(factors, factors_again)


def test_load_observed_factors_builds_proxy(tmp_path: Path) -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01],
            "B": [0.015, -0.005, 0.02],
        },
        index=pd.date_range("2024-02-01", periods=3, freq="B"),
    )

    factors = load_observed_factors(returns=returns, data_dir=tmp_path)
    assert list(factors.columns) == ["MKT"]
    expected = returns.mean(axis=1)
    pd.testing.assert_series_equal(factors["MKT"], expected, check_names=False)


def test_load_observed_factors_auto_scales_percentages(tmp_path: Path) -> None:
    factor_path = tmp_path / "factors_ff5_mom.csv"
    _write_factor_csv(factor_path, scale=100.0)

    factors = load_observed_factors(path=factor_path)
    assert np.isclose(factors.iloc[0, 0], 0.1)


def test_load_observed_factors_errors_without_returns(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_observed_factors(data_dir=tmp_path)
