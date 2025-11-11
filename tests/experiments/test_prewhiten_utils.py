from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.prewhiten import apply_prewhitening


def _mock_returns(rows: int = 20, cols: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    data = rng.normal(scale=0.01, size=(rows, cols))
    return pd.DataFrame(data, index=dates, columns=[f"A{idx}" for idx in range(cols)])


def _mock_factors(rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(7)
    columns = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    data = rng.normal(scale=0.005, size=(rows, len(columns)))
    return pd.DataFrame(data, index=dates, columns=columns)


def test_apply_prewhitening_off_mode_returns_identity() -> None:
    returns = _mock_returns()
    whitening, telemetry = apply_prewhitening(returns, factors=None, requested_mode="off")

    assert telemetry.mode_effective == "off"
    assert telemetry.factor_columns == ()
    np.testing.assert_allclose(whitening.residuals.values, returns.values)


def test_apply_prewhitening_with_factors_uses_requested_mode() -> None:
    returns = _mock_returns()
    factors = _mock_factors()

    whitening, telemetry = apply_prewhitening(returns, factors=factors, requested_mode="ff5mom")

    assert telemetry.mode_effective == "ff5mom"
    assert len(telemetry.factor_columns) == 6
    assert not np.allclose(whitening.residuals.values, returns.values)
