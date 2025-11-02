from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DailyLoaderConfig, load_daily_panel


def _toy_returns_frame() -> pd.DataFrame:
    records = [
        ("2025-01-01", "A", 0.10),
        ("2025-01-01", "B", -0.50),
        ("2025-01-01", "C", 0.00),
        ("2025-01-02", "A", 5.00),
        ("2025-01-02", "B", 0.03),
        ("2025-01-02", "C", 0.02),
        ("2025-01-03", "A", -0.04),
        ("2025-01-03", "B", -0.07),
        ("2025-01-03", "C", 0.04),
    ]
    return pd.DataFrame(records, columns=["date", "ticker", "ret"])


def test_load_daily_panel_balances_and_winsorises() -> None:
    df = _toy_returns_frame()
    config = DailyLoaderConfig(winsor_lower=0.1, winsor_upper=0.9, min_history=3)
    panel = load_daily_panel(df, config=config)

    returns = panel.returns
    assert returns.shape == (3, 3)
    assert returns.index.is_monotonic_increasing
    assert list(returns.columns) == ["A", "B", "C"]
    # Winsorisation clamps cross-sectional extremes.
    assert pytest.approx(returns.loc[pd.Timestamp("2025-01-01"), "A"], rel=1e-6) == 0.08
    assert pytest.approx(returns.loc[pd.Timestamp("2025-01-02"), "A"], rel=1e-6) == 4.006
    assert pytest.approx(returns.loc[pd.Timestamp("2025-01-02"), "C"], rel=1e-6) == 0.022

    meta = panel.meta
    assert meta["p"] == 3
    assert meta["n_days"] == 3
    assert meta["symbols"] == ["A", "B", "C"]


def test_load_daily_panel_required_symbol_missing_errors() -> None:
    df = _toy_returns_frame()
    config = DailyLoaderConfig(min_history=3, required_symbols=["Z"])
    with pytest.raises(ValueError):
        load_daily_panel(df, config=config)
