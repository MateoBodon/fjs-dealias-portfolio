from __future__ import annotations

from inspect import isclass, isfunction
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.equity_panel import run as equity_run
from experiments.synthetic_oneway import run as synthetic_run

from finance import (
    OptimizationResult,
    build_design_matrix,
    compute_log_returns,
    equal_weight,
    evaluate_portfolio,
    ledoit_wolf_shrinkage,
    load_market_data,
    oos_variance_forecast,
    optimize_portfolio,
    risk_metrics,
    rolling_windows,
    to_daily_returns,
    weekly_panel,
)
from fjs import (
    BalancedConfig,
    DealiasingResult,
    MarchenkoPasturModel,
    compute_balanced_weights,
    dealias_covariance,
    estimate_spectrum,
    marchenko_pastur_edges,
    marchenko_pastur_pdf,
)


def test_core_types_are_accessible() -> None:
    assert isclass(BalancedConfig)
    assert isclass(DealiasingResult)
    assert isclass(MarchenkoPasturModel)
    assert isclass(OptimizationResult)


def test_core_functions_are_callable() -> None:
    for candidate in [
        compute_balanced_weights,
        dealias_covariance,
        estimate_spectrum,
        marchenko_pastur_edges,
        marchenko_pastur_pdf,
        load_market_data,
        compute_log_returns,
        build_design_matrix,
        ledoit_wolf_shrinkage,
        optimize_portfolio,
        evaluate_portfolio,
        equal_weight,
        oos_variance_forecast,
        risk_metrics,
        rolling_windows,
        weekly_panel,
    ]:
        assert isfunction(candidate)


def test_experiment_entry_points_are_callable() -> None:
    assert isfunction(synthetic_run.run_experiment)
    assert isfunction(equity_run.run_experiment)


def test_experiment_configs_load() -> None:
    synth_config = synthetic_run.load_config(
        Path("experiments/synthetic_oneway/config.yaml")
    )
    equity_config = equity_run.load_config(Path("experiments/equity_panel/config.yaml"))
    assert isinstance(synth_config, dict)
    assert isinstance(equity_config, dict)


def test_single_equity_window_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-06", periods=15, freq="B")
    tickers = ["A", "B", "C", "D"]

    records = []
    for ticker in tickers:
        returns = rng.normal(scale=0.01, size=len(dates))
        prices = 100.0 * np.exp(np.cumsum(returns))
        records.append(
            pd.DataFrame({"date": dates, "ticker": ticker, "price_close": prices})
        )

    price_frame = pd.concat(records, ignore_index=True)
    csv_path = tmp_path / "prices.csv"
    price_frame.to_csv(csv_path, index=False)

    prices = load_market_data(csv_path)
    daily_returns = to_daily_returns(prices)
    weekly_returns = weekly_panel(daily_returns, start=dates[0], end=dates[-1])
    windows = list(rolling_windows(weekly_returns, window_weeks=2, horizon_weeks=1))
    assert windows

    fit, hold = windows[0]
    weights = equal_weight(weekly_returns.shape[1])

    forecast_de, realized_de = oos_variance_forecast(
        fit.to_numpy(dtype=np.float64),
        hold.to_numpy(dtype=np.float64),
        weights,
        estimator="dealias",
    )
    forecast_lw, realized_lw = oos_variance_forecast(
        fit.to_numpy(dtype=np.float64),
        hold.to_numpy(dtype=np.float64),
        weights,
        estimator="lw",
    )

    metrics = risk_metrics([forecast_lw], [realized_lw])
    assert np.isfinite(forecast_de)
    assert np.isfinite(forecast_lw)
    assert np.isfinite(metrics["mse"])
