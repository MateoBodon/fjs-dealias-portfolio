# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finance.eval import oos_variance_forecast, risk_metrics, rolling_windows
from finance.io import load_prices_csv, to_daily_returns
from finance.portfolios import equal_weight
from finance.returns import weekly_panel
from fjs.spectra import plot_spectrum_with_edges, plot_spike_timeseries

DEFAULT_CONFIG = {
    "data_path": "data/prices_sample.csv",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "window_weeks": 156,
    "horizon_weeks": 4,
    "output_dir": "experiments/equity_panel/outputs",
}


def load_config(path: Path | str) -> dict[str, Any]:
    """Load experiment configuration, falling back to defaults."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    merged = DEFAULT_CONFIG | data
    return merged


def _generate_synthetic_prices(path: Path) -> None:
    """Create a synthetic price panel for quick smoke testing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2024-12-31")
    dates = pd.date_range(start, end, freq="B")
    tickers = [f"T{idx:04d}" for idx in range(200)]

    market = rng.normal(scale=0.005, size=len(dates))
    records = []
    for ticker in tickers:
        beta = rng.normal(scale=0.5)
        idiosyncratic = rng.normal(scale=0.01, size=len(dates))
        log_returns = beta * market + idiosyncratic
        prices = 100 * np.exp(np.cumsum(log_returns))
        records.append(
            pd.DataFrame({"date": dates, "ticker": ticker, "price_close": prices})
        )

    full = pd.concat(records, ignore_index=True)
    full.to_csv(path, index=False)


def _mp_edges(
    noise_variance: float, n_assets: int, n_samples: int
) -> tuple[float, float]:
    """Return approximate Marčenko–Pastur bulk edges."""

    aspect_ratio = n_assets / max(n_samples, 1)
    sqrt_ratio = np.sqrt(aspect_ratio)
    upper = noise_variance * (1.0 + sqrt_ratio) ** 2
    lower = noise_variance * max(0.0, (1.0 - sqrt_ratio) ** 2)
    return lower, upper


def _prepare_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices (or synthesise) and return daily/weekly returns."""

    data_path = Path(config["data_path"])
    if not data_path.exists():
        _generate_synthetic_prices(data_path)

    prices = load_prices_csv(data_path)
    daily_returns = to_daily_returns(prices)
    weekly_returns = weekly_panel(
        daily_returns,
        start=config["start_date"],
        end=config["end_date"],
    )
    return daily_returns, weekly_returns


def run_experiment(config_path: Path | str | None = None) -> None:
    """Execute the rolling equity forecasting experiment."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("config.yaml")
    )
    config = load_config(path)

    daily_returns, weekly_returns = _prepare_data(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cov_weekly = np.cov(weekly_returns.to_numpy(), rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(cov_weekly)
    avg_noise = float(np.median(np.diag(cov_weekly)))
    edges = _mp_edges(
        avg_noise,
        n_assets=cov_weekly.shape[0],
        n_samples=weekly_returns.shape[0],
    )

    plot_spectrum_with_edges(
        eigenvalues,
        edges=edges,
        out_path=output_dir / "spectrum.png",
        title="Weekly covariance spectrum",
    )
    plot_spectrum_with_edges(
        eigenvalues,
        edges=edges,
        out_path=output_dir / "spectrum.pdf",
        title="Weekly covariance spectrum",
    )

    window_weeks = int(config["window_weeks"])
    horizon_weeks = int(config["horizon_weeks"])

    w = equal_weight(len(weekly_returns.columns))

    records: list[dict[str, Any]] = []
    var_forecasts_de: list[float] = []
    var_forecasts_lw: list[float] = []
    realized_vars: list[float] = []
    var95_de: list[float] = []
    var95_lw: list[float] = []
    realized_returns: list[float] = []

    for fit, hold in rolling_windows(weekly_returns, window_weeks, horizon_weeks):
        if hold.empty:
            continue

        y_fit = fit.to_numpy(dtype=np.float64)
        y_hold = hold.to_numpy(dtype=np.float64)

        forecast_de, realized_de = oos_variance_forecast(
            y_fit, y_hold, w, estimator="dealias"
        )
        forecast_lw, realized_lw = oos_variance_forecast(
            y_fit, y_hold, w, estimator="lw"
        )

        portfolio_returns = y_hold @ w
        var_forecasts_de.append(forecast_de)
        var_forecasts_lw.append(forecast_lw)
        realized_vars.append(realized_de)

        var95 = -1.65 * np.sqrt(max(forecast_de, 0.0))
        var95_alt = -1.65 * np.sqrt(max(forecast_lw, 0.0))
        var95_de.extend([var95] * len(portfolio_returns))
        var95_lw.extend([var95_alt] * len(portfolio_returns))
        realized_returns.extend(portfolio_returns.tolist())

        records.append(
            {
                "fit_start": fit.index[0],
                "fit_end": fit.index[-1],
                "hold_start": hold.index[0],
                "forecast_var_de": forecast_de,
                "forecast_var_lw": forecast_lw,
                "realized_var": realized_de,
            }
        )

    metrics_de = risk_metrics(var95_de, realized_returns)
    metrics_lw = risk_metrics(var95_lw, realized_returns)

    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "rolling_results.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {"estimator": "dealias", **metrics_de},
            {"estimator": "lw", **metrics_lw},
        ]
    )
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    x_axis = np.arange(len(var_forecasts_de))
    plot_spike_timeseries(
        x_axis,
        var_forecasts_lw,
        var_forecasts_de,
        out_path=output_dir / "variance_forecasts.png",
        title="Forecast variance comparison",
        xlabel="Window",
        ylabel="Variance",
    )

    plot_spike_timeseries(
        np.arange(len(var95_de)),
        var95_lw,
        var95_de,
        out_path=output_dir / "var95_forecasts.png",
        title="95% VaR comparison",
        xlabel="Hold observation",
        ylabel="VaR",
    )


def main() -> None:
    """Entry point for CLI execution."""

    run_experiment()


if __name__ == "__main__":
    main()
