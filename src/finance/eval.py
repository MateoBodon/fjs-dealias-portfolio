from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ledoit import lw_cov


def rolling_windows(
    panel: pd.DataFrame,
    window_weeks: int,
    horizon_weeks: int,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Yield expanding fit/hold windows over the weekly panel."""

    if window_weeks <= 0 or horizon_weeks <= 0:
        raise ValueError("window_weeks and horizon_weeks must be positive.")
    total = len(panel)
    for start in range(0, total - window_weeks - horizon_weeks + 1):
        fit = panel.iloc[start : start + window_weeks]
        hold = panel.iloc[start + window_weeks : start + window_weeks + horizon_weeks]
        yield fit, hold


def risk_metrics(
    forecasts: Iterable[float], realised: Iterable[float]
) -> dict[str, float]:
    """Compute mean squared error and 95% VaR coverage error."""

    forecasts_arr = np.asarray(list(forecasts), dtype=float)
    realised_arr = np.asarray(list(realised), dtype=float)
    mask = np.isfinite(forecasts_arr) & np.isfinite(realised_arr)
    if not mask.any():
        return {"mse": float("nan"), "var95_coverage_error": float("nan")}

    forecasts_arr = forecasts_arr[mask]
    realised_arr = realised_arr[mask]
    mse = float(np.mean((forecasts_arr - realised_arr) ** 2))
    coverage = float(np.mean(realised_arr < forecasts_arr))
    coverage_error = coverage - 0.05
    return {"mse": mse, "var95_coverage_error": coverage_error}


def oos_variance_forecast(
    y_fit: NDArray[np.float64],
    y_hold: NDArray[np.float64],
    w: NDArray[np.float64],
    estimator: Literal["dealias", "lw"],
    **kwargs,
) -> tuple[float, float]:
    """Compute out-of-sample variance forecasts and realised variance."""

    x_fit = np.asarray(y_fit, dtype=np.float64)
    x_hold = np.asarray(y_hold, dtype=np.float64)
    w_vec = np.asarray(w, dtype=np.float64).reshape(-1)

    if estimator == "lw":
        sigma = lw_cov(x_fit)
    elif estimator == "dealias":
        sigma_emp = np.cov(x_fit, rowvar=False, ddof=1)
        clip = float(kwargs.get("clip", 1e-4))
        eigvals, eigvecs = np.linalg.eigh(sigma_emp)
        eigvals = np.clip(eigvals, clip, None)
        sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    else:
        raise ValueError(f"Unknown estimator '{estimator}'.")

    forecast_var = float(w_vec.T @ sigma @ w_vec)
    if x_hold.size == 0:
        realised_var = float("nan")
    else:
        portfolio_returns = x_hold @ w_vec
        if portfolio_returns.size > 1:
            realised_var = float(np.var(portfolio_returns, ddof=1))
        else:
            realised_var = float(portfolio_returns.var())
    return forecast_var, realised_var


def evaluate_portfolio(
    returns: pd.DataFrame,
    weights: NDArray[np.float64],
) -> dict[str, float]:
    """Compute realised return and volatility for the supplied weights."""

    portfolio_returns = returns.to_numpy() @ weights
    mean_return = float(np.mean(portfolio_returns))
    volatility = float(np.std(portfolio_returns, ddof=1))
    return {"mean": mean_return, "vol": volatility}
