from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ledoit import lw_cov


def rolling_windows(
    panel: pd.DataFrame,
    window_weeks: int,
    horizon_weeks: int,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Yield expanding fit/hold windows over the weekly panel.

    Parameters
    ----------
    panel
        Weekly return matrix indexed by week start date.
    window_weeks
        Number of weeks used for estimation.
    horizon_weeks
        Number of weeks reserved for the hold-out period.

    Yields
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        The fit window followed by the hold window.
    """

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
    """Compute mean squared error and 95% VaR coverage error.

    Parameters
    ----------
    forecasts
        Iterable of variance or VaR forecasts.
    realised
        Iterable of realised outcomes matched with ``forecasts``.

    Returns
    -------
    dict[str, float]
        Keys ``mse`` and ``var95_coverage_error``.
    """

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
    **kwargs: Any,
) -> tuple[float, float]:
    """Compute out-of-sample variance forecasts and realised variance.

    Parameters
    ----------
    y_fit
        In-sample observations shaped ``(n_fit, p)``.
    y_hold
        Hold-out observations shaped ``(n_hold, p)``.
    w
        Portfolio weight vector of length ``p``.
    estimator
        Covariance estimator to use (``"dealias"`` or ``"lw"``).
    **kwargs
        Optional estimator-specific parameters.

    Returns
    -------
    tuple[float, float]
        Forecasted portfolio variance and realised variance.
    """

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
    """Compute realised return and volatility for the supplied weights.

    Parameters
    ----------
    returns
        Matrix of asset returns with assets as columns.
    weights
        Weight vector aligned with ``returns`` columns.

    Returns
    -------
    dict[str, float]
        Mean and standard deviation of realised portfolio returns.
    """

    portfolio_returns = returns.to_numpy() @ weights
    mean_return = float(np.mean(portfolio_returns))
    volatility = float(np.std(portfolio_returns, ddof=1))
    return {"mean": mean_return, "vol": volatility}
