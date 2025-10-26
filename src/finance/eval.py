from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from collections.abc import Iterable as IterableType
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fjs.balanced import mean_squares
from fjs.dealias import dealias_covariance

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
    estimator: Literal["dealias", "lw", "scm"],
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
    elif estimator == "scm":
        # Unbiased sample covariance on fit data
        if x_fit.shape[0] <= 1:
            sigma = np.asarray(np.cov(x_fit, rowvar=False), dtype=np.float64)
        else:
            sigma = np.asarray(np.cov(x_fit, rowvar=False, ddof=1), dtype=np.float64)
    elif estimator == "dealias":
        sigma_emp = np.cov(x_fit, rowvar=False, ddof=1)
        detections = kwargs.get("detections")
        if detections:
            result = dealias_covariance(sigma_emp, detections)
            sigma = result.covariance
        else:
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


def weekly_cov_from_components(
    ms1: NDArray[np.float64],
    ms2: NDArray[np.float64],
    replicates: int | float,
    mu_hats: Sequence[float] | None = None,
    vecs: NDArray[np.float64] | None = None,
    clip_top: int | None = None,
) -> NDArray[np.float64]:
    """
    Construct the weekly covariance of summed daily returns from MANOVA components.

    Parameters
    ----------
    ms1, ms2
        Balanced one-way MANOVA mean squares shaped ``(p, p)``.
    replicates
        Number of trading days (or replicates) per group.
    mu_hats
        Optional sequence of adjusted spike variances for the between-group component.
    vecs
        Optional matrix whose columns contain the eigenvectors aligned with
        ``mu_hats``.
    clip_top
        Optional cap on the number of leading directions substituted by ``mu_hats``.
    """

    ms1_arr = np.asarray(ms1, dtype=np.float64)
    ms2_arr = np.asarray(ms2, dtype=np.float64)
    if ms1_arr.shape != ms2_arr.shape:
        raise ValueError("MS1 and MS2 must have matching shapes.")
    if ms1_arr.ndim != 2 or ms1_arr.shape[0] != ms1_arr.shape[1]:
        raise ValueError("MS1 must be a square matrix.")

    replicates_float = float(replicates)
    if not np.isfinite(replicates_float) or replicates_float <= 0.0:
        raise ValueError("replicates must be a positive scalar.")

    sigma1_hat = (ms1_arr - ms2_arr) / replicates_float
    sigma2_hat = ms2_arr
    sigma1_adj = sigma1_hat.copy()

    if mu_hats is not None and vecs is not None:
        mu_arr = np.asarray(mu_hats, dtype=np.float64).reshape(-1)
        if mu_arr.size == 0:
            pass
        else:
            vecs_arr = np.asarray(vecs, dtype=np.float64)
            if vecs_arr.ndim != 2:
                raise ValueError("vecs must be a two-dimensional array.")
            if vecs_arr.shape[0] != ms1_arr.shape[0]:
                raise ValueError("vecs columns must match MS1 dimension.")

            k = mu_arr.size
            if clip_top is not None:
                if clip_top <= 0:
                    raise ValueError("clip_top must be positive when provided.")
                k = min(k, int(clip_top))
            k = min(k, vecs_arr.shape[1])
            if k > 0:
                for idx in range(k):
                    vec = vecs_arr[:, idx].astype(np.float64, copy=True)
                    norm = np.linalg.norm(vec)
                    if not np.isfinite(norm) or norm <= 0.0:
                        continue
                    vec /= norm
                    outer = np.outer(vec, vec)
                    rayleigh = float(vec.T @ sigma1_adj @ vec)
                    sigma1_adj += (float(mu_arr[idx]) - rayleigh) * outer

    weekly_covariance = (
        replicates_float * replicates_float
    ) * sigma1_adj + replicates_float * sigma2_hat
    return weekly_covariance


def variance_forecast_from_components(
    y_fit: IterableType[Sequence[float]] | NDArray[np.float64],
    y_hold: IterableType[Sequence[float]] | NDArray[np.float64],
    replicates: int,
    w: Sequence[float] | NDArray[np.float64],
    detections: Sequence[dict[str, Any]] | None = None,
) -> tuple[float, float]:
    """
    Forecast portfolio variance from balanced MANOVA components and compare to realised.

    Parameters
    ----------
    y_fit, y_hold
        Daily return panels shaped ``(n_days, p)`` for fit and hold-out windows.
    replicates
        Number of trading days per week.
    w
        Portfolio weights of length ``p``.
    detections
        Optional sequence where each entry contains ``mu_hat`` and ``eigvec`` fields.
    """

    x_fit = np.asarray(y_fit, dtype=np.float64)
    x_hold = np.asarray(y_hold, dtype=np.float64)
    w_vec = np.asarray(w, dtype=np.float64).reshape(-1)

    if x_fit.ndim != 2:
        raise ValueError("y_fit must be a two-dimensional array.")
    if x_hold.ndim != 2:
        raise ValueError("y_hold must be a two-dimensional array.")
    if w_vec.size != x_fit.shape[1]:
        raise ValueError("Weights must align with the number of assets.")
    if x_hold.shape[1] != x_fit.shape[1]:
        raise ValueError("y_hold must have the same number of assets as y_fit.")
    if replicates <= 0:
        raise ValueError("replicates must be positive.")
    if x_fit.shape[0] % replicates != 0:
        raise ValueError("Number of fit observations must be a multiple of replicates.")

    n_weeks_fit = x_fit.shape[0] // replicates
    groups_fit = np.repeat(np.arange(n_weeks_fit), replicates)
    stats = mean_squares(x_fit, groups_fit)
    ms1 = stats["MS1"].astype(np.float64)
    ms2 = stats["MS2"].astype(np.float64)

    if detections:
        mu_values = [float(item["mu_hat"]) for item in detections]
        vec_columns = [
            np.asarray(item["eigvec"], dtype=np.float64).reshape(-1)
            for item in detections
        ]
        if vec_columns:
            vec_matrix = np.column_stack(vec_columns)
            weekly_covariance = weekly_cov_from_components(
                ms1,
                ms2,
                replicates,
                mu_hats=mu_values,
                vecs=vec_matrix,
                clip_top=len(mu_values),
            )
        else:
            weekly_covariance = weekly_cov_from_components(ms1, ms2, replicates)
    else:
        weekly_covariance = weekly_cov_from_components(ms1, ms2, replicates)

    forecast_var = float(w_vec.T @ weekly_covariance @ w_vec)

    realised_var = float("nan")
    if x_hold.size:
        n_complete_weeks = x_hold.shape[0] // replicates
        if n_complete_weeks > 0:
            trimmed = x_hold[: n_complete_weeks * replicates]
            weekly_sums = trimmed.reshape(n_complete_weeks, replicates, w_vec.size).sum(
                axis=1
            )
            weekly_portfolio = weekly_sums @ w_vec
            if weekly_portfolio.size <= 1:
                realised_var = float(np.var(weekly_portfolio))
            else:
                realised_var = float(np.var(weekly_portfolio, ddof=1))
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
