"""
Grouping utilities for replicated covariance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

__all__ = ["VolStateConfig", "group_dayofweek", "group_volstate"]


@dataclass(frozen=True)
class VolStateConfig:
    """Configuration for the volatility state grouping."""

    n_bins: int = 4
    method: str = "quantile"
    realized_span: int = 21

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("Volatility state grouping requires at least two bins.")
        if self.method not in {"quantile"}:
            raise ValueError(f"Unsupported vol-state method: {self.method!r}")
        if self.realized_span <= 1:
            raise ValueError("realized_span must exceed 1 day to smooth realised volatility.")


def group_dayofweek(returns: pd.DataFrame) -> pd.Series:
    """
    Assign a Day-of-Week group label (1=Monday, …, 5=Friday) to each row.
    """

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns DataFrame must be indexed by dates.")
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    index = returns.index.tz_localize(None)
    day_numbers = index.dayofweek
    if np.any(day_numbers >= 5):
        raise ValueError("Day-of-Week grouping expects Monday–Friday trading days only.")

    labels = pd.Series(day_numbers + 1, index=index, name="day_of_week")
    return labels.astype(int)


def _default_vol_labels(n_bins: int) -> list[str]:
    if n_bins == 2:
        return ["low", "high"]
    if n_bins == 3:
        return ["low", "medium", "high"]
    if n_bins == 4:
        return ["low", "medium", "high", "crash"]
    return [f"vol_bin_{i+1}" for i in range(n_bins)]


def _realised_volatility_proxy(returns: pd.DataFrame, span: int) -> pd.Series:
    """
    Build a realised-volatility proxy using an EWMA of cross-sectional variance.
    """

    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")
    numeric = returns.astype(float)
    squared_mean = numeric.pow(2).mean(axis=1)
    ewma_var = squared_mean.ewm(span=max(span, 2), adjust=False, min_periods=1).mean()
    realised = np.sqrt(ewma_var)
    realised.name = "realised_vol"
    return realised


def _prepare_cfg(config: VolStateConfig | None, n_bins: int | None) -> VolStateConfig:
    base = config or VolStateConfig()
    if n_bins is not None and n_bins != base.n_bins:
        return replace(base, n_bins=int(n_bins))
    return base


def group_volstate(
    returns_or_proxy: pd.DataFrame | Iterable[float] | pd.Series,
    vix: pd.Series | None = None,
    *,
    config: VolStateConfig | None = None,
    n_bins: int | None = None,
) -> pd.Series:
    """
    Bucket each date according to a volatility proxy (VIX when supplied, otherwise
    a realised-volatility fallback computed from the return panel).
    """

    cfg = _prepare_cfg(config, n_bins)

    if isinstance(returns_or_proxy, pd.DataFrame):
        returns = returns_or_proxy.copy()
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("Returns DataFrame must be indexed by dates.")
        if returns.empty:
            raise ValueError("Returns DataFrame is empty.")
        index = returns.index.tz_localize(None)
        returns.index = index

        proxy = None
        if vix is not None:
            if not isinstance(vix.index, pd.DatetimeIndex):
                raise ValueError("VIX series must be indexed by dates.")
            proxy = vix.astype(float).copy()
            proxy.index = proxy.index.tz_localize(None)
            proxy = proxy.reindex(index)

        realised = _realised_volatility_proxy(returns, cfg.realized_span)
        if proxy is None:
            series = realised
        else:
            proxy = proxy.astype(float)
            missing = proxy.isna()
            if missing.any():
                proxy.loc[missing] = realised.loc[missing]
            series = proxy.ffill().bfill()
            if series.isna().any():
                raise ValueError("Volatility proxy contains NaN after alignment and fallback.")
        series.index = index
    else:
        if vix is not None:
            raise ValueError("Do not supply VIX separately when passing a volatility proxy series.")
        if isinstance(returns_or_proxy, pd.Series):
            series = returns_or_proxy.astype(float).copy()
        else:
            values = list(returns_or_proxy)
            series = pd.Series(values, dtype=float)
        if series.empty:
            raise ValueError("Volatility proxy is empty.")
        index = series.index

    series = series.astype(float)
    finite_mask = np.isfinite(series)
    if not finite_mask.any():
        raise ValueError("Volatility proxy does not contain finite values.")

    valid = series[finite_mask]
    ranks = valid.rank(method="first")
    fractions = (ranks - 1.0) / max(float(valid.shape[0]), 1.0)
    bins = np.floor(fractions * cfg.n_bins).astype(int)
    bins = np.clip(bins, 0, cfg.n_bins - 1)
    labels = _default_vol_labels(cfg.n_bins)

    mapped = [labels[idx] for idx in bins.to_numpy()]
    result = pd.Series(pd.NA, index=index, dtype="object", name="vol_state")
    result.loc[valid.index] = mapped
    dtype = CategoricalDtype(categories=labels, ordered=True)
    return result.astype(dtype)
