"""
Grouping utilities for replicated covariance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError("Volatility state grouping requires at least two bins.")
        if self.method not in {"quantile"}:
            raise ValueError(f"Unsupported vol-state method: {self.method!r}")


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
    if n_bins == 4:
        return ["low", "medium", "high", "crash"]
    return [f"vol_bin_{i+1}" for i in range(n_bins)]


def group_volstate(
    volatility_proxy: Iterable[float] | pd.Series,
    *,
    config: VolStateConfig | None = None,
) -> pd.Series:
    """
    Bucket each date according to a volatility proxy (e.g., VIX or realised vol).
    """

    cfg = config or VolStateConfig()
    if isinstance(volatility_proxy, pd.Series):
        original_index = volatility_proxy.index
        series = volatility_proxy.astype(float)
    else:
        values = list(volatility_proxy)
        series = pd.Series(values, dtype=float)
        original_index = series.index
    if series.empty:
        raise ValueError("Volatility proxy is empty.")

    series = series.dropna()
    if series.empty:
        raise ValueError("Volatility proxy does not contain finite values.")
    if series.shape[0] < cfg.n_bins:
        raise ValueError("Not enough observations to form the requested number of bins.")

    rank = series.rank(method="first")
    fractions = (rank - 1.0) / float(series.shape[0])
    bins = np.floor(fractions * cfg.n_bins).astype(int)
    bins = np.clip(bins, 0, cfg.n_bins - 1)

    labels = _default_vol_labels(cfg.n_bins)
    categorical = pd.Categorical.from_codes(bins.to_numpy(), categories=labels, ordered=True)

    result = pd.Series(pd.NA, index=original_index, dtype="object", name="vol_state")
    result.loc[series.index] = list(categorical)
    dtype = CategoricalDtype(categories=labels, ordered=True)
    return result.astype(dtype)
