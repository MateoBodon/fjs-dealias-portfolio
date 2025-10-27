from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from data.panels import build_balanced_weekday_panel


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for a wide price DataFrame.

    Parameters
    ----------
    prices
        Wide DataFrame of prices indexed by business date with tickers as
        columns.

    Returns
    -------
    pandas.DataFrame
        Log return matrix sorted by date.
    """

    if prices.index.inferred_type != "datetime64":
        raise ValueError("Prices index must be a DatetimeIndex.")
    prices = prices.sort_index()
    log_prices = cast(pd.DataFrame, np.log(prices))
    returns = log_prices.diff().dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    return returns


def weekly_panel(
    daily_returns: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Aggregate daily log returns into weekly (Monday-start) log returns.

    Parameters
    ----------
    daily_returns
        Wide matrix of daily log returns indexed by date.
    start, end
        Inclusive date range delimiting the sample window.

    Returns
    -------
    tuple[pandas.DataFrame, int]
        Tuple containing the weekly log return panel indexed by week-start
        date, and the number of weeks dropped because they contained only
        missing values.
    """

    if daily_returns.index.inferred_type != "datetime64":
        raise ValueError("daily_returns must use a DatetimeIndex.")

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    mask = (daily_returns.index >= start_ts) & (daily_returns.index <= end_ts)
    subset = daily_returns.loc[mask]
    if subset.empty:
        raise ValueError("No data within the requested window.")

    weekly = subset.resample("W-FRI").sum(min_count=1)
    week_index = weekly.index
    if not isinstance(week_index, pd.DatetimeIndex):
        raise TypeError("Weekly aggregation must yield a DatetimeIndex.")
    adjusted_index = week_index - pd.Timedelta(days=4)
    weekly.index = adjusted_index.rename("week_start")
    total_weeks = int(weekly.shape[0])
    weekly = weekly.dropna(axis=0, how="all")
    dropped_weeks = total_weeks - int(weekly.shape[0])
    weekly = weekly.dropna(axis=1, how="all")
    return weekly, dropped_weeks


def balance_weeks(
    panel: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create a balanced week/day design from daily returns."""

    balanced = build_balanced_weekday_panel(panel, partial_week_policy="drop")
    ordered = balanced.ordered_tickers
    replicates = balanced.replicates
    balanced_blocks: list[np.ndarray] = []
    groups: list[int] = []

    for idx, week_start in enumerate(balanced.weekly.index):
        frame = balanced.week_map[week_start].loc[:, ordered]
        block = frame.to_numpy(dtype=np.float64)
        balanced_blocks.append(block)
        groups.extend([idx] * replicates)

    y_matrix = np.vstack(balanced_blocks)
    groups_array = np.asarray(groups, dtype=np.intp)
    week_index = cast(pd.DatetimeIndex, balanced.weekly.index.rename(None))
    return y_matrix, groups_array, week_index
