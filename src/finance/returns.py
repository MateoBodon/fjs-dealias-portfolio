from __future__ import annotations

import numpy as np
import pandas as pd


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
    log_prices = np.log(prices)
    returns = log_prices.diff().dropna(how="all")
    returns = returns.dropna(axis=1, how="all")
    return returns


def weekly_panel(
    daily_returns: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate daily log returns into weekly (Monday-start) log returns.

    Parameters
    ----------
    daily_returns
        Wide matrix of daily log returns indexed by date.
    start, end
        Inclusive date range delimiting the sample window.

    Returns
    -------
    pandas.DataFrame
        Weekly log returns indexed by week-start date.
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
    weekly.index = (weekly.index - pd.Timedelta(days=4)).rename("week_start")
    weekly = weekly.dropna(axis=0, how="all")
    weekly = weekly.dropna(axis=1, how="all")
    return weekly


def balance_weeks(
    panel: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create a balanced week/day design from daily returns.

    Parameters
    ----------
    panel
        DataFrame of log returns indexed by date with tickers as columns.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, pandas.DatetimeIndex)
        Balanced observation matrix shaped ``(5 * W, P)``, group labels per
        row, and the corresponding week start dates.
    """

    if panel.index.inferred_type != "datetime64":
        raise ValueError("panel must use a DatetimeIndex.")

    panel = panel.sort_index()
    week_periods = panel.index.to_period("W-MON")
    grouped = panel.groupby(week_periods)

    frames: list[pd.DataFrame] = []
    week_labels: list[pd.Timestamp] = []

    for period, frame in grouped:
        frame = frame.dropna(axis=1, how="all")
        frame = frame.dropna(axis=0, how="any")
        if frame.shape[0] < 5:
            continue
        frame = frame.sort_index().iloc[:5]
        frames.append(frame)
        week_labels.append(period.start_time)

    if not frames:
        raise ValueError("No balanced weeks available in the panel.")

    common_tickers = set(frames[0].columns)
    for frame in frames[1:]:
        common_tickers &= set(frame.columns)

    if not common_tickers:
        raise ValueError("No common tickers across balanced weeks.")

    ordered_tickers = sorted(common_tickers)
    balanced_rows: list[np.ndarray] = []
    groups: list[int] = []

    for idx, frame in enumerate(frames):
        trimmed = frame.loc[:, ordered_tickers]
        if trimmed.isna().any().any():
            continue
        balanced_rows.append(trimmed.to_numpy(dtype=np.float64))
        groups.extend([idx] * trimmed.shape[0])

    if not balanced_rows:
        raise ValueError("Balanced data empty after ticker intersection.")

    y_matrix = np.vstack(balanced_rows)
    groups_array = np.asarray(groups, dtype=np.intp)
    week_index = pd.DatetimeIndex(week_labels[: len(np.unique(groups_array))])
    return y_matrix, groups_array, week_index
