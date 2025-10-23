from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd

from .io import load_prices_csv, to_daily_returns


@dataclass(frozen=True)
class WeeklyLoadResult:
    weekly: pd.DataFrame
    dropped_weeks: int
    p: int


def _balanced_weekly_from_daily(
    daily_returns: pd.DataFrame, replicates: int = 5
) -> tuple[pd.DataFrame, int, list[str]]:
    """Internal helper: balanced weekly panel with a fixed universe.

    - Drops partial weeks (requires exactly ``replicates`` daily rows per week).
    - Intersects tickers across all kept weeks to enforce a fixed universe.
    - Sums daily log returns within each week to form weekly log returns.
    """

    if daily_returns.index.inferred_type != "datetime64":
        raise ValueError("daily_returns must use a DatetimeIndex.")

    panel = daily_returns.sort_index()
    grouped = panel.groupby(panel.index.to_period("W-MON"))

    total_weeks = 0
    week_frames: list[pd.DataFrame] = []
    week_labels: list[pd.Timestamp] = []

    for period, frame in grouped:
        total_weeks += 1
        cleaned = frame.dropna(axis=1, how="all")
        cleaned = cleaned.dropna(axis=0, how="any").sort_index()
        if cleaned.shape[0] < replicates:
            continue
        trimmed = cleaned.iloc[:replicates]
        if trimmed.isna().any().any():
            continue
        week_frames.append(trimmed)
        week_labels.append(period.start_time)

    if not week_frames:
        raise ValueError("No balanced weeks available after cleaning.")

    common_tickers = set(week_frames[0].columns)
    for frame in week_frames[1:]:
        common_tickers &= set(frame.columns)
    if not common_tickers:
        raise ValueError("No common tickers across balanced weeks.")
    ordered_tickers = sorted(common_tickers)

    weekly_rows: list[np.ndarray] = []
    for frame in week_frames:
        arr = frame.loc[:, ordered_tickers].to_numpy(dtype=np.float64)
        weekly_rows.append(arr.sum(axis=0))

    weekly = pd.DataFrame(
        np.vstack(weekly_rows),
        index=pd.Index(week_labels, name="week_start"),
        columns=ordered_tickers,
    )

    dropped_weeks = int(total_weeks - weekly.shape[0])
    return weekly, dropped_weeks, ordered_tickers


def load_weekly_from_daily_csv(
    path: str | Path, *, start: str | None = None, end: str | None = None, min_p: int = 50
) -> WeeklyLoadResult:
    """Load daily prices CSV, build balanced weekly panel, and print counters.

    - Aggregates to Mon–Fri, dropping partial weeks.
    - Enforces a fixed universe across the full horizon.
    - Fails fast if the asset count (p) is below ``min_p``.
    """

    prices = load_prices_csv(str(path))
    if start is not None:
        prices = prices.loc[prices["date"] >= pd.to_datetime(start)]
    if end is not None:
        prices = prices.loc[prices["date"] <= pd.to_datetime(end)]
    if prices.empty:
        raise ValueError("No price rows after applying date filters.")

    daily = to_daily_returns(prices)
    weekly, dropped_weeks, ordered = _balanced_weekly_from_daily(daily)
    p = len(ordered)

    print(f"Dropped weeks: {dropped_weeks}; p={p}")
    if p < min_p:
        raise ValueError(
            f"Insufficient asset count after balancing: p={p} < {min_p} (demo requires high dimension)."
        )

    return WeeklyLoadResult(weekly=weekly, dropped_weeks=dropped_weeks, p=p)


def rolling_windows_fixed_universe(
    weekly: pd.DataFrame, *, window_weeks: int, horizon_weeks: int, min_p: int = 50
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Yield (fit, hold) windows with per-window fixed-universe enforcement.

    Subsets each window to the intersection of tickers that are present (no NaNs)
    in both the fit and hold slices. Fails fast if the resulting p < min_p.
    """

    if weekly.index.inferred_type != "datetime64":
        raise ValueError("weekly must use a DatetimeIndex indexed by week_start.")
    weekly = weekly.sort_index()

    total = weekly.shape[0]
    if window_weeks <= 0 or horizon_weeks <= 0:
        raise ValueError("window_weeks and horizon_weeks must be positive.")
    if total < window_weeks + horizon_weeks:
        return

    for start in range(0, total - window_weeks - horizon_weeks + 1):
        fit = weekly.iloc[start : start + window_weeks]
        hold = weekly.iloc[start + window_weeks : start + window_weeks + horizon_weeks]

        # Enforce fixed universe within this window (no NaNs across either slice)
        ok_fit = fit.columns[~fit.isna().any(axis=0)]
        ok_hold = hold.columns[~hold.isna().any(axis=0)]
        common = sorted(set(ok_fit).intersection(set(ok_hold)))
        fit_w = fit.loc[:, common]
        hold_w = hold.loc[:, common]

        p = len(common)
        print(
            f"Window {fit_w.index.min().date()}→{hold_w.index.max().date()} | p={p}"
        )
        if p < min_p:
            raise ValueError(
                f"Low-dimensional window after enforcing fixed universe: p={p} < {min_p}."
            )

        yield fit_w, hold_w

