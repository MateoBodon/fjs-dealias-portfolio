from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = {"date", "ticker", "price_close"}
REQUIRED_RET_COLUMNS = {"date", "ticker", "ret"}


def load_prices_csv(path: str) -> pd.DataFrame:
    """
    Load a tidy price history CSV with canonical dtypes.

    Parameters
    ----------
    path
        Location of the CSV file containing `date`, `ticker`, and
        `price_close` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame sorted by date/ticker and limited to the required columns. The
        columns use the canonical dtypes: `date` (`datetime64[ns]`), `ticker`
        (string dtype), and `price_close` (`float64`).
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Pricing file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pricing file missing required columns: {sorted(missing)}")

    df = df.loc[:, ["date", "ticker", "price_close"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype("string")
    df["price_close"] = pd.to_numeric(df["price_close"], errors="coerce")
    df = df.dropna(subset=["price_close"]).sort_values(["date", "ticker"])
    df = df.reset_index(drop=True)
    df["price_close"] = df["price_close"].astype("float64")
    return df


def to_daily_returns(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy prices to a wide matrix of daily log returns.

    Parameters
    ----------
    price_frame
        DataFrame produced by :func:`load_prices_csv`.

    Returns
    -------
    pandas.DataFrame
        Matrix of log returns indexed by date with tickers as columns.
    """

    required = REQUIRED_PRICE_COLUMNS - set(price_frame.columns)
    if required:
        raise ValueError(f"DataFrame missing required columns: {sorted(required)}")

    pivot = price_frame.pivot(index="date", columns="ticker", values="price_close")
    pivot = pivot.sort_index()
    log_prices = cast(pd.DataFrame, np.log(pivot))
    log_returns = log_prices.diff().dropna(how="all")
    log_returns = log_returns.dropna(axis=0, how="any")
    log_returns.index.name = "date"
    return log_returns


def load_market_data(path: str | Path, *, parse_dates: bool = True) -> pd.DataFrame:
    """Backward-compatible alias for :func:`load_prices_csv`.

    Parameters
    ----------
    path
        Location of the pricing CSV.
    parse_dates
        Unused compatibility flag; maintained for legacy callers.

    Returns
    -------
    pandas.DataFrame
        Same object returned by :func:`load_prices_csv`.
    """

    _ = parse_dates  # kept for API compatibility
    return load_prices_csv(str(path))


def load_returns_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a tidy daily returns CSV into a wide date-indexed matrix.

    Expected columns: ``date``, ``ticker``, ``ret``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Returns file not found: {p}")
    df = pd.read_csv(p)
    missing = REQUIRED_RET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Returns file missing required columns: {sorted(missing)}")
    df = df.loc[:, ["date", "ticker", "ret"]].copy()
    df["date"] = pd.to_datetime(df["date"], utc=False).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype("string")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df = df.dropna(subset=["ret"]).sort_values(["date", "ticker"])  # type: ignore[list-item]
    # Wide pivot
    wide = df.pivot(index="date", columns="ticker", values="ret").sort_index()
    return wide
