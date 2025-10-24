from __future__ import annotations

"""
Build daily returns and a balanced Week×Day dataset (Mon–Fri, fixed universe).

Usage:
    python scripts/data/make_balanced_weekly.py \
        --prices data/prices_daily.csv \
        --returns-out data/returns_daily.csv \
        --balanced-out data/returns_balanced_weekly.parquet \
        --winsor 0.01

Input schema (prices CSV):
    date,ticker,adj_close,volume

Outputs:
    returns-out CSV: date,ticker,ret
    balanced-out Parquet: balanced daily rows for kept weeks with metadata columns
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", "adj_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"prices CSV must contain columns {sorted(required)}; missing {sorted(missing)}"
        )
    df = df.loc[:, ["date", "ticker", "adj_close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype("string")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])  # type: ignore[list-item]
    return df


def _to_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker daily log returns."""
    def _ret(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        r = np.log(g["adj_close"]).diff()
        out = g.loc[:, ["date", "ticker"]].copy()
        out["ret"] = r
        return out.dropna(subset=["ret"])  # type: ignore[list-item]

    returns = prices.groupby("ticker", group_keys=False).apply(_ret)
    returns = returns.sort_values(["date", "ticker"])  # type: ignore[list-item]
    return returns


def _winsorize(
    returns: pd.DataFrame, *, q: float
) -> pd.DataFrame:
    """Symmetric per-ticker winsorization at quantiles (q, 1-q)."""
    if q <= 0.0:
        return returns

    def _clip(g: pd.DataFrame) -> pd.DataFrame:
        lo = float(g["ret"].quantile(q))
        hi = float(g["ret"].quantile(1.0 - q))
        gg = g.copy()
        gg["ret"] = np.clip(gg["ret"], lo, hi)
        return gg

    return returns.groupby("ticker", group_keys=False).apply(_clip)


def _balanced_weekday_panel(returns: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return balanced Week×Day rows (Mon–Fri) with a fixed universe.

    Returns (balanced_rows, kept_weeks_count, tickers_count). balanced_rows has
    columns [date, week_start, weekday, ticker, ret] and contains only weeks with
    exactly 5 trading days and tickers that are present on all 5 days of every kept week.
    """
    df = returns.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["weekday"] = df["date"].dt.weekday
    # ISO weeks ending on Friday; convert to a canonical Monday-start label
    periods = df["date"].dt.to_period("W-FRI")
    df["week_start"] = (periods.asfreq("D", "start") - pd.Timedelta(days=4)).dt.normalize()

    # Keep weeks with exactly 5 unique dates
    date_counts = df.groupby(["week_start"]) ["date"].nunique()
    full_weeks = date_counts[date_counts == 5].index
    df = df[df["week_start"].isin(full_weeks)]

    # For each kept week, require per-ticker 5 rows
    counts = df.groupby(["week_start", "ticker"]).size().reset_index(name="n")
    complete = counts[counts["n"] == 5]
    by_week_sets = complete.groupby("week_start")["ticker"].apply(lambda s: set(s))
    # Fixed universe = intersection across all kept weeks
    universe: set[str]
    if by_week_sets.empty:
        return df.iloc[0:0], 0, 0
    universe = set.intersection(*by_week_sets.tolist())
    df_bal = df[df["ticker"].isin(universe)].copy()
    # Verify per-week/per-ticker completeness
    check = df_bal.groupby(["week_start", "ticker"]).size()
    assert int(check.min()) == 5, "Balanced output must have exactly 5 days per week"
    return df_bal.sort_values(["week_start", "date", "ticker"]), len(full_weeks), len(universe)  # type: ignore[list-item]


@dataclass(frozen=True)
class Args:
    prices: Path
    returns_out: Path
    balanced_out: Path
    winsor: float


def parse_args(argv: Sequence[str] | None = None) -> Args:
    ap = argparse.ArgumentParser(description="Build balanced weekly dataset from daily prices")
    ap.add_argument("--prices", required=True, type=Path)
    ap.add_argument("--returns-out", required=True, type=Path)
    ap.add_argument("--balanced-out", required=True, type=Path)
    ap.add_argument("--winsor", default=0.01, type=float)
    ns = ap.parse_args(argv)
    return Args(
        prices=ns.prices,
        returns_out=ns.returns_out,
        balanced_out=ns.balanced_out,
        winsor=float(ns.winsor),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prices = _load_prices(args.prices)
    returns = _to_daily_returns(prices)
    if args.winsor and args.winsor > 0.0:
        returns = _winsorize(returns, q=float(args.winsor))

    # Write daily returns
    args.returns_out.parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(args.returns_out, index=False)

    # Build balanced panel and write parquet (daily rows for kept weeks)
    balanced_rows, kept_weeks, tickers = _balanced_weekday_panel(returns)
    args.balanced_out.parent.mkdir(parents=True, exist_ok=True)
    balanced_rows.to_parquet(args.balanced_out, index=False)

    dropped = int(returns["date"].dt.to_period("W-FRI").nunique()) - kept_weeks
    print(
        f"Balanced weeks: kept={kept_weeks}, dropped={dropped}, tickers={tickers}. "
        "Each kept week has exactly 5 trading days."
    )


if __name__ == "__main__":  # pragma: no cover - CLI surface
    main()

