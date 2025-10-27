from __future__ import annotations

import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.mark.skipif(
    not os.getenv("NASDAQ_DATA_LINK_API_KEY"),
    reason="NASDAQ_DATA_LINK_API_KEY not set; skipping Sharadar smoke.",
)
@pytest.mark.slow
def test_sharadar_fetch_and_balance_smoke(tmp_path: Path) -> None:
    prices_csv = tmp_path / "prices.csv"
    returns_csv = tmp_path / "returns.csv"
    balanced_parquet = tmp_path / "balanced.parquet"

    # Small universe and narrow window for speed
    cmd_fetch = [
        sys.executable,
        "scripts/data/fetch_sharadar.py",
        "--start",
        "2020-01-15",
        "--end",
        "2020-03-15",
        "--pre-start",
        "2019-01-01",
        "--pre-end",
        "2019-12-31",
        "--p",
        "5",
        "--min-price",
        "5",
        "--out",
        str(prices_csv),
    ]
    subprocess.run(cmd_fetch, check=True)
    assert prices_csv.exists()

    cmd_bal = [
        sys.executable,
        "scripts/data/make_balanced_weekly.py",
        "--prices",
        str(prices_csv),
        "--returns-out",
        str(returns_csv),
        "--balanced-out",
        str(balanced_parquet),
        "--winsor",
        "0.0",
    ]
    subprocess.run(cmd_bal, check=True)
    assert returns_csv.exists()

    # Verify at least one full week with exactly 5 business days and >=5 tickers in intersection
    df = pd.read_csv(returns_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.dropna(subset=["ret"])  # type: ignore[list-item]
    periods = df["date"].dt.to_period("W-FRI")
    # Robust start-of-week computation across pandas versions
    week_start = (periods.dt.start_time - pd.Timedelta(days=4)).dt.normalize()
    df["week_start"] = week_start
    week_counts = df.groupby("week_start")["date"].nunique()
    kept_weeks = week_counts[week_counts == 5].index
    assert len(kept_weeks) >= 1
    sub = df[df["week_start"].isin(kept_weeks)].copy()
    per = sub.groupby(["week_start", "ticker"]).size().reset_index(name="n")
    complete = per[per["n"] == 5]
    by_week = complete.groupby("week_start")["ticker"].apply(lambda s: set(s))
    if by_week.empty:
        pytest.skip("No complete week found after intersection; data gap")
    universe = set.intersection(*by_week.tolist())
    assert len(universe) >= 5
    # Check each kept week has exactly 5 days for each ticker in the intersection
    sub2 = sub[sub["ticker"].isin(list(universe))]
    counts = sub2.groupby(["week_start", "ticker"]).size()
    assert int(counts.min()) == 5
