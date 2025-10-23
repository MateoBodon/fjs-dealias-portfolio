from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance.io import load_prices_csv, to_daily_returns


def _balanced_weekly_from_daily(
    daily_returns: pd.DataFrame, replicates: int = 5
) -> tuple[pd.DataFrame, int, list[str]]:
    """Return a balanced weekly panel (Mon–Fri) and counters.

    - Drops partial weeks (requires exactly ``replicates`` daily rows per week).
    - Intersects tickers across all kept weeks to enforce a fixed universe.
    - Sums daily log returns within each week to form weekly log returns.

    Returns the weekly panel, number of dropped weeks, and ordered tickers.
    """

    if daily_returns.index.inferred_type != "datetime64":
        raise ValueError("daily_returns must use a DatetimeIndex.")

    # Group by weeks starting on Monday
    panel = daily_returns.sort_index()
    grouped = panel.groupby(panel.index.to_period("W-MON"))

    total_weeks = 0
    week_frames: list[pd.DataFrame] = []
    week_labels: list[pd.Timestamp] = []

    for period, frame in grouped:
        total_weeks += 1
        # Clean per-week: remove all-NaN cols, any-NaN rows; require full 5 days
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

    # Enforce a fixed universe across all kept weeks
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


def _resolve_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common adjusted-close column names to the expected schema.

    Accepts `adj_close`/`adj_close_price` by renaming to `price_close` if
    present. Otherwise assumes the CSV already uses `price_close`.
    """

    if "price_close" in df.columns:
        return df
    rename_map: dict[str, str] = {}
    for name in ("adj_close", "adj_close_price", "adjusted_close"):
        if name in df.columns:
            rename_map[name] = "price_close"
            break
    if rename_map:
        new_df = df.rename(columns=rename_map)
        return new_df
    return df


def run(
    *,
    input_csv: Path,
    output_csv: Path,
    output_meta: Path | None,
    start: str | None,
    end: str | None,
    p_target: int,
    dry_run: bool,
) -> dict[str, Any]:
    # Load and normalise the pricing CSV
    raw = pd.read_csv(input_csv)
    raw = _resolve_price_columns(raw)
    tmp = input_csv.with_suffix(".tmp.prices.csv")
    try:
        raw.to_csv(tmp, index=False)
        prices = load_prices_csv(str(tmp))
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]

    # Optional date window
    if start is not None:
        prices = prices.loc[prices["date"] >= pd.to_datetime(start)]
    if end is not None:
        prices = prices.loc[prices["date"] <= pd.to_datetime(end)]

    if prices.empty:
        raise ValueError("No price rows after applying date filters.")

    # Daily log returns (wide)
    daily = to_daily_returns(prices)

    # Balanced weeks and fixed universe across all weeks
    weekly, dropped_weeks, ordered = _balanced_weekly_from_daily(daily)

    # Choose a fixed universe of size ~p_target (deterministic ordering)
    tickers = ordered[:p_target]
    weekly_p = weekly.loc[:, tickers]

    # Prepare metadata
    meta: dict[str, Any] = {
        "source": str(input_csv),
        "start_date": (daily.index.min()).strftime("%Y-%m-%d"),
        "end_date": (daily.index.max()).strftime("%Y-%m-%d"),
        "weeks_start": weekly_p.index.min().strftime("%Y-%m-%d"),
        "weeks_end": weekly_p.index.max().strftime("%Y-%m-%d"),
        "frequency": "weekly_mon-fri_sum",
        "balanced_weeks": int(weekly_p.shape[0]),
        "dropped_weeks": int(dropped_weeks),
        "p": int(weekly_p.shape[1]),
        "universe": tickers,
    }

    # Logging summary
    print(
        f"Weekly dataset: weeks={meta['balanced_weeks']}, dropped={meta['dropped_weeks']}, p={meta['p']}"
    )

    # Dry-run: only report stats
    if dry_run:
        return meta

    # Write outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    weekly_p.to_csv(output_csv, index=True)

    if output_meta is None:
        output_meta = output_csv.with_suffix(".meta.json")
    with open(output_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a weekly (Mon–Fri) balanced equity dataset with a fixed universe."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prices_sample.csv"),
        help="Path to daily adjusted prices CSV (date,ticker,price_close)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/prices_weekly_200.csv"),
        help="Output path for the wide weekly returns CSV",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=None,
        help="Optional path for metadata JSON (defaults to CSV with .meta.json)",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "-p",
        "--p-target",
        type=int,
        default=200,
        help="Target universe size (assets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report balance stats without writing outputs",
    )

    args = parser.parse_args()
    run(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_meta=args.output_meta,
        start=args.start,
        end=args.end,
        p_target=args.p_target,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

