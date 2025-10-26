from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finance.io import load_returns_csv
from finance.returns import balance_weeks


def _summary_stats(returns_path: Path) -> dict[str, Any]:
    raw = pd.read_csv(returns_path)
    duplicates = int(raw.duplicated(subset=["date", "ticker"]).sum())
    unique_pairs = int(raw.drop_duplicates(subset=["date", "ticker"]).shape[0])

    daily = load_returns_csv(returns_path)
    daily.sort_index(inplace=True)
    dates = daily.index.to_series()
    tickers = daily.columns.astype(str).tolist()

    stats: dict[str, Any] = {
        "returns_csv": str(returns_path),
        "rows_raw": int(raw.shape[0]),
        "duplicates_dropped": duplicates,
        "unique_pairs": unique_pairs,
        "n_days": int(daily.shape[0]),
        "n_tickers": int(daily.shape[1]),
        "date_start": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
        "date_end": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
    }

    # Balanced weekly panel
    y_matrix, groups, week_index = balance_weeks(daily)
    replicates = int(np.bincount(groups).min()) if groups.size else 0
    stats.update(
        {
            "balanced_weeks": int(len(week_index)),
            "replicates_per_week": replicates,
            "balanced_assets": int(y_matrix.shape[1]),
            "week_start": week_index.min().strftime("%Y-%m-%d")
            if len(week_index)
            else None,
            "week_end": week_index.max().strftime("%Y-%m-%d")
            if len(week_index)
            else None,
        }
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the daily returns dataset and balanced weekly panel."
    )
    parser.add_argument(
        "--returns",
        type=Path,
        default=Path("data/returns_daily.csv"),
        help="Path to the tidy daily returns CSV (date,ticker,ret).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON (defaults to stdout).",
    )
    args = parser.parse_args()

    stats = _summary_stats(args.returns)
    if args.output is None:
        print(json.dumps(stats, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
