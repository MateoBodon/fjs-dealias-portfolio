from __future__ import annotations

"""
Fetch daily equity prices or returns from WRDS (CRSP) and write tidy CSVs.

Usage:
    python scripts/data/fetch_wrds_crsp.py \
        --start 2010-01-01 --end 2025-06-30 \
        --pre-start 2014-01-01 --pre-end 2018-12-31 \
        --p 300 --min-price 5 \
        --out data/prices_daily.csv

Credentials:
  - Supply WRDS credentials via environment or ~/.pgpass; do not hardcode.
  - The wrds client will prompt if not available; prefer non-interactive setups.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import wrds  # type: ignore
except Exception as exc:  # pragma: no cover - CLI surface
    raise ImportError("wrds package is required: pip install wrds") from exc


@dataclass(frozen=True)
class Args:
    start: str
    end: str
    pre_start: str
    pre_end: str
    p: int
    min_price: float
    out: Path


def parse_args(argv: Sequence[str] | None = None) -> Args:
    ap = argparse.ArgumentParser(description="WRDS/CRSP fetcher: prices to tidy CSV")
    ap.add_argument("--start", required=True, type=str)
    ap.add_argument("--end", required=True, type=str)
    ap.add_argument("--pre-start", required=True, type=str)
    ap.add_argument("--pre-end", required=True, type=str)
    ap.add_argument("--p", required=True, type=int, help="Top p tickers by ADV")
    ap.add_argument("--min-price", default=5.0, type=float)
    ap.add_argument("--out", required=True, type=Path)
    ns = ap.parse_args(argv)
    return Args(
        start=ns.start,
        end=ns.end,
        pre_start=ns.__dict__["pre_start"],
        pre_end=ns.__dict__["pre_end"],
        p=int(ns.p),
        min_price=float(ns.min_price),
        out=ns.out,
    )


def _sep_price_col(df: pd.DataFrame) -> str:
    for cand in ("prc", "closeadj", "close"):
        if cand in df.columns:
            return cand
    raise ValueError("Could not infer price column in CRSP export.")


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: date, ticker or permno, price, volume
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("CRSP export must include 'date'.")
    price_col = _sep_price_col(out)
    ticker_col = "ticker" if "ticker" in out.columns else ("permno" if "permno" in out.columns else None)
    if ticker_col is None:
        raise ValueError("CRSP export must include 'ticker' or 'permno'.")
    out = out.loc[:, ["date", ticker_col, price_col]].copy()
    out = out.rename(columns={ticker_col: "ticker", price_col: "price_close"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype("string")
    out["price_close"] = pd.to_numeric(out["price_close"], errors="coerce")
    out = out.dropna(subset=["price_close"]).sort_values(["date", "ticker"])  # type: ignore[list-item]
    return out


def _compute_adv(df: pd.DataFrame, *, price_col: str, vol_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp = tmp.dropna(subset=[price_col, vol_col])
    tmp["adv"] = np.abs(pd.to_numeric(tmp[price_col], errors="coerce")) * pd.to_numeric(
        tmp[vol_col], errors="coerce"
    )
    return (
        tmp.groupby("ticker", as_index=False)["adv"].mean().rename(columns={"adv": "mean_adv"})
    )


def _select_universe(pre_df: pd.DataFrame, *, min_price: float, top_p: int) -> list[str]:
    price_col = "price_close"
    median_price = (
        pre_df.groupby("ticker", as_index=False)[price_col].median().rename(columns={price_col: "median_price"})
    )
    # Approximate ADV: use volume if present; else equal weights
    vol_col = "volume" if "volume" in pre_df.columns else None
    if vol_col is not None:
        adv = _compute_adv(pre_df, price_col=price_col, vol_col=vol_col)
        merged = pre_df.loc[:, ["ticker"]].drop_duplicates().merge(median_price, on="ticker", how="inner").merge(
            adv, on="ticker", how="left"
        )
        merged["mean_adv"] = merged["mean_adv"].fillna(0.0)
    else:
        merged = pre_df.loc[:, ["ticker"]].drop_duplicates().merge(median_price, on="ticker", how="inner")
        merged["mean_adv"] = 1.0
    filtered = merged[merged["median_price"] >= float(min_price)]
    filtered = filtered.sort_values("mean_adv", ascending=False)
    return filtered["ticker"].head(int(top_p)).tolist()


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    args = parse_args(argv)
    conn = wrds.Connection()

    # Pull daily CRSP (DSF) with ticker mapping
    # Note: users may want to adapt this to DASDAQ tables if available in their WRDS account.
    query = f"""
        SELECT a.date, b.ticker, a.prc AS prc, a.vol AS volume
        FROM crsp.dsf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{args.start}' AND '{args.end}'
    """
    df_main = conn.raw_sql(query)
    if df_main is None or df_main.empty:
        raise RuntimeError("WRDS query returned no rows for main period.")
    tidy_main = _normalize_prices(df_main)

    # Pre-period for universe selection
    query_pre = f"""
        SELECT a.date, b.ticker, a.prc AS prc, a.vol AS volume
        FROM crsp.dsf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{args.pre_start}' AND '{args.pre_end}'
    """
    df_pre = conn.raw_sql(query_pre)
    if df_pre is None or df_pre.empty:
        raise RuntimeError("WRDS query returned no rows for pre period.")
    tidy_pre = _normalize_prices(df_pre)

    universe = _select_universe(tidy_pre, min_price=args.min_price, top_p=args.p)
    filtered = tidy_main[tidy_main["ticker"].isin(universe)].copy()
    filtered.to_csv(args.out, index=False)
    print(f"[WRDS] Wrote tidy prices CSV to {args.out} with {len(universe)} tickers")


if __name__ == "__main__":  # pragma: no cover - CLI surface
    main()


