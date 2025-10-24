from __future__ import annotations

"""
Sharadar data fetcher: build a liquid equity universe and export tidy daily prices.

Usage:
    python scripts/data/fetch_sharadar.py \
        --start 2010-01-01 --end 2025-06-30 \
        --pre-start 2014-01-01 --pre-end 2018-12-31 \
        --p 300 --min-price 5 --out data/prices_daily.csv

Environment:
    NASDAQ_DATA_LINK_API_KEY must be set (read from env; do not hardcode).

Output schema (CSV):
    date,ticker,adj_close,volume
"""

import argparse
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

try:
    import nasdaqdatalink as ndl  # type: ignore
except Exception as exc:  # pragma: no cover - dependency error
    raise ImportError(
        "nasdaq-data-link is required. Please install it via pip."
    ) from exc


DATA_DIR = Path("data")


def _require_api_key() -> str:
    key = os.getenv("NASDAQ_DATA_LINK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing NASDAQ_DATA_LINK_API_KEY env var. Create .env and export it."
        )
    return key


def _get_table_paginated(table: str, params: dict[str, object]) -> pd.DataFrame:
    """Call get_table with pagination and return a DataFrame.

    This avoids SSL idiosyncrasies with export_table in constrained environments.
    """
    df = ndl.get_table(table, paginate=True, **params)
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    return df


def _chunked(seq: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _fetch_tickers() -> pd.DataFrame:
    """Return SHARADAR/TICKERS filtered to domestic common shares (table=SEP)."""
    df = ndl.get_table(
        "SHARADAR/TICKERS",
        table="SEP",
        paginate=True,
    )
    if df is None or df.empty:
        raise RuntimeError("No rows returned from SHARADAR/TICKERS.")
    # Normalize cases and filter by category
    df = df.copy()
    # Columns of interest: ticker, permaticker, category, exchange, isdelisted
    required = {"ticker", "permaticker", "category"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"TICKERS missing required columns: {sorted(missing)}; got {df.columns.tolist()}"
        )
    mask_domestic = df["category"].astype(str).str.contains(
        "Domestic Common Stock", case=False, na=False
    )
    kept = df.loc[mask_domestic, ["ticker", "permaticker", "category"]].drop_duplicates()
    if kept.empty:
        raise RuntimeError("No domestic common shares after filtering TICKERS.")
    return kept


def _compute_adv(
    sep_df: pd.DataFrame, *, price_col: str, vol_col: str = "volume"
) -> pd.DataFrame:
    """Compute mean ADV per ticker using |price|*volume in the provided frame."""
    df = sep_df.copy()
    if price_col not in df.columns:
        raise ValueError(f"Missing price column '{price_col}' in SEP data.")
    if vol_col not in df.columns:
        raise ValueError(f"Missing volume column '{vol_col}' in SEP data.")
    df = df.dropna(subset=[price_col, vol_col])
    df["adv"] = np.abs(pd.to_numeric(df[price_col], errors="coerce")) * pd.to_numeric(
        df[vol_col], errors="coerce"
    )
    adv = (
        df.groupby("ticker", as_index=False)["adv"].mean().rename(columns={"adv": "mean_adv"})
    )
    return adv


def _select_universe(
    tickers_meta: pd.DataFrame,
    pre_sep: pd.DataFrame,
    *,
    min_price: float,
    top_p: int,
) -> list[str]:
    price_col = "closeadj" if "closeadj" in pre_sep.columns else "close"
    # Compute per-ticker median price over pre-period
    median_price = (
        pre_sep.groupby("ticker", as_index=False)[price_col]
        .median()
        .rename(columns={price_col: "median_price"})
    )
    adv = _compute_adv(pre_sep, price_col=price_col)
    merged = tickers_meta.merge(adv, on="ticker", how="inner").merge(
        median_price, on="ticker", how="inner"
    )
    filtered = merged[merged["median_price"] >= float(min_price)]
    if filtered.empty:
        raise RuntimeError("No tickers pass the min_price filter in pre-period.")
    filtered = filtered.sort_values("mean_adv", ascending=False)
    universe = filtered["ticker"].head(int(top_p)).tolist()
    if len(universe) < min(10, top_p):
        raise RuntimeError(
            f"Selected too few tickers ({len(universe)}) after filters; relax constraints."
        )
    return universe


def _fetch_sep_for(
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    columns: Sequence[str] | None = None,
    batch: int = 300,
) -> pd.DataFrame:
    """Export SEP for provided tickers and date range in batches; concat to a DataFrame."""
    params_base: dict[str, object] = {
        "date.gte": start,
        "date.lte": end,
    }
    if columns:
        params_base["qopts"] = {"columns": ",".join(columns)}

    frames: list[pd.DataFrame] = []
    for chunk in _chunked(list(tickers), batch):
        params = dict(params_base)
        params["ticker"] = ",".join(chunk)
        df = _get_table_paginated("SHARADAR/SEP", params)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw SEP into tidy columns: date,ticker,adj_close,volume."""
    required_any = {"ticker", "date", "volume"}
    if not required_any.issubset(df.columns):
        raise ValueError(f"SEP missing columns: {required_any - set(df.columns)}")
    price_col = "closeadj" if "closeadj" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        raise ValueError("Neither 'closeadj' nor 'close' present in SEP export.")
    out = df.loc[:, ["date", "ticker", price_col, "volume"]].copy()
    out = out.rename(columns={price_col: "adj_close"})
    # Ensure dtypes
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype("string")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["adj_close", "volume"]).sort_values(["ticker", "date"])  # type: ignore[list-item]
    return out


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
    ap = argparse.ArgumentParser(description="Sharadar fetcher: prices to tidy CSV")
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    key = _require_api_key()
    ndl.ApiConfig.api_key = key  # configure client
    # Optional: allow insecure SSL for environments with broken root CAs
    if os.getenv("ALLOW_INSECURE_SSL", "0") == "1":  # pragma: no cover - unsafe path
        try:
            ndl.ApiConfig.verify_ssl = False  # type: ignore[attr-defined]
            print("[Sharadar] WARNING: SSL verification disabled (ALLOW_INSECURE_SSL=1)")
        except Exception:
            pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[Sharadar] Fetching TICKERS (domestic common shares)...", flush=True)
    tickers_meta = _fetch_tickers()
    print(f"[Sharadar] Candidate tickers: {len(tickers_meta)}")

    print("[Sharadar] Exporting SEP pre-period for ADV ranking...", flush=True)
    pre_cols = ["ticker", "date", "closeadj", "close", "volume"]
    pre_df = _fetch_sep_for(
        tickers_meta["ticker"].tolist(),
        start=args.pre_start,
        end=args.pre_end,
        columns=pre_cols,
        batch=300,
    )
    if pre_df.empty:
        raise RuntimeError("Pre-period SEP export returned no data.")
    print(f"[Sharadar] Pre-period rows: {len(pre_df):,}")

    universe = _select_universe(
        tickers_meta,
        pre_df,
        min_price=args.min_price,
        top_p=args.p,
    )
    print(f"[Sharadar] Selected universe size: {len(universe)}")

    print("[Sharadar] Exporting SEP daily prices for selected universe...", flush=True)
    main_cols = ["ticker", "date", "closeadj", "close", "volume"]
    main_df = _fetch_sep_for(
        universe,
        start=args.start,
        end=args.end,
        columns=main_cols,
        batch=200,
    )
    if main_df.empty:
        raise RuntimeError("Main-period SEP export returned no data for selected universe.")
    print(f"[Sharadar] Main-period rows: {len(main_df):,}")

    tidy = _normalize_prices(main_df)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_path, index=False)
    print(f"[Sharadar] Wrote tidy daily prices to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover - CLI surface
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
