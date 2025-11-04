from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.io.crsp_daily import (
    CrspQueryParams,
    build_dow_vol_labels,
    explain_rowcount,
    fetch_crsp_daily_snapshot,
    write_labels_parquet,
)
from src.io.wrds_connect import wrds_conn


def _parse_int_list(values: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CRSP daily returns + labels from WRDS.")
    parser.add_argument("--start", type=str, default="2016-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--min-volume", type=int, default=0)
    parser.add_argument("--exchanges", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--share-codes", nargs="+", default=["10", "11"])
    parser.add_argument("--returns-out", type=Path, default=Path("data/wrds/returns_daily.parquet"))
    parser.add_argument("--labels-out", type=Path, default=Path("data/wrds/labels.parquet"))
    parser.add_argument("--ewma-span", type=int, default=21)
    parser.add_argument("--calm-quantile", type=float, default=0.2)
    parser.add_argument("--crisis-quantile", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def dry_run(params: CrspQueryParams, *, returns_out: Path, labels_out: Path) -> None:
    sql = params.render_sql()
    with wrds_conn() as connection:
        est_rows = explain_rowcount(sql, connection=connection)
    print(f"[DRY-RUN] CRSP daily snapshot query ({params.start} → {params.end})")
    print(sql)
    print(f"[DRY-RUN] Estimated rows: {est_rows:,}")
    print(f"[DRY-RUN] Returns parquet: {returns_out}")
    print(f"[DRY-RUN] Labels parquet: {labels_out}")


def run(params: CrspQueryParams, *, returns_out: Path, labels_out: Path, ewma_span: int, calm_q: float, crisis_q: float) -> None:
    snapshot = fetch_crsp_daily_snapshot(returns_out, params=params)
    labels = build_dow_vol_labels(
        snapshot,
        ewma_span=ewma_span,
        calm_quantile=calm_q,
        crisis_quantile=crisis_q,
    )
    write_labels_parquet(labels, labels_out)
    print(f"[WRDS] Wrote returns parquet: {returns_out} ({len(snapshot):,} rows, {snapshot['ticker'].nunique()} tickers)")
    print(f"[WRDS] Wrote labels parquet: {labels_out} ({len(labels):,} business days)")


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI surface
    args = parse_args(argv)
    params = CrspQueryParams(
        start=pd.Timestamp(args.start).date(),
        end=pd.Timestamp(args.end).date(),
        exchanges=_parse_int_list(args.exchanges),
        share_codes=_parse_int_list(args.share_codes),
        min_price=float(args.min_price),
        min_volume=int(args.min_volume),
    )
    if args.dry_run:
        dry_run(params, returns_out=args.returns_out, labels_out=args.labels_out)
    else:
        run(
            params,
            returns_out=args.returns_out,
            labels_out=args.labels_out,
            ewma_span=int(args.ewma_span),
            calm_q=float(args.calm_quantile),
            crisis_q=float(args.crisis_quantile),
        )


if __name__ == "__main__":  # pragma: no cover - CLI surface
    main()
