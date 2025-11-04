from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import psycopg2

from src.io.wrds_connect import wrds_conn

DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date(2024, 12, 31)
DEFAULT_EXCHANGES: tuple[int, ...] = (1, 2, 3)
DEFAULT_SHARE_CODES: tuple[int, ...] = (10, 11)


@dataclass(frozen=True)
class CrspQueryParams:
    """Configuration for the CRSP daily snapshot."""

    start: date = DEFAULT_START
    end: date = DEFAULT_END
    exchanges: tuple[int, ...] = DEFAULT_EXCHANGES
    share_codes: tuple[int, ...] = DEFAULT_SHARE_CODES
    min_price: float = 1.0
    min_volume: int = 0

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)

    def format_lists(self, values: Sequence[int]) -> str:
        return ", ".join(str(int(v)) for v in values)

    def render_sql(self) -> str:
        exchange_filter = self.format_lists(self.exchanges)
        shrcd_filter = self.format_lists(self.share_codes)
        return f"""
SELECT
    d.permno,
    d.date,
    COALESCE(n.ticker, '') AS ticker,
    d.ret,
    d.prc,
    d.vol,
    d.shrout,
    d.cfacpr,
    d.cfacshr,
    COALESCE(n.exchcd, d.hexcd) AS exchcd,
    n.shrcd
FROM crsp.dsf AS d
LEFT JOIN LATERAL (
    SELECT ticker, exchcd, shrcd
    FROM crsp.dsenames AS n1
    WHERE n1.permno = d.permno
      AND n1.namedt <= d.date
      AND (n1.nameendt >= d.date OR n1.nameendt = DATE '9999-12-31')
      AND n1.ticker IS NOT NULL
    ORDER BY n1.namedt DESC
    LIMIT 1
) AS n ON TRUE
WHERE d.date BETWEEN DATE '{self.start.isoformat()}' AND DATE '{self.end.isoformat()}'
  AND COALESCE(n.exchcd, d.hexcd) IN ({exchange_filter})
  AND COALESCE(n.shrcd, -1) IN ({shrcd_filter})
  AND d.ret IS NOT NULL
  AND ABS(COALESCE(d.prc, 0.0)) >= {self.min_price}
  AND COALESCE(d.vol, 0) >= {self.min_volume}
ORDER BY d.date ASC, d.permno ASC;
""".strip()


def explain_rowcount(sql: str, *, connection: psycopg2.extensions.connection) -> int:
    """Return the planner's estimated rowcount for the provided query."""

    with connection.cursor() as cur:
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        result = cur.fetchone()
        if not result:
            return 0
        payload = result[0]
        if isinstance(payload, (bytes, str)):
            plan_root = json.loads(payload)[0]
        elif isinstance(payload, list) and payload:
            plan_root = payload[0]
        elif isinstance(payload, dict):
            plan_root = payload
        else:
            return 0
        plan = plan_root.get("Plan", {})
        return int(plan.get("Plan Rows", 0))


def _clean_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned = cleaned.sort_values(["date", "permno"]).reset_index(drop=True)

    numeric_cols = ["ret", "prc", "vol", "shrout", "cfacpr", "cfacshr"]
    for column in numeric_cols:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["ret"] = cleaned["ret"].replace({-66.0: np.nan, -77.0: np.nan, -88.0: np.nan})
    cleaned = cleaned.dropna(subset=["ret", "ticker"])
    cleaned["ticker"] = cleaned["ticker"].str.upper()
    cleaned["abs_prc"] = cleaned["prc"].abs()
    # CRSP stores shrout as number of shares / 1000
    cleaned["shares_outstanding"] = cleaned["shrout"] * cleaned["cfacshr"] * 1000.0
    cleaned["market_cap"] = cleaned["abs_prc"] * cleaned["shares_outstanding"]

    columns = [
        "date",
        "permno",
        "ticker",
        "ret",
        "abs_prc",
        "vol",
        "shares_outstanding",
        "market_cap",
        "exchcd",
        "shrcd",
    ]
    cleaned = cleaned.loc[:, columns]
    cleaned = cleaned.rename(
        columns={
            "abs_prc": "price",
            "vol": "volume",
            "shares_outstanding": "shares_out",
        }
    )
    return cleaned


def fetch_crsp_daily_snapshot(
    out_path: Path,
    *,
    params: CrspQueryParams | None = None,
) -> pd.DataFrame:
    """Fetch CRSP daily snapshot and persist to parquet."""

    query_params = params or CrspQueryParams()
    sql = query_params.render_sql()
    with wrds_conn() as connection:
        frame = pd.read_sql(sql, connection)
    cleaned = _clean_snapshot(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_path, index=False)
    return cleaned


def build_dow_vol_labels(
    returns: pd.DataFrame,
    *,
    ewma_span: int = 21,
    calm_quantile: float = 0.2,
    crisis_quantile: float = 0.8,
) -> pd.DataFrame:
    """Compute day-of-week and volatility-state labels."""

    if returns.empty:
        raise ValueError("Empty returns frame.")
    grouped = returns.groupby("date")["ret"].agg(lambda series: float(np.nanmean(series.astype(float) ** 2)))
    grouped = grouped.sort_index()
    vol_signal = np.sqrt(
        grouped.ewm(span=max(ewma_span, 2), adjust=False, min_periods=max(5, int(ewma_span / 2))).mean()
    )
    vol_signal = vol_signal.dropna()
    calm_cut = float(vol_signal.quantile(calm_quantile))
    crisis_cut = float(vol_signal.quantile(crisis_quantile))

    labels = pd.DataFrame(
        {
            "date": vol_signal.index,
            "vol_signal": vol_signal.to_numpy(dtype=np.float64),
        }
    )
    labels["dow"] = labels["date"].dt.dayofweek.astype(np.int8)
    labels["dow_label"] = labels["dow"].map({0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"})

    def _state(value: float) -> str:
        if not np.isfinite(value):
            return "unknown"
        if value <= calm_cut:
            return "calm"
        if value >= crisis_cut:
            return "crisis"
        return "mid"

    labels["vol_state"] = [ _state(val) for val in labels["vol_signal"].to_numpy(dtype=np.float64) ]
    labels["calm_threshold"] = calm_cut
    labels["crisis_threshold"] = crisis_cut
    labels["ewma_span"] = int(ewma_span)

    # drop weekends
    labels = labels[labels["dow"] < 5].reset_index(drop=True)
    return labels


def write_labels_parquet(labels: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out_path, index=False)
