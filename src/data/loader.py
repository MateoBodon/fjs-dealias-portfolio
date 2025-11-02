"""
Daily data loader for replicated group estimation.

This module ingests daily equity returns, enforces a balanced universe, and
applies light cross-sectional winsorisation to guard against outliers ahead of
robust edge estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DailyLoaderConfig",
    "DailyPanel",
    "load_daily_panel",
]

_LONG_COLUMNS = ("date", "ticker", "ret")


@dataclass(frozen=True)
class DailyLoaderConfig:
    """Configuration knobs for the daily loader."""

    winsor_lower: float = 0.005
    winsor_upper: float = 0.995
    min_history: int = 252
    forward_fill: bool = False
    required_symbols: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.winsor_lower < self.winsor_upper <= 1.0):
            raise ValueError("winsor_lower and winsor_upper must satisfy 0 <= lower < upper <= 1.")
        if self.min_history <= 0:
            raise ValueError("min_history must be positive.")


@dataclass(frozen=True)
class DailyPanel:
    """Container for the balanced daily panel data."""

    returns: pd.DataFrame
    meta: dict[str, object]


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for daily loader: {suffix}")


def _to_long_frame(source: str | Path | Iterable[Path] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, (str, Path)):
        frame = _read_frame(Path(source))
    else:
        paths = list(source)
        if not paths:
            raise ValueError("Iterable source provided to load_daily_panel is empty.")
        frame = pd.concat((_read_frame(Path(p)) for p in paths), ignore_index=True)

    if set(_LONG_COLUMNS).issubset(frame.columns):
        long = frame.loc[:, _LONG_COLUMNS].copy()
    elif isinstance(frame.index, pd.DatetimeIndex):
        long = (
            frame.copy()
            .stack(dropna=False)
            .reset_index()
            .rename(columns={"level_0": "date", "level_1": "ticker", 0: "ret"})
        )
    else:
        raise ValueError("Input must contain columns (date, ticker, ret) or a DatetimeIndex.")

    long["date"] = pd.to_datetime(long["date"], utc=False)
    long = long.dropna(subset=["date", "ticker"]).copy()
    long["ticker"] = long["ticker"].astype(str)
    long = long.dropna(subset=["ret"])
    long["ret"] = long["ret"].astype(float)
    long = long.sort_values(["date", "ticker"])
    return long


def _winsorise_cross_section(df: pd.DataFrame, lower_q: float, upper_q: float) -> pd.DataFrame:
    values = df.to_numpy(dtype=np.float64, copy=True)
    if values.size == 0:
        return df.copy()
    lower = np.quantile(values, lower_q, axis=1, keepdims=True, method="linear")
    upper = np.quantile(values, upper_q, axis=1, keepdims=True, method="linear")
    clipped = np.clip(values, lower, upper)
    return pd.DataFrame(clipped, index=df.index.copy(), columns=df.columns.copy())


def _select_common_symbols(
    panel: pd.DataFrame,
    *,
    min_history: int,
    required_symbols: Sequence[str] | None,
) -> pd.DataFrame:
    if panel.empty:
        raise ValueError("No rows available to build a balanced panel.")

    candidate = panel.dropna(axis=1, how="all")
    if candidate.empty:
        raise ValueError("All symbols were empty after ingesting daily returns.")

    coverage = candidate.notna().sum(axis=0)
    eligible = coverage[coverage >= min_history].index.tolist()
    if not eligible:
        raise ValueError("No symbols meet the minimum history requirement.")

    if required_symbols:
        missing = sorted(set(required_symbols) - set(candidate.columns))
        if missing:
            raise ValueError(f"Required symbols missing from input: {', '.join(missing)}")
        eligible = sorted(set(eligible).intersection(required_symbols))
        if not eligible:
            raise ValueError("Required symbols do not meet the minimum history requirement.")

    candidate = candidate.loc[:, eligible]

    mask = candidate.notna()
    complete_rows = mask.all(axis=1)
    if complete_rows.sum() >= min_history:
        balanced = candidate.loc[complete_rows]
        balanced = balanced.dropna(axis=1, how="any")
        if balanced.shape[1] > 0:
            return balanced

    # Fallback: iteratively keep the most-covered symbols until rows are balanced.
    coverage_sorted = coverage.loc[eligible].sort_values(ascending=False)
    if coverage_sorted.empty:
        raise ValueError("Unable to find symbols with consistent coverage.")
    for count in range(len(coverage_sorted), 0, -1):
        subset_cols = coverage_sorted.index[:count].tolist()
        submask = mask.loc[:, subset_cols]
        complete_rows = submask.all(axis=1)
        if complete_rows.sum() < min_history:
            continue
        balanced = candidate.loc[complete_rows, subset_cols].dropna(axis=1, how="any")
        if balanced.empty:
            continue
        if balanced.shape[0] >= min_history and not balanced.isna().any().any():
            return balanced

    raise ValueError("Unable to construct a balanced universe from the supplied data.")


def load_daily_panel(
    source: str | Path | Iterable[Path] | pd.DataFrame,
    *,
    config: DailyLoaderConfig | None = None,
) -> DailyPanel:
    """
    Load and clean daily return data, enforcing a balanced universe.
    """

    cfg = config or DailyLoaderConfig()
    long = _to_long_frame(source)
    if long.empty:
        raise ValueError("Daily returns input is empty.")

    panel = (
        long.pivot(index="date", columns="ticker", values="ret")
        .sort_index()
        .dropna(axis=0, how="all")
    )

    if cfg.forward_fill:
        panel = panel.ffill()
    balanced = _select_common_symbols(panel, min_history=cfg.min_history, required_symbols=cfg.required_symbols)
    balanced = balanced.sort_index()
    if balanced.shape[0] < cfg.min_history:
        raise ValueError("Insufficient history after enforcing balanced universe.")

    winsorised = _winsorise_cross_section(balanced, cfg.winsor_lower, cfg.winsor_upper)
    winsorised.index.name = "date"

    meta: dict[str, object] = {
        "symbols": list(winsorised.columns),
        "start": winsorised.index.min(),
        "end": winsorised.index.max(),
        "n_days": int(winsorised.shape[0]),
        "p": int(winsorised.shape[1]),
        "winsor": (cfg.winsor_lower, cfg.winsor_upper),
    }

    return DailyPanel(returns=winsorised, meta=meta)
