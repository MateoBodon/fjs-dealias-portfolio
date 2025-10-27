from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Literal, Tuple

import numpy as np
import pandas as pd

PartialWeekPolicy = Literal["drop", "impute"]


@dataclass(frozen=True)
class PanelManifest:
    """Metadata describing a balanced Week×Day panel."""

    asset_count: int
    weeks: int
    days_per_week: int
    dropped_weeks: int
    imputed_weeks: int
    partial_week_policy: PartialWeekPolicy
    start_week: str
    end_week: str
    data_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "asset_count": int(self.asset_count),
            "weeks": int(self.weeks),
            "days_per_week": int(self.days_per_week),
            "dropped_weeks": int(self.dropped_weeks),
            "imputed_weeks": int(self.imputed_weeks),
            "partial_week_policy": self.partial_week_policy,
            "start_week": self.start_week,
            "end_week": self.end_week,
            "data_hash": self.data_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PanelManifest":
        return cls(
            asset_count=int(payload["asset_count"]),
            weeks=int(payload["weeks"]),
            days_per_week=int(payload["days_per_week"]),
            dropped_weeks=int(payload.get("dropped_weeks", 0)),
            imputed_weeks=int(payload.get("imputed_weeks", 0)),
            partial_week_policy=str(payload.get("partial_week_policy", "drop")),
            start_week=str(payload.get("start_week", "")),
            end_week=str(payload.get("end_week", "")),
            data_hash=str(payload.get("data_hash", "")),
        )


@dataclass
class BalancedPanel:
    """Balanced weekly aggregate with supporting daily blocks."""

    weekly: pd.DataFrame
    week_map: Dict[pd.Timestamp, pd.DataFrame]
    replicates: int
    ordered_tickers: list[str]
    dropped_weeks: int
    imputed_weeks: int
    manifest: PanelManifest


def hash_daily_returns(frame: pd.DataFrame) -> str:
    """Stable hash for a wide daily returns frame."""

    if frame.empty:
        return "empty"
    ordered = frame.sort_index().sort_index(axis=1)
    values = ordered.to_numpy(dtype=np.float64, copy=True)
    mask = np.isnan(values)
    values[mask] = 0.0
    hasher = hashlib.sha256()
    hasher.update(values.tobytes())
    hasher.update(mask.tobytes())
    index_bytes = ordered.index.astype("int64", copy=False).to_numpy(dtype=np.int64).tobytes()
    hasher.update(index_bytes)
    col_bytes = "|".join(map(str, ordered.columns)).encode("utf-8")
    hasher.update(col_bytes)
    return hasher.hexdigest()


def _expected_week_index(start: pd.Timestamp, days_per_week: int) -> pd.DatetimeIndex:
    """Business-day sequence for a Monday-aligned week."""

    return pd.date_range(start=start, periods=days_per_week, freq="B")


def build_balanced_weekday_panel(
    daily_returns: pd.DataFrame,
    *,
    days_per_week: int = 5,
    partial_week_policy: PartialWeekPolicy = "drop",
    impute_fill_value: float = 0.0,
) -> BalancedPanel:
    """Construct a balanced Week×Day panel from daily returns."""

    if daily_returns.index.inferred_type != "datetime64":
        raise ValueError("daily_returns must use a DatetimeIndex.")

    panel = daily_returns.copy()
    panel = panel.sort_index()
    panel = panel[~panel.index.duplicated(keep="first")]
    panel_index = panel.index

    total_weeks = 0
    dropped_weeks = 0
    imputed_weeks = 0
    week_frames: list[pd.DataFrame] = []
    week_labels: list[pd.Timestamp] = []

    grouped: Iterator[Tuple[pd.Period, pd.DataFrame]] = panel.groupby(
        panel_index.to_period("W-MON")
    )
    for period, frame in grouped:
        total_weeks += 1
        frame = frame.dropna(axis=1, how="all").sort_index()
        if frame.empty:
            dropped_weeks += 1
            continue

        if partial_week_policy == "drop":
            if frame.shape[0] < days_per_week:
                dropped_weeks += 1
                continue
            trimmed = frame.iloc[:days_per_week]
            trimmed = trimmed.dropna(axis=1, how="any")
            if trimmed.shape[0] != days_per_week or trimmed.shape[1] == 0:
                dropped_weeks += 1
                continue
            week_frames.append(trimmed)
        else:  # impute
            expected_index = _expected_week_index(period.start_time, days_per_week)
            reindexed = frame.reindex(expected_index)
            # Drop tickers that never trade during the week
            reindexed = reindexed.dropna(axis=1, how="all")
            if reindexed.empty:
                dropped_weeks += 1
                continue
            had_missing = reindexed.isna().any().any()
            if had_missing:
                reindexed = reindexed.fillna(impute_fill_value)
                imputed_weeks += 1
            reindexed = reindexed.astype(np.float64)
            if reindexed.shape[0] != days_per_week:
                # Reindex again in case the week had fewer business days (holidays)
                reindexed = reindexed.reindex(expected_index, fill_value=impute_fill_value)
            week_frames.append(reindexed)

        week_labels.append(period.start_time)

    if not week_frames:
        raise ValueError("No balanced weeks available for evaluation.")

    common_tickers = set(week_frames[0].columns)
    for frame in week_frames[1:]:
        common_tickers &= set(frame.columns)
    if not common_tickers:
        raise ValueError("No common tickers across balanced weeks.")

    ordered_tickers = sorted(common_tickers)
    replicate_count = int(week_frames[0].shape[0])
    if any(frame.shape[0] != replicate_count for frame in week_frames):
        raise ValueError("Replicate count varies across balanced weeks.")

    weekly_arrays = [
        frame.loc[:, ordered_tickers].to_numpy(dtype=np.float64) for frame in week_frames
    ]
    weekly_data = np.stack([block.sum(axis=0) for block in weekly_arrays], axis=0)
    weekly_df = pd.DataFrame(
        weekly_data,
        index=pd.Index(week_labels, name="week_start"),
        columns=ordered_tickers,
    )
    week_map = {
        week_labels[idx]: week_frames[idx].copy() for idx in range(len(week_labels))
    }
    manifest = PanelManifest(
        asset_count=len(ordered_tickers),
        weeks=len(week_labels),
        days_per_week=replicate_count,
        dropped_weeks=dropped_weeks,
        imputed_weeks=imputed_weeks,
        partial_week_policy=partial_week_policy,
        start_week=str(week_labels[0].date()),
        end_week=str(week_labels[-1].date()),
        data_hash=hash_daily_returns(daily_returns),
    )
    return BalancedPanel(
        weekly=weekly_df,
        week_map=week_map,
        replicates=replicate_count,
        ordered_tickers=ordered_tickers,
        dropped_weeks=dropped_weeks,
        imputed_weeks=imputed_weeks,
        manifest=manifest,
    )


def save_balanced_panel(panel: BalancedPanel, path: Path) -> None:
    """Persist a balanced panel payload via pickle."""

    payload = {
        "weekly": panel.weekly,
        "week_map": panel.week_map,
        "replicates": panel.replicates,
        "ordered_tickers": panel.ordered_tickers,
        "dropped_weeks": panel.dropped_weeks,
        "imputed_weeks": panel.imputed_weeks,
        "manifest": panel.manifest.to_dict(),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_balanced_panel(path: Path) -> BalancedPanel:
    """Load a previously persisted balanced panel payload."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    manifest = PanelManifest.from_dict(payload["manifest"])
    return BalancedPanel(
        weekly=payload["weekly"],
        week_map=payload["week_map"],
        replicates=int(payload["replicates"]),
        ordered_tickers=list(payload["ordered_tickers"]),
        dropped_weeks=int(payload.get("dropped_weeks", 0)),
        imputed_weeks=int(payload.get("imputed_weeks", 0)),
        manifest=manifest,
    )


def write_manifest(manifest: PanelManifest, path: Path) -> None:
    """Write manifest JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
