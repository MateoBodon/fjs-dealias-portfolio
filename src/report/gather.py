from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

__all__ = ["load_run", "find_runs", "collect_estimator_panel"]

TAGGED_PATTERN = re.compile(r"^[^/]+_J\d+_solver-[^/]+_est-[^/]+_prep-[^/]+$")

DM_SUFFIXES = {
    "Ledoit-Wolf": "lw",
    "OAS": "oas",
    "Constant-Correlation": "cc",
    "Factor": "factor",
    "Tyler-Shrink": "tyler",
}


def load_run(path: Path | str) -> dict[str, pd.DataFrame]:
    """Load core artifacts for a single run directory."""

    run_path = Path(path).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory {run_path} does not exist.")

    frames: dict[str, pd.DataFrame] = {}

    metrics_path = run_path / "metrics_summary.csv"
    if metrics_path.exists():
        frames["metrics"] = pd.read_csv(metrics_path)
    else:
        frames["metrics"] = pd.DataFrame()

    rolling_path = run_path / "rolling_results.csv"
    if rolling_path.exists():
        frames["rolling"] = pd.read_csv(rolling_path)
    else:
        frames["rolling"] = pd.DataFrame()

    summary_path = run_path / "summary.json"
    if summary_path.exists():
        summary_obj = json.loads(summary_path.read_text())
        frames["summary"] = pd.json_normalize(summary_obj)
    else:
        frames["summary"] = pd.DataFrame()

    frames["run_path"] = pd.DataFrame({"run": [run_path]})
    return frames


def find_runs(root: Path | str, pattern: str | None = None) -> list[Path]:
    """Discover run directories, preferring tagged folders."""

    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root {root_path} does not exist.")

    if pattern:
        candidates = [p for p in root_path.glob(pattern) if p.is_dir()]
    else:
        candidates = [p for p in root_path.iterdir() if p.is_dir()]

    tagged = sorted({p.resolve() for p in candidates if TAGGED_PATTERN.match(p.name)}, key=lambda p: p.name)
    legacy = sorted({p.resolve() for p in candidates if not TAGGED_PATTERN.match(p.name)}, key=lambda p: p.name)

    if tagged:
        if legacy:
            print(
                f"[gather] Skipping {len(legacy)} legacy run(s) under {root_path}",
                file=sys.stderr,
            )
        return tagged

    return legacy


def _extract_detection(summary_df: pd.DataFrame) -> float:
    if summary_df.empty:
        return float("nan")
    return float(summary_df.get("detection_rate", pd.Series([float("nan")])).iloc[0])


def _extract_edge_stats(summary_df: pd.DataFrame) -> dict[str, float]:
    stats = {"edge_margin_count": float("nan"), "edge_margin_median": float("nan"), "edge_margin_iqr": float("nan")}
    if summary_df.empty:
        return stats
    for key, column in [
        ("edge_margin_count", "edge_margin_stats.count"),
        ("edge_margin_median", "edge_margin_stats.median"),
        ("edge_margin_iqr", "edge_margin_stats.iqr"),
    ]:
        if column in summary_df:
            value = summary_df[column].iloc[0]
            stats[key] = float(value) if pd.notna(value) else float("nan")
    return stats


def _dm_values(de_row: pd.Series, estimator: str) -> tuple[float, float]:
    suffix = DM_SUFFIXES.get(estimator)
    if not suffix:
        return float("nan"), float("nan")
    p_col = f"dm_p_de_vs_{suffix}"
    stat_col = f"dm_stat_de_vs_{suffix}"
    dm_p = float(de_row.get(p_col, float("nan")))
    dm_stat = float(de_row.get(stat_col, float("nan")))
    return dm_p, dm_stat


def collect_estimator_panel(run_paths: Sequence[Path | str]) -> pd.DataFrame:
    """Combine estimator diagnostics across runs into a single table."""

    records: list[dict[str, float | int | str]] = []

    for run in map(Path, run_paths):
        run = run.resolve()
        data = load_run(run)
        metrics = data["metrics"].copy()
        summary = data["summary"].copy()

        detection_rate = _extract_detection(summary)
        edge_stats = _extract_edge_stats(summary)

        if metrics.empty:
            continue

        if "strategy" not in metrics or "estimator" not in metrics:
            continue

        for strategy, strategy_df in metrics.groupby("strategy"):
            de_mask = strategy_df["estimator"] == "De-aliased"
            if not de_mask.any():
                continue
            de_row = strategy_df[de_mask].iloc[0]
            de_mean = float(de_row.get("mean_mse", float("nan")))

            for _, row in strategy_df.iterrows():
                estimator = row["estimator"]
                if estimator == "De-aliased":
                    continue

                mean_mse = float(row.get("mean_mse", float("nan")))
                delta_mse = mean_mse - de_mean if pd.notna(mean_mse) and pd.notna(de_mean) else float("nan")
                dm_p, dm_stat = _dm_values(de_row, estimator)

                record = {
                    "run": run.name,
                    "run_path": str(run),
                    "crisis_label": row.get("label", ""),
                    "strategy": strategy,
                    "estimator": estimator,
                    "mean_mse": mean_mse,
                    "delta_mse_vs_de": delta_mse,
                    "dm_p": dm_p,
                    "dm_stat": dm_stat,
                    "n_windows": int(row.get("n_windows", 0)),
                    "detection_rate": detection_rate,
                    **edge_stats,
                }
                records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "run",
                "run_path",
                "label",
                "strategy",
                "estimator",
                "mean_mse",
                "delta_mse_vs_de",
                "dm_p",
                "dm_stat",
                "n_windows",
                "detection_rate",
                "edge_margin_count",
                "edge_margin_median",
                "edge_margin_iqr",
            ]
        )

    return pd.DataFrame.from_records(records)
