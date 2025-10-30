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
    "Aliased": "alias",
    "SCM": "scm",
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


def _ci_bounds(de_row: pd.Series, estimator: str) -> tuple[float, float]:
    suffix = DM_SUFFIXES.get(estimator)
    if not suffix:
        return float("nan"), float("nan")
    lo_key = f"ci_lo_de_minus_{suffix}"
    hi_key = f"ci_hi_de_minus_{suffix}"
    ci_lo = de_row.get(lo_key, float("nan"))
    ci_hi = de_row.get(hi_key, float("nan"))
    if pd.isna(ci_lo) or pd.isna(ci_hi):
        return float("nan"), float("nan")
    # Stored bounds are (De - estimator); convert to (estimator - De)
    try:
        lo_val = float(ci_lo)
        hi_val = float(ci_hi)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    est_lo = -hi_val
    est_hi = -lo_val
    if est_lo > est_hi:
        est_lo, est_hi = est_hi, est_lo
    return est_lo, est_hi


def collect_estimator_panel(run_paths: Sequence[Path | str]) -> pd.DataFrame:
    """Combine estimator diagnostics across runs into a single table."""

    records: list[dict[str, float | int | str]] = []

    for run in map(Path, run_paths):
        run = run.resolve()
        data = load_run(run)
        metrics = data["metrics"].copy()
        summary = data["summary"].copy()

        design_value = ""
        if not summary.empty and "design" in summary:
            design_series = summary["design"].dropna()
            if not design_series.empty:
                design_value = str(design_series.iloc[0])
        windows_evaluated = float("nan")
        if not summary.empty and "rolling_windows_evaluated" in summary:
            window_series = summary["rolling_windows_evaluated"].dropna()
            if not window_series.empty:
                try:
                    windows_evaluated = float(window_series.iloc[0])
                except (TypeError, ValueError):
                    windows_evaluated = float("nan")
        summary_crisis = ""
        if not summary.empty and "crisis_label" in summary:
            crisis_series = summary["crisis_label"].dropna()
            if not crisis_series.empty:
                summary_crisis = str(crisis_series.iloc[0])

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
                ci_lo, ci_hi = _ci_bounds(de_row, estimator)

                crisis_value = summary_crisis or str(row.get("label", ""))

                record = {
                    "run": run.name,
                    "run_path": str(run),
                    "crisis_label": crisis_value,
                    "strategy": strategy,
                    "estimator": estimator,
                    "mean_mse": mean_mse,
                    "delta_mse_vs_de": delta_mse,
                    "dm_p": dm_p,
                    "dm_stat": dm_stat,
                    "n_windows": int(row.get("n_windows", 0)),
                    "detection_rate": detection_rate,
                    **edge_stats,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "design": design_value,
                    "rolling_windows_evaluated": windows_evaluated,
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
