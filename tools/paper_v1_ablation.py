#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHRINKER_ORDER = ("scm", "oas", "rie")
OVERLAY_ORDER = ("off", "on")
PORTFOLIOS = ("ew", "mv")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary file: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Summary file is empty: {path}") from exc


def _normalise(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _parse_run_name(name: str) -> tuple[str, str]:
    tokens = re.split(r"[^a-z0-9]+", name.lower())
    shrinker = next((tok for tok in tokens if tok in SHRINKER_ORDER), None)
    overlay = next((tok for tok in tokens if tok in OVERLAY_ORDER), None)
    if shrinker is None or overlay is None:
        raise ValueError(f"Unable to parse shrinker/overlay from rc_run='{name}'.")
    return shrinker, overlay


def _pick_row(
    df: pd.DataFrame, *, rc_run: str, regime: str, portfolio: str | None = None
) -> pd.Series | None:
    if df.empty:
        return None
    if "rc_run" not in df.columns:
        raise ValueError("summary CSV missing required column: rc_run")
    mask = _normalise(df["rc_run"]).eq(rc_run.lower())
    if "regime" in df.columns:
        mask &= _normalise(df["regime"]).eq(regime.lower())
    if portfolio is not None and "portfolio" in df.columns:
        mask &= _normalise(df["portfolio"]).eq(portfolio.lower())
    subset = df.loc[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _numeric(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row:
        return float("nan")
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return float("nan")


def _string(row: pd.Series | None, key: str, default: str = "") -> str:
    if row is None or key not in row:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return str(value)


def _boolish(row: pd.Series | None, key: str) -> bool | float:
    if row is None or key not in row:
        return float("nan")
    value = row[key]
    if pd.isna(value):
        return float("nan")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return float("nan")


def _cap_sources(row: pd.Series | None) -> str:
    raw = _string(row, "cap_sources")
    if raw in {"nan", "None"}:
        return ""
    return raw


def _overlay_nan(value: float, overlay_flag: str) -> float:
    if overlay_flag == "off":
        return float("nan")
    return value


def _safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(numer) / float(denom)


def build_ablation_table(rc_dir: Path) -> Path:
    summary_dir = rc_dir / "summary"
    perf_path = summary_dir / "summary_perf.csv"
    det_path = summary_dir / "summary_detection.csv"

    perf_df = _read_csv(perf_path)
    det_df = _read_csv(det_path)

    if perf_df.empty or det_df.empty:
        raise ValueError("summary_perf.csv or summary_detection.csv is empty.")

    run_names = sorted(
        set(perf_df.get("rc_run", pd.Series(dtype=str)).dropna().astype(str).unique())
        | set(det_df.get("rc_run", pd.Series(dtype=str)).dropna().astype(str).unique())
    )
    if not run_names:
        raise ValueError("No rc_run entries found in summary tables.")

    rows: list[dict[str, Any]] = []
    for run_name in run_names:
        shrinker, overlay_flag = _parse_run_name(run_name)
        det_row = _pick_row(det_df, rc_run=run_name, regime="full")
        perf_rows = {
            portfolio: _pick_row(perf_df, rc_run=run_name, regime="full", portfolio=portfolio)
            for portfolio in PORTFOLIOS
        }

        cap_active = _boolish(det_row, "cap_active")
        if isinstance(cap_active, float) and np.isnan(cap_active):
            cap_active = _boolish(perf_rows.get("ew"), "cap_active")
        cap_sources = _cap_sources(det_row) or _cap_sources(perf_rows.get("ew"))
        window_coverage = _numeric(det_row, "window_coverage")
        if not np.isfinite(window_coverage):
            window_coverage = _numeric(perf_rows.get("ew"), "window_coverage")

        windows = _numeric(det_row, "windows")
        detection_windows = _numeric(det_row, "detection_windows")
        changed_share = _safe_ratio(detection_windows, windows)

        row = {
            "rc_run": run_name,
            "shrinker": shrinker,
            "overlay_flag": overlay_flag,
            "cap_active": cap_active,
            "cap_sources": cap_sources,
            "windows_evaluated": windows,
            "window_coverage": window_coverage,
            "detection_rate_mean": _numeric(det_row, "detection_rate_mean"),
            "detection_rate_median": _numeric(det_row, "detection_rate_median"),
            "detection_windows": detection_windows,
            "windows": windows,
            "changed_share": changed_share,
            "isolation_share_mean": _numeric(det_row, "isolation_share_mean"),
            "edge_margin_mean": _numeric(det_row, "edge_margin_mean"),
            "stability_margin_mean": _numeric(det_row, "stability_margin_mean"),
            "alignment_cos_mean": _numeric(det_row, "alignment_cos_mean"),
            "reason_code_mode": _string(det_row, "reason_code_mode"),
        }

        for portfolio in PORTFOLIOS:
            perf_row = perf_rows.get(portfolio)
            prefix = portfolio
            row[f"delta_mse_{prefix}"] = _overlay_nan(
                _numeric(perf_row, "delta_mse_vs_baseline"), overlay_flag
            )
            row[f"delta_qlike_{prefix}"] = _overlay_nan(
                _numeric(perf_row, "delta_qlike_vs_baseline"), overlay_flag
            )
            row[f"n_effective_mse_{prefix}"] = _numeric(perf_row, "n_effective_mse")
            row[f"n_effective_qlike_{prefix}"] = _numeric(perf_row, "n_effective_qlike")
            row[f"comparison_valid_mse_{prefix}"] = _numeric(
                perf_row, "comparison_valid_mse"
            )
            row[f"comparison_valid_qlike_{prefix}"] = _numeric(
                perf_row, "comparison_valid_qlike"
            )
            row[f"n_effective_dm_{prefix}"] = _numeric(perf_row, "n_effective")
            row[f"comparison_valid_dm_{prefix}"] = _numeric(
                perf_row, "comparison_valid_dm"
            )
            row[f"comparison_valid_delta_{prefix}"] = _numeric(
                perf_row, "comparison_valid_delta"
            )

        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("No valid shrinker/overlay rows parsed from summary tables.")

    out_df["_shrinker_order"] = pd.Categorical(
        out_df["shrinker"], categories=list(SHRINKER_ORDER), ordered=True
    )
    out_df["_overlay_order"] = pd.Categorical(
        out_df["overlay_flag"], categories=list(OVERLAY_ORDER), ordered=True
    )
    out_df = out_df.sort_values(["_shrinker_order", "_overlay_order", "rc_run"], kind="mergesort")
    out_df = out_df.drop(columns=["_shrinker_order", "_overlay_order"])

    summary_dir.mkdir(parents=True, exist_ok=True)
    out_path = summary_dir / "paper_v1_ablation.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-v1 ablation table from summary CSVs.")
    parser.add_argument("--rc-dir", type=Path, required=True, help="Root RC directory containing summary/*.csv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = build_ablation_table(args.rc_dir.resolve())
    print(f"[paper_v1_ablation] Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
