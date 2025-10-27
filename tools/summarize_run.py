#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        return None
    return None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _fmt_float(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.6g}"


def summarize_run(output_dir: Path) -> int:
    if not (output_dir / "run_meta.json").exists():
        candidates = [
            path
            for path in sorted(output_dir.iterdir())
            if path.is_dir() and (path / "run_meta.json").exists()
        ]
        if len(candidates) == 1:
            output_dir = candidates[0]
        elif len(candidates) > 1:
            print(
                f"Found multiple run directories under {output_dir}.\n"
                "Select one explicitly when summarizing."
            )
            return 1

    run_meta = _read_json(output_dir / "run_meta.json") or {}
    summary = _read_json(output_dir / "summary.json") or {}
    det_df = _safe_read_csv(output_dir / "detection_summary.csv")
    results_df = _safe_read_csv(output_dir / "rolling_results.csv")
    metrics_df = _safe_read_csv(output_dir / "metrics_summary.csv")

    # Consistency checks
    issues: list[str] = []
    n_assets = int(summary.get("n_assets")) if "n_assets" in summary else None
    if n_assets is None and run_meta.get("n_assets") is not None:
        n_assets = int(run_meta["n_assets"])  # type: ignore[index]

    # Basic stats
    n_windows = int(summary.get("rolling_windows_evaluated", 0))
    detections_total = 0
    L_max = 0
    if not det_df.empty and "n_detections" in det_df.columns:
        detections_total = int(
            pd.to_numeric(det_df["n_detections"], errors="coerce").fillna(0).sum()
        )
        L_max = int(
            pd.to_numeric(det_df["n_detections"], errors="coerce").fillna(0).max()
        )

    # Cross-check n_assets with any obvious small-panel signals
    if n_assets is not None and n_assets <= 4:
        issues.append(
            f"n_assets={n_assets} suggests a tiny panel; verify configuration."
        )

    # Check duplicate forecasts when detections exist
    identical_forecasts = 0
    if not results_df.empty and "n_detections" in results_df.columns:
        df = results_df.copy()
        df["n_detections"] = pd.to_numeric(df["n_detections"], errors="coerce").fillna(
            0
        )
        with_det = df[df["n_detections"] > 0]
        if not with_det.empty:
            for prefix in ("eq", "mv", "mvlo"):
                a = f"{prefix}_aliased_forecast"
                d = f"{prefix}_dealiased_forecast"
                if a in with_det.columns and d in with_det.columns:
                    identical_forecasts += int(
                        np.sum(
                            np.isfinite(with_det[a].to_numpy())
                            & np.isfinite(with_det[d].to_numpy())
                            & (np.abs(with_det[a] - with_det[d]) <= 1e-12)
                        )
                    )
            if identical_forecasts > 0:
                issues.append(
                    "De-aliased forecasts identical to aliased in some detection windows."
                )

    # Summarise metrics
    metrics_line = ""
    estimator_lines: list[str] = []
    dm_lines: list[str] = []
    if not metrics_df.empty:
        # Prefer Equal Weight Aliased vs De-aliased if available
        mask_alias = (
            (metrics_df["strategy"] == "Equal Weight")
            & (metrics_df["estimator"] == "Aliased")
        )
        mask_de = (
            (metrics_df["strategy"] == "Equal Weight")
            & (metrics_df["estimator"] == "De-aliased")
        )
        if mask_alias.any() and mask_de.any():
            mse_alias = float(metrics_df.loc[mask_alias, "mean_mse"].iloc[0])
            mse_de = float(metrics_df.loc[mask_de, "mean_mse"].iloc[0])
            metrics_line = f"MSE(eq): aliased={_fmt_float(mse_alias)}, de={_fmt_float(mse_de)}"
        else:
            metrics_line = "Metrics summary available (see CSV)."

        for strategy in sorted(metrics_df["strategy"].unique()):
            subset = metrics_df[metrics_df["strategy"] == strategy]
            pieces = [
                f"{row['estimator']}={_fmt_float(row.get('mean_mse'))}"
                for _, row in subset.sort_values("mean_mse").iterrows()
            ]
            if pieces:
                estimator_lines.append(f"  {strategy}: " + "; ".join(pieces))

        de_rows = metrics_df[metrics_df["estimator"] == "De-aliased"]
        comparator_labels = {
            "lw": "Ledoit-Wolf",
            "oas": "OAS",
            "cc": "Const-Corr",
            "factor": "Factor",
            "tyler": "Tyler-Shrink",
        }
        for _, row in de_rows.iterrows():
            comps: list[str] = []
            for suffix, label in comparator_labels.items():
                p_col = f"dm_p_de_vs_{suffix}"
                stat_col = f"dm_stat_de_vs_{suffix}"
                if p_col in row and stat_col in row:
                    p_val = row[p_col]
                    stat_val = row[stat_col]
                    if np.isfinite(p_val) and np.isfinite(stat_val):
                        comps.append(
                            f"{label}: p={_fmt_float(p_val)}, stat={_fmt_float(stat_val)}"
                        )
            if comps:
                dm_lines.append(f"  {row['strategy']}: " + "; ".join(comps))

    # Print one-page textual summary
    print("== Run Summary ==")
    print(f"Output dir: {output_dir}")
    if run_meta:
        print(
            f"Git SHA: {run_meta.get('git_sha', 'unknown')}  |  a-grid: {run_meta.get('a_grid', 'n/a')}  "
            f"|  signed_a: {run_meta.get('signed_a', 'n/a')}"
        )
        design = run_meta.get("design", "n/a")
        nested = run_meta.get("nested_replicates", "n/a")
        estimator = run_meta.get("estimator", "n/a")
        preprocess = run_meta.get("preprocess_flags") or run_meta.get(
            "panel_preprocess_flags", {}
        )
        preprocess_str = (
            ", ".join(f"{k}={v}" for k, v in sorted(preprocess.items()))
            if isinstance(preprocess, dict) and preprocess
            else "none"
        )
        solver_list = run_meta.get("solver_used")
        solver_str = (
            ", ".join(str(s) for s in solver_list)
            if isinstance(solver_list, list) and solver_list
            else "n/a"
        )
        print(
            f"Design: {design}  |  nested J: {nested}  |  estimator: {estimator}  "
            f"|  preprocess: {preprocess_str}  |  solver_used: {solver_str}"
        )
        print(
            f"delta: {_fmt_float(run_meta.get('delta'))}  |  delta_frac: {_fmt_float(run_meta.get('delta_frac'))}  "
            f"|  sigma2_plugin: {run_meta.get('sigma2_plugin', 'n/a')}"
        )
    if summary:
        print(
            f"Period: {summary.get('start_date', '?')} → {summary.get('end_date', '?')}  "
            f"|  n_assets: {n_assets if n_assets is not None else 'n/a'}  "
            f"|  windows: {n_windows}"
        )
    detection_windows = int(summary.get("detection_windows", 0))
    detection_rate = summary.get("detection_rate")
    if detections_total or L_max:
        print(f"Detections: total={detections_total}  |  L(max)={L_max}")
    if n_windows:
        frac = (
            float(detection_windows) / float(n_windows)
            if detection_rate is None or not np.isfinite(detection_rate)
            else float(detection_rate)
        )
        pct = 100.0 * frac
        print(
            f"Detection rate: {detection_windows}/{n_windows} windows ({pct:.1f}%)"
        )
    edge_stats = summary.get("edge_margin_stats", {})
    if isinstance(edge_stats, dict) and edge_stats.get("count"):
        median = edge_stats.get("median")
        iqr = edge_stats.get("iqr")
        print(
            "Edge margin (lambda_hat - z_plus): "
            f"count={edge_stats.get('count')}  |  median={_fmt_float(median)}  |  IQR={_fmt_float(iqr)}"
        )
    reject_stats = summary.get("rejection_stats", {})
    if isinstance(reject_stats, dict) and reject_stats:
        total_reject = int(sum(int(v) for v in reject_stats.values()))
        if total_reject > 0:
            top_items = [
                item
                for item in sorted(
                    ((str(k), int(v)) for k, v in reject_stats.items()),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                if item[1] > 0
            ][:5]
            if top_items:
                print("Rejection reasons:")
                for reason, count in top_items:
                    pct = 100.0 * count / float(total_reject)
                    print(f"  - {reason}: {count} ({pct:.1f}%)")
            else:
                print("Rejection reasons: none recorded")
        else:
            print("Rejection reasons: none recorded")
    if metrics_line:
        print(metrics_line)
    if estimator_lines:
        print("Estimator MSE breakdown:")
        for line in estimator_lines:
            print(line)
    if dm_lines:
        print("DM comparison (De-aliased vs baselines):")
        for line in dm_lines:
            print(line)
    if run_meta and run_meta.get("figure_sha256"):
        pdfs = run_meta["figure_sha256"]
        print(f"Figures (PDF hashes): {len(pdfs)} files")
    if issues:
        print("-- Consistency checks --")
        for msg in issues:
            print(f"[WARN] {msg}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a de-aliasing run")
    parser.add_argument(
        "output_dir", type=Path, help="Directory with CSVs and run_meta.json"
    )
    args = parser.parse_args()
    summarize_run(args.output_dir)


if __name__ == "__main__":
    main()
