#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_memo import (
    _discover_run_paths,
    _load_config,
    _load_summary_artifacts,
    _markdown_table,
    _prettify_reason,
    _format_kill_criteria_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from report.gather import collect_estimator_panel, load_run


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _aggregate_reason_table(reason_df: pd.DataFrame) -> tuple[str, str]:
    if reason_df.empty:
        return "(no reason-code diagnostics)", "n/a"
    totals = reason_df.groupby("reason_code")["count"].sum().sort_values(ascending=False)
    top_reason = "n/a"
    if not totals.empty:
        top_reason_key = totals.index[0]
        top_share = totals.iloc[0] / totals.sum() if totals.sum() else float("nan")
        top_reason = f"{_prettify_reason(top_reason_key)} ({_format_percent(top_share)})"
    reason_by_run = reason_df.groupby(["run", "reason_code"], as_index=False)["count"].sum()
    totals_per_run = reason_by_run.groupby("run")["count"].sum().reset_index(name="total")
    pivot = reason_by_run.merge(totals_per_run, on="run", how="left")
    pivot["share"] = pivot.apply(
        lambda row: row["count"] / row["total"] if row["total"] else float("nan"),
        axis=1,
    )
    table = pivot.pivot(index="run", columns="reason_code", values="share").fillna(0.0).reset_index()
    table = table.rename(columns=lambda col: _prettify_reason(col) if col != "run" else col)
    for col in table.columns:
        if col == "run":
            continue
        table[col] = table[col].apply(_format_percent)
    return _markdown_table(table), top_reason


def build_brief(config_path: Path) -> Path:
    config = _load_config(config_path)
    gallery_cfg = config.get("gallery", {}) or {}
    gallery_name = gallery_cfg.get("name", "advisor_brief")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_entries = config.get("runs", []) or []
    run_paths = _discover_run_paths(run_entries)
    if not run_paths:
        raise ValueError("No runs discovered for advisor brief.")

    detection_rates: list[float] = []
    dm_pvalues: list[float] = []
    reason_records: list[dict[str, object]] = []
    factor_baseline_lines: list[str] = []

    for run_path in run_paths:
        frames = load_run(run_path)
        panel_df = collect_estimator_panel([run_path])
        if panel_df.empty:
            continue
        summary_json: dict[str, object] = {}
        summary_json_path = run_path / "summary.json"
        if summary_json_path.exists():
            try:
                summary_json = json.loads(summary_json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary_json = {}
        prewhiten_payload = summary_json.get("prewhiten") if isinstance(summary_json, dict) else {}
        if isinstance(prewhiten_payload, dict) and prewhiten_payload:
            mode_effective = str(prewhiten_payload.get("mode_effective", "off"))
            r2_mean_val = prewhiten_payload.get("r2_mean")
            factor_cols = prewhiten_payload.get("factor_columns") or []
            if factor_cols:
                factor_fragment = ", ".join(factor_cols[:3])
                if len(factor_cols) > 3:
                    factor_fragment += ", …"
            else:
                factor_fragment = "n/a"
            if isinstance(r2_mean_val, (int, float)):
                r2_fragment = f"{float(r2_mean_val):.2f}"
            else:
                r2_fragment = "n/a"
            factor_baseline_lines.append(
                f"{run_path.name}: prewhiten {mode_effective} (R²≈{r2_fragment}, factors: {factor_fragment})"
            )
        if "detection_rate" in panel_df:
            detection_rates.extend(panel_df["detection_rate"].dropna().astype(float).tolist())
        dm_columns = [col for col in panel_df.columns if col.startswith("dm_p")]
        for col in dm_columns:
            dm_pvalues.extend(panel_df[col].dropna().astype(float).tolist())
        detail_df = frames.get("diagnostics_detail", pd.DataFrame())
        if not detail_df.empty and "reason_code" in detail_df.columns:
            detail = detail_df.copy()
            detail["run"] = run_path.name
            detail["reason_code"] = detail["reason_code"].fillna("unknown")
            counts = detail.groupby(["run", "regime", "reason_code"], dropna=False).size().reset_index(name="count")
            reason_records.extend(counts.to_dict("records"))

    if detection_rates:
        detection_avg = float(np.nanmean(detection_rates))
        detection_summary = f"Average detection coverage {_format_percent(detection_avg)} across {len(run_paths)} run(s)."
    else:
        detection_summary = "Detection coverage unavailable."

    dm_total = sum(not np.isnan(val) for val in dm_pvalues)
    dm_sig = sum(float(val) < 0.05 for val in dm_pvalues if not np.isnan(val))
    if dm_total:
        dm_summary = f"{dm_sig} of {dm_total} DM comparisons show p < 0.05."
    else:
        dm_summary = "Diebold–Mariano statistics unavailable."

    if reason_records:
        reason_df = pd.DataFrame(reason_records)
    else:
        reason_df = pd.DataFrame(columns=["run", "regime", "reason_code", "count"])

    reason_table_md, top_reason = _aggregate_reason_table(reason_df)

    summary_artifacts = _load_summary_artifacts(config)
    summary_det_df = summary_artifacts["det_df"]
    summary_perf_df = summary_artifacts["perf_df"]
    kill_data = summary_artifacts["kill_data"]
    limitations_text = summary_artifacts["limitations_text"]
    kill_criteria_md, kill_status = _format_kill_criteria_payload(kill_data)
    limitations_fragment = limitations_text if limitations_text else "No critical limitations detected under current criteria."

    if not summary_det_df.empty:
        full_det = summary_det_df[summary_det_df["regime"].astype(str).str.lower() == "full"]
        if not full_det.empty:
            det_row = full_det.iloc[0]
            det_rate = _safe_float(det_row.get("detection_rate_mean"))
            edge_margin = _safe_float(det_row.get("edge_margin_mean"))
            stability_margin = _safe_float(det_row.get("stability_margin_mean"))
            reason_code_mode = det_row.get("reason_code_mode", "")
            if not np.isnan(det_rate):
                detection_summary = (
                    f"Full regime detection {_format_percent(det_rate)} (edge {edge_margin:.3f}, stability {stability_margin:.3f})."
                )
            if reason_code_mode:
                top_reason = _prettify_reason(str(reason_code_mode))

    if not summary_perf_df.empty:
        full_perf = summary_perf_df[summary_perf_df["regime"].astype(str).str.lower() == "full"]
        if not full_perf.empty:
            ew_row = full_perf[full_perf["portfolio"].astype(str).str.lower() == "ew"].iloc[0]
            mv_row = full_perf[full_perf["portfolio"].astype(str).str.lower() == "mv"].iloc[0]
            ew_dm = _safe_float(ew_row.get("dm_p_value"))
            mv_dm = _safe_float(mv_row.get("dm_p_value"))
            dm_parts = []
            if not np.isnan(ew_dm):
                dm_parts.append(f"EW p={ew_dm:.3g}")
            if not np.isnan(mv_dm):
                dm_parts.append(f"MV p={mv_dm:.3g}")
            if dm_parts:
                dm_summary = "Full regime DM p-values: " + ", ".join(dm_parts) + "."

    if top_reason != "n/a":
        alerts_text = f"Monitor gating diagnostics: {top_reason} dominates recent windows."
        next_steps = "Revisit gating thresholds and review flagged tickers with advisors."
    else:
        alerts_text = "No dominant gating reason detected across runs."
        next_steps = "Maintain current calibration; confirm advisor feedback after next RC."

    if kill_status == "CHECK":
        alerts_text = "Kill criteria flagged outstanding issues; review failed checks below."
        primary_issue = limitations_text.splitlines()[0] if limitations_text else "Investigate kill criteria failures with overlay calibration owners."
        next_steps = primary_issue
    elif kill_status == "PASS":
        alerts_text = "Kill criteria satisfied; continue monitoring gating diagnostics."
    elif kill_status == "UNKNOWN" and not kill_criteria_md.strip():
        alerts_text = "Kill criteria unavailable; rely on latest RC diagnostics."

    factor_baseline_text = factor_baseline_lines[0] if factor_baseline_lines else "n/a"

    template_path = Path("reports/templates/brief.md.j2")
    template = template_path.read_text(encoding="utf-8")
    context = {
        "gallery_name": gallery_name,
        "detection_summary": detection_summary,
        "dm_summary": dm_summary,
        "top_reason": top_reason,
        "reason_table_md": reason_table_md,
        "alerts_text": alerts_text,
        "next_steps": next_steps,
        "kill_status": kill_status,
        "kill_criteria_md": kill_criteria_md,
        "limitations_fragment": limitations_fragment,
        "factor_baseline_text": factor_baseline_text,
    }
    brief_text = template.format(**context)

    brief_path = reports_dir / "brief.md"
    brief_path.write_text(brief_text, encoding="utf-8")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    brief_timestamp_path = reports_dir / f"brief_{timestamp}.md"
    brief_timestamp_path.write_text(brief_text, encoding="utf-8")

    print(f"Advisor brief written to {brief_path} and {brief_timestamp_path}")
    return brief_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a one-page advisor brief for RC runs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/equity_panel/config.rc.yaml"),
        help="YAML configuration describing runs to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_brief(args.config.resolve())


if __name__ == "__main__":
    main()
