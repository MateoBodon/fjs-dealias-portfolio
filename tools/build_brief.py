#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from build_memo import (
    _discover_run_paths,
    _load_config,
    _markdown_table,
    _prettify_reason,
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

    for run_path in run_paths:
        frames = load_run(run_path)
        panel_df = collect_estimator_panel([run_path])
        if panel_df.empty:
            continue
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

    if top_reason != "n/a":
        alerts_text = f"Monitor gating diagnostics: {top_reason} dominates recent windows."
        next_steps = "Revisit gating thresholds and review flagged tickers with advisors."
    else:
        alerts_text = "No dominant gating reason detected across runs."
        next_steps = "Maintain current calibration; confirm advisor feedback after next RC."

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
