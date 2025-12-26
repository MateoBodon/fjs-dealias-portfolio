#!/usr/bin/env python3
"""
Lightweight aggregator for rc-lite-sanity outputs.

Combines daily eval (dow/vol) diagnostics with weekly equity-panel runs
to emit a compact summary JSON and optional regime CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meta.completeness import CompletenessResult, evaluate_eval_run, evaluate_weekly_run


def _delta_mse(metrics: pd.DataFrame, portfolio: str) -> float | None:
    if metrics.empty:
        return None
    subset = metrics[metrics["regime"] == "full"]
    overlay = subset[(subset["estimator"] == "overlay") & (subset["portfolio"] == portfolio)]
    baseline = subset[(subset["estimator"] == "baseline") & (subset["portfolio"] == portfolio)]
    if overlay.empty:
        return None
    if "delta_mse_vs_baseline" in overlay.columns:
        value = pd.to_numeric(overlay["delta_mse_vs_baseline"], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])
    if baseline.empty or "sq_error" not in overlay.columns or "sq_error" not in baseline.columns:
        return None
    return float(overlay["sq_error"].mean() - baseline["sq_error"].mean())


def _effect_label(delta_ew: float | None, delta_mv: float | None) -> str:
    vals = [val for val in (delta_ew, delta_mv) if val is not None]
    if not vals:
        return "unknown"
    harmful = any(val > 0 for val in vals)
    helpful = any(val < 0 for val in vals)
    if harmful and helpful:
        return "mixed"
    if harmful:
        return "harmful"
    if helpful:
        return "helpful"
    return "neutral"


def _load_daily_payload(path: Path, label: str) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Best-effort loader for daily diagnostics/metrics."""

    record: dict[str, Any] = {}
    diag_path = path / "diagnostics.csv"
    if not diag_path.exists():
        diag_path = path / "full" / "diagnostics.csv"
    regime_path = path / "regime.csv"
    regime_df: pd.DataFrame | None = None

    if diag_path.exists():
        try:
            diag_df = pd.read_csv(diag_path)
        except Exception:
            diag_df = pd.DataFrame()
        if not diag_df.empty:
            record["detection_rate"] = float(
                pd.to_numeric(diag_df.get("detection_rate"), errors="coerce").mean()
            )
            if "percent_changed" in diag_df.columns:
                record["percent_changed"] = float(
                    pd.to_numeric(diag_df.get("percent_changed"), errors="coerce").mean()
                )
            if "alignment_cos_mean" in diag_df.columns:
                record["alignment_cos"] = float(
                    pd.to_numeric(diag_df["alignment_cos_mean"], errors="coerce").mean()
                )
            if "reason_code" in diag_df.columns:
                reasons = diag_df["reason_code"].dropna()
                if not reasons.empty:
                    record["reason_mode"] = reasons.mode().iloc[0]

    metrics_path = path / "metrics_detail.csv"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
        except Exception:
            metrics_df = pd.DataFrame()
        if not metrics_df.empty:
            delta_ew = _delta_mse(metrics_df, "ew")
            delta_mv = _delta_mse(metrics_df, "mv")
            record["delta_mse_ew"] = delta_ew
            record["delta_mse_mv"] = delta_mv
            record["overlay_effect"] = _effect_label(delta_ew, delta_mv)

    if regime_path.exists():
        try:
            regime_df = pd.read_csv(regime_path)
            regime_df.insert(0, "design", label)
        except Exception:
            regime_df = None

    return record, regime_df


def _load_weekly_payload(path: Path) -> dict[str, Any]:
    """Best-effort loader for weekly summary + detection."""

    record: dict[str, Any] = {}
    summary_path = path / "summary.json"
    if not summary_path.exists():
        candidates = list(path.glob("*/summary.json"))
        if candidates:
            summary_path = candidates[0]
    det_path = path / "detection_summary.csv"
    if not det_path.exists():
        candidates = list(path.glob("*/detection_summary.csv"))
        if candidates:
            det_path = candidates[0]
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
        except Exception:
            payload = {}
        record["detection_rate"] = payload.get("detection_rate")
        record["rolling_windows"] = payload.get("rolling_windows_evaluated")
        if payload.get("nested_scope"):
            record["nested_scope"] = payload.get("nested_scope")
        if payload.get("nested_preparation_events") is not None:
            record["nested_prep_events"] = payload.get("nested_preparation_events")
        gating = payload.get("gating") or {}
        if gating.get("skip_reasons"):
            record["gating_skips"] = gating.get("skip_reasons")
    if det_path.exists():
        try:
            det_df = pd.read_csv(det_path)
        except Exception:
            det_df = pd.DataFrame()
        if not det_df.empty and "n_detections" in det_df.columns:
            accept_share = (
                pd.to_numeric(det_df["n_detections"], errors="coerce").fillna(0).gt(0).mean()
            )
            record["accept_share"] = float(accept_share)
            if "skip_reason" in det_df.columns:
                reasons = det_df["skip_reason"].dropna()
                if not reasons.empty:
                    record["skip_reason_top"] = reasons.value_counts().head(3).to_dict()
    return record


def _merge_completeness(entry: dict[str, Any], comp: CompletenessResult) -> None:
    entry.update(
        {
            "status": comp.status,
            "is_complete": comp.is_complete,
            "missing_files": comp.missing_files,
            "incomplete_reason": comp.incomplete_reason,
            "cap_active": comp.cap_active,
            "cap_sources": comp.cap_sources,
            "window_coverage": comp.window_coverage,
            "windows_evaluated": comp.windows_evaluated,
            "windows_total": comp.windows_total,
            "excluded_from_aggregate": comp.excluded_from_aggregate,
        }
    )


def _build_daily_entry(label: str, path: Path) -> tuple[dict[str, Any], pd.DataFrame | None, CompletenessResult]:
    completeness = evaluate_eval_run(path, label=label, allow_unknown_coverage=True)
    entry: dict[str, Any] = {"label": label, "path": str(path), "section": "daily"}
    regime_df: pd.DataFrame | None = None
    if completeness.present and not completeness.missing_files:
        payload, regime_df = _load_daily_payload(path, label)
        entry.update(payload)
    _merge_completeness(entry, completeness)
    if completeness.incomplete_reason and "reason_mode" not in entry:
        entry["reason_mode"] = completeness.incomplete_reason
    return entry, regime_df, completeness


def _build_weekly_entry(label: str, path: Path) -> tuple[dict[str, Any], CompletenessResult]:
    completeness = evaluate_weekly_run(path, label=label)
    entry: dict[str, Any] = {"label": label, "path": str(path), "section": "weekly"}
    if completeness.present and not completeness.missing_files:
        entry.update(_load_weekly_payload(path))
    _merge_completeness(entry, completeness)
    return entry, completeness


def _aggregate_entries(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    usable = [e for e in entries if not e.get("excluded_from_aggregate", True)]
    metrics: dict[str, Any] = {"included": [e["label"] for e in usable]}
    if not usable:
        metrics["reason"] = "no complete runs eligible for aggregation"
        return metrics

    def _mean(values: list[float | None]) -> float | None:
        clean = [float(v) for v in values if v is not None and not np.isnan(v)]
        if not clean:
            return None
        return float(np.mean(clean))

    metrics["detection_rate_mean"] = _mean([e.get("detection_rate") for e in usable])
    metrics["delta_mse_ew_mean"] = _mean([e.get("delta_mse_ew") for e in usable])
    metrics["delta_mse_mv_mean"] = _mean([e.get("delta_mse_mv") for e in usable])
    metrics["accept_share_mean"] = _mean([e.get("accept_share") for e in usable])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rc-lite-sanity outputs.")
    parser.add_argument("--rc-dir", required=True, help="Root rc output directory.")
    parser.add_argument("--dow-dir", required=True, help="Daily DoW run directory.")
    parser.add_argument("--vol-dir", required=True, help="Daily vol-state run directory.")
    parser.add_argument("--weekly-dow-dir", required=True, help="Weekly DoW run directory.")
    parser.add_argument("--nested-dir", required=True, help="Weekly nested run directory.")
    args = parser.parse_args()

    root = Path(args.rc_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    entries: dict[str, Any] = {}
    regime_frames: list[pd.DataFrame] = []
    completeness_records: list[CompletenessResult] = []

    daily_dow, reg_dow, comp_dow = _build_daily_entry("daily_dow", Path(args.dow_dir))
    entries["daily_dow"] = daily_dow
    completeness_records.append(comp_dow)
    if reg_dow is not None:
        regime_frames.append(reg_dow)

    daily_vol, reg_vol, comp_vol = _build_daily_entry("daily_vol", Path(args.vol_dir))
    entries["daily_vol"] = daily_vol
    completeness_records.append(comp_vol)
    if reg_vol is not None:
        regime_frames.append(reg_vol)

    weekly_dow, comp_weekly_dow = _build_weekly_entry("weekly_dow", Path(args.weekly_dow_dir))
    entries["weekly_dow"] = weekly_dow
    completeness_records.append(comp_weekly_dow)

    nested_weekly, comp_nested = _build_weekly_entry("nested_weekly", Path(args.nested_dir))
    entries["nested_weekly"] = nested_weekly
    completeness_records.append(comp_nested)

    incomplete_runs = [
        {
            "label": comp.label,
            "path": str(comp.path),
            "status": comp.status,
            "reason": comp.incomplete_reason,
            "missing_files": comp.missing_files,
            "cap_active": comp.cap_active,
            "window_coverage": comp.window_coverage,
            "excluded_from_aggregate": comp.excluded_from_aggregate,
        }
        for comp in completeness_records
        if comp.status != "complete" or comp.excluded_from_aggregate
    ]

    aggregate = _aggregate_entries(entries.values())

    summary = {
        "rc_dir": str(root),
        "entries": entries,
        "aggregate": aggregate,
        "incomplete_runs": incomplete_runs,
    }

    regime_out = root / "regime.csv"
    if regime_frames:
        pd.concat(regime_frames, ignore_index=True).to_csv(regime_out, index=False)
        summary["regime_csv"] = str(regime_out)

    summary_path = root / "summary_sanity.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[rc-lite-sanity] wrote {summary_path}")
    if regime_out.exists():
        print(f"[rc-lite-sanity] wrote {regime_out}")


if __name__ == "__main__":
    main()
