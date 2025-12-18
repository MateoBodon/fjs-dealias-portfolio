#!/usr/bin/env python3
"""
Lightweight aggregator for rc-lite-sanity outputs.

Combines daily eval (dow/vol) diagnostics with weekly equity-panel runs
to emit a compact summary JSON and optional regime CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_manifest_payload(run_dir: Path) -> dict[str, Any] | None:
    """Load run manifest (run_manifest.json preferred, fall back to run.json/run_meta.json)."""

    candidates = ("run_manifest.json", "run.json", "run_meta.json")
    for name in candidates:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text())
        except Exception:
            continue
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        for name in candidates:
            path = child / name
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text())
            except Exception:
                continue
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _window_meta(run_dir: Path) -> dict[str, Any]:
    manifest = _load_manifest_payload(run_dir)
    if manifest is None:
        return {
            "windows_total": None,
            "windows_evaluated": None,
            "window_coverage": None,
            "max_windows": None,
            "cap_active": True,
            "cap_sources": ["manifest_missing"],
            "incomplete_reason": "manifest_missing",
            "included": False,
        }

    base = manifest.get("manifest", manifest)
    windows_section = base.get("windows", {})

    def pick(key: str) -> Any:
        if key in base and base[key] is not None:
            return base[key]
        return windows_section.get(key)

    windows_total = _int_or_none(
        pick("windows_total") or pick("total") or pick("windows")
    )
    windows_evaluated = _int_or_none(
        pick("windows_evaluated")
        or pick("evaluated")
        or pick("after_regime")
        or pick("completed")
    )
    max_windows = _int_or_none(pick("max_windows"))
    coverage = pick("window_coverage") or pick("coverage")
    cap_active = bool(pick("cap_active"))
    cap_sources = pick("cap_sources") or []
    reason = pick("incomplete_reason")

    if coverage is None and windows_total is not None and windows_evaluated is not None:
        if windows_total > 0:
            coverage = float(windows_evaluated) / float(windows_total)
        else:
            coverage = None

    included = True
    if windows_total is None or windows_evaluated is None:
        included = False
        reason = reason or "missing_window_counts"
    if cap_active:
        included = False
    if coverage is not None and coverage < 0.9999:
        included = False
        reason = reason or "window_coverage_lt_1"

    return {
        "windows_total": windows_total,
        "windows_evaluated": windows_evaluated,
        "window_coverage": coverage,
        "max_windows": max_windows,
        "cap_active": cap_active,
        "cap_sources": cap_sources if isinstance(cap_sources, list) else [cap_sources],
        "incomplete_reason": reason,
        "included": included,
    }


def _delta_mse(metrics: pd.DataFrame, portfolio: str) -> float | None:
    if metrics.empty:
        return None
    subset = metrics[metrics["regime"] == "full"]
    overlay = subset[(subset["estimator"] == "overlay") & (subset["portfolio"] == portfolio)]
    baseline = subset[(subset["estimator"] == "baseline") & (subset["portfolio"] == portfolio)]
    if overlay.empty or baseline.empty:
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


def _load_daily(path: Path, label: str) -> tuple[dict[str, Any], pd.DataFrame | None]:
    meta = _window_meta(path)
    record: dict[str, Any] = {
        "path": str(path),
        "windows_total": meta["windows_total"],
        "windows_evaluated": meta["windows_evaluated"],
        "window_coverage": meta["window_coverage"],
        "max_windows": meta["max_windows"],
        "cap_active": meta["cap_active"],
        "cap_sources": meta["cap_sources"],
        "incomplete_reason": meta["incomplete_reason"],
        "included_in_summary": meta["included"],
    }
    if not meta["included"]:
        print(
            f"[summarize_rc_sanity] excluding {label} (cap/incomplete): {meta['incomplete_reason']}",
            file=sys.stderr,
        )
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

    if not record.get("included_in_summary", True):
        for key in (
            "detection_rate",
            "percent_changed",
            "alignment_cos",
            "delta_mse_ew",
            "delta_mse_mv",
            "overlay_effect",
            "reason_mode",
        ):
            if key in record:
                record[key] = None
    return record, regime_df


def _load_weekly(path: Path, label: str) -> dict[str, Any]:
    meta = _window_meta(path)
    record: dict[str, Any] = {
        "path": str(path),
        "windows_total": meta["windows_total"],
        "windows_evaluated": meta["windows_evaluated"],
        "window_coverage": meta["window_coverage"],
        "max_windows": meta["max_windows"],
        "cap_active": meta["cap_active"],
        "cap_sources": meta["cap_sources"],
        "incomplete_reason": meta["incomplete_reason"],
        "included_in_summary": meta["included"],
    }
    if not meta["included"]:
        print(
            f"[summarize_rc_sanity] excluding {label} (cap/incomplete): {meta['incomplete_reason']}",
            file=sys.stderr,
        )
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
                pd.to_numeric(det_df["n_detections"], errors="coerce")
                .fillna(0)
                .gt(0)
                .mean()
            )
            record["accept_share"] = float(accept_share)
            if "skip_reason" in det_df.columns:
                reasons = det_df["skip_reason"].dropna()
                if not reasons.empty:
                    record["skip_reason_top"] = reasons.value_counts().head(3).to_dict()
    if not record.get("included_in_summary", True):
        for key in (
            "detection_rate",
            "rolling_windows",
            "nested_scope",
            "nested_prep_events",
            "gating_skips",
            "accept_share",
            "skip_reason_top",
        ):
            if key in record:
                record[key] = None
    return record


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

    daily_dow, reg_dow = _load_daily(Path(args.dow_dir), "daily_dow")
    entries["daily_dow"] = daily_dow
    if reg_dow is not None:
        regime_frames.append(reg_dow)

    daily_vol, reg_vol = _load_daily(Path(args.vol_dir), "daily_vol")
    entries["daily_vol"] = daily_vol
    if reg_vol is not None:
        regime_frames.append(reg_vol)

    entries["weekly_dow"] = _load_weekly(Path(args.weekly_dow_dir), "weekly_dow")
    entries["nested_weekly"] = _load_weekly(Path(args.nested_dir), "nested_weekly")

    summary = {"rc_dir": str(root), "entries": entries}
    exclusions = [
        {
            "label": key,
            "reason": value.get("incomplete_reason"),
            "cap_active": value.get("cap_active"),
        }
        for key, value in entries.items()
        if not value.get("included_in_summary", True)
    ]
    if exclusions:
        summary["excluded"] = exclusions
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
