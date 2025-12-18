#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def _load_manifest_payload(run_dir: Path) -> dict[str, Any] | None:
    """Load run manifest (prefers run_manifest.json, falls back to run.json/run_meta.json)."""

    for name in ("run_manifest.json", "run.json", "run_meta.json"):
        candidate = run_dir / name
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text())
        except Exception:
            continue
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        for name in ("run_manifest.json", "run.json", "run_meta.json"):
            candidate = child / name
            if not candidate.exists():
                continue
            try:
                return json.loads(candidate.read_text())
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


def _aggregate_window_meta(metas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    meta_list = list(metas)
    if not meta_list:
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
    totals = [m.get("windows_total") for m in meta_list]
    evals = [m.get("windows_evaluated") for m in meta_list]
    if any(val is None for val in totals) or any(val is None for val in evals):
        total_windows = None
        eval_windows = None
    else:
        total_windows = int(sum(int(val) for val in totals)) if totals else None
        eval_windows = int(sum(int(val) for val in evals)) if evals else None
    coverage = None
    if total_windows is not None and eval_windows is not None and total_windows > 0:
        coverage = float(eval_windows) / float(total_windows)
    cap_active = any(bool(m.get("cap_active")) for m in meta_list)
    cap_sources: list[str] = []
    for meta in meta_list:
        cap_sources.extend(meta.get("cap_sources") or [])
    cap_sources = list(dict.fromkeys(cap_sources))
    reason_parts = [
        meta.get("incomplete_reason") for meta in meta_list if meta.get("incomplete_reason")
    ]
    max_windows = max(
        [mw for mw in (m.get("max_windows") for m in meta_list) if mw is not None],
        default=None,
    )
    included = all(meta.get("included", True) for meta in meta_list)
    if total_windows is None or eval_windows is None:
        included = False
        reason_parts.append("missing_window_counts")
    if cap_active:
        included = False
    if coverage is not None and coverage < 0.9999:
        included = False
        reason_parts.append("window_coverage_lt_1")
    reason = ";".join(dict.fromkeys([r for r in reason_parts if r])) or None
    return {
        "windows_total": total_windows,
        "windows_evaluated": eval_windows,
        "window_coverage": coverage,
        "max_windows": max_windows,
        "cap_active": cap_active,
        "cap_sources": cap_sources,
        "incomplete_reason": reason,
        "included": included,
    }

REGIMES: Sequence[str] = ("full", "calm", "crisis")
PORTFOLIOS: Sequence[str] = ("ew", "mv")


@dataclass(frozen=True, slots=True)
class SummaryArtifacts:
    performance: pd.DataFrame
    detection: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _normalise(series: pd.Series, value: str) -> pd.Series:
    if series.empty:
        return series
    return series.astype(str).str.strip().str.lower() == value


def _pick_row(df: pd.DataFrame, *, regime: str | None, estimator: str | None, portfolio: str | None) -> pd.Series | None:
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    if regime is not None and "regime" in df.columns:
        mask &= _normalise(df["regime"], regime)
    if estimator is not None and "estimator" in df.columns:
        mask &= _normalise(df["estimator"], estimator)
    if portfolio is not None and "portfolio" in df.columns:
        mask &= _normalise(df["portfolio"], portfolio)
    subset = df[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _pick_dm_row(df: pd.DataFrame, *, regime: str, portfolio: str, baseline: str = "baseline") -> pd.Series | None:
    if df.empty:
        return None
    mask = _normalise(df["portfolio"], portfolio)
    if "regime" in df.columns:
        mask &= _normalise(df["regime"], regime)
    if "baseline" in df.columns and baseline:
        mask &= _normalise(df["baseline"], baseline)
    subset = df[mask]
    if subset.empty:
        return None
    return subset.iloc[0]


def _aggregate_row(df: pd.DataFrame, *, regime: str | None, estimator: str | None, portfolio: str | None) -> pd.Series | None:
    """Aggregate matching rows (mean of numeric columns, first of non-numeric)."""
    if df.empty:
        return None
    mask = pd.Series(True, index=df.index)
    if regime is not None and "regime" in df.columns:
        mask &= _normalise(df["regime"], regime)
    if estimator is not None and "estimator" in df.columns:
        mask &= _normalise(df["estimator"], estimator)
    if portfolio is not None and "portfolio" in df.columns:
        mask &= _normalise(df["portfolio"], portfolio)
    subset = df[mask]
    if subset.empty:
        return None
    base = subset.iloc[0].copy()
    numeric_cols = subset.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        base[col] = pd.to_numeric(subset[col], errors="coerce").mean()
    return base


def _aggregate_dm_row(df: pd.DataFrame, *, regime: str, portfolio: str, baseline: str = "baseline") -> pd.Series | None:
    if df.empty:
        return None
    mask = _normalise(df["portfolio"], portfolio)
    if "regime" in df.columns:
        mask &= _normalise(df["regime"], regime)
    if "baseline" in df.columns and baseline:
        mask &= _normalise(df["baseline"], baseline)
    subset = df[mask]
    if subset.empty:
        return None
    base = subset.iloc[0].copy()
    numeric_cols = subset.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        base[col] = pd.to_numeric(subset[col], errors="coerce").mean()
    return base


def _nan_median(series: pd.Series) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return float("nan")
    return float(np.median(cleaned))


def _nan_quantile(series: pd.Series, q: float) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return float("nan")
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(cleaned, q))


def _count_nonzero(series: pd.Series) -> int:
    if series.empty:
        return 0
    cleaned = pd.to_numeric(series, errors="coerce")
    return int((cleaned > 0).sum())


def _concat_if_exists(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = _read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_diag_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg: dict[str, Any] = {}
    for col in numeric_cols:
        agg[col] = pd.to_numeric(df[col], errors="coerce").mean()
    for col in ("reason_code", "resolved_config_path", "gating_mode"):
        if col in df:
            series = df[col].dropna().astype(str)
            if not series.empty:
                agg[col] = series.mode().iloc[0]
    return pd.Series(agg)


def _numeric(series: pd.Series, key: str) -> float:
    if key not in series:
        return float("nan")
    try:
        return float(series[key])
    except (TypeError, ValueError):
        return float("nan")


def _string(series: pd.Series, key: str, default: str = "") -> str:
    if key not in series or pd.isna(series[key]):
        return default
    return str(series[key])


def _load_detail(rc_dir: Path, regime: str, root_detail: pd.DataFrame) -> pd.DataFrame:
    regime_detail = _read_csv(rc_dir / regime / "diagnostics_detail.csv")
    if not regime_detail.empty:
        return regime_detail
    if root_detail.empty:
        return pd.DataFrame()
    data = root_detail.copy()
    if "regime" not in data.columns:
        return data
    mask = _normalise(data["regime"], regime)
    filtered = data[mask]
    return filtered.reset_index(drop=True)


def _row_for(perf_df: pd.DataFrame, regime: str, portfolio: str | None = None) -> pd.Series:
    if perf_df.empty:
        return pd.Series(dtype=float)
    mask = perf_df["regime"].astype(str).str.lower().eq(regime.lower())
    if portfolio is not None and "portfolio" in perf_df.columns:
        mask &= perf_df["portfolio"].astype(str).str.lower().eq(portfolio.lower())
    subset = perf_df.loc[mask]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.iloc[0]


def _criterion_entry(
    key: str,
    label: str,
    value: float | str | None,
    passed: bool | None,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "value": value,
        "pass": passed,
        "threshold": threshold,
    }


def _evaluate_kill_criteria(
    perf_df: pd.DataFrame,
    det_df: pd.DataFrame,
    rc_run: str,
    regime: str = "full",
) -> tuple[dict[str, Any], list[str]]:
    if "included_in_summary" in perf_df.columns:
        perf_df = perf_df[perf_df["included_in_summary"] != False]
    if "included_in_summary" in det_df.columns:
        det_df = det_df[det_df["included_in_summary"] != False]
    ew_row = _row_for(perf_df, regime, "ew")
    mv_row = _row_for(perf_df, regime, "mv")
    det_row = _row_for(det_df, regime)

    results: list[dict[str, Any]] = []
    limitations: list[str] = []

    def add_numeric_criterion(
        key: str,
        label: str,
        value: float,
        predicate: Callable[[float], bool | None],
        threshold: Any,
        formatter: Callable[[float], str] | None = None,
    ) -> None:
        nonlocal results, limitations
        if np.isnan(value):
            status = None
        else:
            status = predicate(value)
        results.append(_criterion_entry(key, label, value, status, threshold))
        if status is False:
            display = formatter(value) if formatter else f"{value:.3g}"
            limitations.append(f"{label}: observed {display} vs threshold {threshold}.")
        elif status is None:
            limitations.append(f"{label}: value unavailable.")

    ew_delta = _numeric(ew_row, "delta_mse_vs_baseline")
    mv_delta = _numeric(mv_row, "delta_mse_vs_baseline")
    add_numeric_criterion(
        "delta_mse_ew",
        "EW ΔMSE must not exceed baseline",
        ew_delta,
        lambda x: x <= 0.0,
        {"max": 0.0},
    )
    add_numeric_criterion(
        "delta_mse_mv",
        "MV ΔMSE must not exceed baseline",
        mv_delta,
        lambda x: x <= 0.0,
        {"max": 0.0},
    )

    det_rate = _numeric(det_row, "detection_rate_mean")
    rate_bounds = {"min": 0.01, "max": 0.25}
    add_numeric_criterion(
        "detection_rate",
        "Detection coverage within target band",
        det_rate,
        lambda x: rate_bounds["min"] <= x <= rate_bounds["max"],
        rate_bounds,
    )

    edge_margin = _numeric(det_row, "edge_margin_mean")
    add_numeric_criterion(
        "edge_margin",
        "Average edge margin positive",
        edge_margin,
        lambda x: x > 0.0,
        {"min": 0.0},
    )

    alignment_cos = _numeric(det_row, "alignment_cos_mean")
    add_numeric_criterion(
        "alignment_cos",
        "Alignment cosine above 0.9",
        alignment_cos,
        lambda x: x >= 0.9,
        {"min": 0.9},
    )

    reason_code = str(det_row.get("reason_code", "")) if not det_row.empty else ""
    allowed_reasons = {"", "accepted"}
    status = True if reason_code in allowed_reasons else False if reason_code else None
    results.append(
        _criterion_entry(
            "reason_code",
            "Dominant reason code acceptable",
            reason_code or None,
            status,
            {"allowed": sorted(allowed_reasons - {""})},
        )
    )
    if status is False:
        limitations.append(f"Reason-code mode '{reason_code}' signals gating issues.")
    elif status is None:
        limitations.append("Reason-code mode unavailable.")

    kill_payload = {
        "rc_run": rc_run,
        "regime": regime,
        "criteria": results,
    }
    return kill_payload, limitations


def summarise_rc_directory(rc_dir: Path) -> SummaryArtifacts:
    if not rc_dir.exists() or not rc_dir.is_dir():
        raise ValueError(f"RC directory '{rc_dir}' does not exist or is not a directory.")

    root_detail = _read_csv(rc_dir / "diagnostics_detail.csv")
    design_dirs = [child for child in rc_dir.iterdir() if child.is_dir() and child.name != "summary"]
    perf_records: list[dict[str, object]] = []
    det_records: list[dict[str, object]] = []
    meta_logged = False

    for regime in REGIMES:
        metrics_path = rc_dir / regime / "metrics.csv"
        dm_path = rc_dir / regime / "dm.csv"
        diag_path = rc_dir / regime / "diagnostics.csv"

        use_design_dirs = not metrics_path.exists() and any((d / regime).exists() for d in design_dirs)
        window_meta = _aggregate_window_meta(
            (_window_meta(d) for d in design_dirs) if use_design_dirs else (_window_meta(rc_dir),)
        )
        if not window_meta["included"] and not meta_logged:
            print(
                f"[make_summary] excluding {rc_dir.name} metrics from aggregates: {window_meta['incomplete_reason']}",
                file=sys.stderr,
            )
            meta_logged = True

        if use_design_dirs:
            metrics_df = _concat_if_exists(d / regime / "metrics.csv" for d in design_dirs)
            dm_df = _concat_if_exists(d / regime / "dm.csv" for d in design_dirs)
            diag_df = _concat_if_exists(d / regime / "diagnostics.csv" for d in design_dirs)
            detail_df = _concat_if_exists(d / regime / "diagnostics_detail.csv" for d in design_dirs)
        else:
            metrics_df = _read_csv(metrics_path)
            dm_df = _read_csv(dm_path)
            diag_df = _read_csv(diag_path)
            detail_df = _load_detail(rc_dir, regime, root_detail)

        diag_row = _aggregate_diag_row(diag_df)

        for portfolio in PORTFOLIOS:
            overlay_row = _aggregate_row(metrics_df, regime=regime, estimator="overlay", portfolio=portfolio)
            baseline_row = _aggregate_row(metrics_df, regime=regime, estimator="baseline", portfolio=portfolio)
            dm_row = _aggregate_dm_row(dm_df, regime=regime, portfolio=portfolio)

            record = {
                "rc_run": rc_dir.name,
                "regime": regime,
                "portfolio": portfolio,
                "delta_mse_vs_baseline": float("nan"),
                "delta_mse_ci_lower": float("nan"),
                "delta_mse_ci_upper": float("nan"),
                "delta_es_vs_baseline": float("nan"),
                "var95_overlay": float("nan"),
                "var95_baseline": float("nan"),
                "es95_overlay": float("nan"),
                "es95_baseline": float("nan"),
                "realised_var_overlay": float("nan"),
                "realised_var_baseline": float("nan"),
                "realised_es_overlay": float("nan"),
                "realised_es_baseline": float("nan"),
                "dm_stat": float("nan"),
                "dm_p_value": float("nan"),
                "n_effective": float("nan"),
            }

            if overlay_row is not None:
                record.update(
                    {
                        "delta_mse_vs_baseline": _numeric(overlay_row, "delta_mse_vs_baseline"),
                        "delta_mse_ci_lower": _numeric(overlay_row, "delta_mse_ci_lower"),
                        "delta_mse_ci_upper": _numeric(overlay_row, "delta_mse_ci_upper"),
                        "delta_es_vs_baseline": _numeric(overlay_row, "delta_es_vs_baseline"),
                        "var95_overlay": _numeric(overlay_row, "var95"),
                        "es95_overlay": _numeric(overlay_row, "es95"),
                        "realised_var_overlay": _numeric(overlay_row, "realised_var"),
                        "realised_es_overlay": _numeric(overlay_row, "realised_es"),
                    }
                )
            if baseline_row is not None:
                record.update(
                    {
                        "var95_baseline": _numeric(baseline_row, "var95"),
                        "es95_baseline": _numeric(baseline_row, "es95"),
                        "realised_var_baseline": _numeric(baseline_row, "realised_var"),
                        "realised_es_baseline": _numeric(baseline_row, "realised_es"),
                    }
                )
            if dm_row is not None:
                record.update(
                    {
                        "dm_stat": _numeric(dm_row, "dm_stat"),
                        "dm_p_value": _numeric(dm_row, "p_value"),
                        "n_effective": _numeric(dm_row, "n_effective"),
                    }
                )

            record.update(
                {
                    "windows_total": window_meta["windows_total"],
                    "windows_evaluated": window_meta["windows_evaluated"],
                    "window_coverage": window_meta["window_coverage"],
                    "max_windows": window_meta["max_windows"],
                    "cap_active": window_meta["cap_active"],
                    "cap_sources": ";".join(window_meta["cap_sources"]),
                    "incomplete_reason": window_meta["incomplete_reason"],
                    "included_in_summary": window_meta["included"],
                }
            )
            if not window_meta["included"]:
                for key in (
                    "delta_mse_vs_baseline",
                    "delta_mse_ci_lower",
                    "delta_mse_ci_upper",
                    "delta_es_vs_baseline",
                    "var95_overlay",
                    "var95_baseline",
                    "es95_overlay",
                    "es95_baseline",
                    "realised_var_overlay",
                    "realised_var_baseline",
                    "realised_es_overlay",
                    "realised_es_baseline",
                    "dm_stat",
                    "dm_p_value",
                    "n_effective",
                ):
                    record[key] = float("nan")

            perf_records.append(record)

        detail_windows = detail_df.copy()
        if not detail_windows.empty and "regime" in detail_windows.columns:
            # When detail comes from root, ensure regime filtering respected.
            mask = _normalise(detail_windows["regime"], regime)
            detail_windows = detail_windows[mask]

        windows = int(detail_windows.shape[0]) if not detail_windows.empty else 0
        detections_col = detail_windows["detections"] if "detections" in detail_windows else pd.Series(dtype=float)
        detection_rate_col = (
            detail_windows["detection_rate"] if "detection_rate" in detail_windows else pd.Series(dtype=float)
        )
        isolation_col = (
            detail_windows["isolation_share"] if "isolation_share" in detail_windows else pd.Series(dtype=float)
        )
        edge_col = (
            detail_windows["edge_margin_mean"] if "edge_margin_mean" in detail_windows else pd.Series(dtype=float)
        )
        stability_col = (
            detail_windows["stability_margin_mean"]
            if "stability_margin_mean" in detail_windows
            else pd.Series(dtype=float)
        )
        alignment_cos_col = (
            detail_windows["alignment_cos_mean"]
            if "alignment_cos_mean" in detail_windows
            else pd.Series(dtype=float)
        )
        alignment_angle_col = (
            detail_windows["alignment_angle_mean"]
            if "alignment_angle_mean" in detail_windows
            else pd.Series(dtype=float)
        )

        det_record = {
            "rc_run": rc_dir.name,
            "regime": regime,
            "windows": windows,
            "detection_windows": _count_nonzero(detections_col),
            "detections_mean": _numeric(diag_row, "detections"),
            "detection_rate_mean": _numeric(diag_row, "detection_rate"),
            "detection_rate_median": _nan_median(detection_rate_col),
            "isolation_share_mean": _numeric(diag_row, "isolation_share"),
            "isolation_share_median": _nan_median(isolation_col),
            "edge_margin_mean": _numeric(diag_row, "edge_margin_mean"),
            "edge_margin_median": _nan_median(edge_col),
            "edge_margin_p10": _nan_quantile(edge_col, 0.10),
            "edge_margin_p90": _nan_quantile(edge_col, 0.90),
            "stability_margin_mean": _numeric(diag_row, "stability_margin_mean"),
            "stability_margin_median": _nan_median(stability_col),
            "stability_margin_p10": _nan_quantile(stability_col, 0.10),
            "stability_margin_p90": _nan_quantile(stability_col, 0.90),
            "isolation_share_p10": _nan_quantile(isolation_col, 0.10),
            "isolation_share_p90": _nan_quantile(isolation_col, 0.90),
            "alignment_cos_mean": _numeric(diag_row, "alignment_cos_mean"),
            "alignment_cos_median": _nan_median(alignment_cos_col),
            "alignment_cos_p10": _nan_quantile(alignment_cos_col, 0.10),
            "alignment_cos_p90": _nan_quantile(alignment_cos_col, 0.90),
            "alignment_angle_mean": _numeric(diag_row, "alignment_angle_mean"),
            "alignment_angle_median": _nan_median(alignment_angle_col),
            "reason_code_mode": _string(diag_row, "reason_code"),
            "calm_threshold_mean": _numeric(diag_row, "calm_threshold"),
            "crisis_threshold_mean": _numeric(diag_row, "crisis_threshold"),
            "vol_signal_mean": _numeric(diag_row, "vol_signal"),
            "resolved_config_path": _string(diag_row, "resolved_config_path"),
        }
        det_record.update(
            {
                "windows_total": window_meta["windows_total"],
                "windows_evaluated": window_meta["windows_evaluated"],
                "window_coverage": window_meta["window_coverage"],
                "max_windows": window_meta["max_windows"],
                "cap_active": window_meta["cap_active"],
                "cap_sources": ";".join(window_meta["cap_sources"]),
                "incomplete_reason": window_meta["incomplete_reason"],
                "included_in_summary": window_meta["included"],
            }
        )
        if not window_meta["included"]:
            for key in (
                "windows",
                "detection_windows",
                "detections_mean",
                "detection_rate_mean",
                "detection_rate_median",
                "isolation_share_mean",
                "isolation_share_median",
                "edge_margin_mean",
                "edge_margin_median",
                "edge_margin_p10",
                "edge_margin_p90",
                "stability_margin_mean",
                "stability_margin_median",
                "stability_margin_p10",
                "stability_margin_p90",
                "isolation_share_p10",
                "isolation_share_p90",
                "alignment_cos_mean",
                "alignment_cos_median",
                "alignment_cos_p10",
                "alignment_cos_p90",
                "alignment_angle_mean",
                "alignment_angle_median",
                "reason_code_mode",
                "calm_threshold_mean",
                "crisis_threshold_mean",
                "vol_signal_mean",
            ):
                det_record[key] = float("nan")
        det_records.append(det_record)

    perf_df = pd.DataFrame(perf_records)
    det_df = pd.DataFrame(det_records)
    return SummaryArtifacts(performance=perf_df, detection=det_df)


def _discover_rc_dirs(root: Path, patterns: Iterable[str] | None, all_runs: bool, rc_dir: Path | None) -> list[Path]:
    if rc_dir is not None:
        return [rc_dir.resolve()]

    candidates: list[Path] = []
    if patterns:
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
    else:
        candidates.extend(child for child in root.iterdir() if child.is_dir() and child.name.startswith("rc-"))

    resolved = sorted({path.resolve() for path in candidates if path.is_dir()})
    if not resolved:
        raise ValueError(f"No RC directories found under '{root}'.")
    if all_runs:
        return resolved
    return [resolved[-1]]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def write_summaries(rc_dirs: Iterable[Path]) -> dict[Path, SummaryArtifacts]:
    outputs: dict[Path, SummaryArtifacts] = {}
    for directory in rc_dirs:
        artifacts = summarise_rc_directory(directory)
        summary_dir = directory / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        perf_path = summary_dir / "summary_perf.csv"
        det_path = summary_dir / "summary_detection.csv"
        artifacts.performance.to_csv(perf_path, index=False)
        artifacts.detection.to_csv(det_path, index=False)
        outputs[directory] = artifacts
        print(f"[make_summary] Wrote {_display_path(perf_path)}")
        print(f"[make_summary] Wrote {_display_path(det_path)}")

        kill_data, limitations = _evaluate_kill_criteria(
            artifacts.performance, artifacts.detection, directory.name
        )
        kill_path = summary_dir / "kill_criteria.json"
        kill_path.write_text(json.dumps(kill_data, indent=2, sort_keys=True), encoding="utf-8")
        if limitations:
            deduped = list(dict.fromkeys(limitations))
            text = "\n".join(f"- {item}" for item in deduped)
        else:
            text = "No critical limitations detected under current criteria."
        limitations_path = summary_dir / "limitations.md"
        limitations_path.write_text(text, encoding="utf-8")
        print(f"[make_summary] Wrote {_display_path(kill_path)}")
        print(f"[make_summary] Wrote {_display_path(limitations_path)}")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate RC evaluation artifacts into summary tables.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reports"),
        help="Root directory containing rc-* runs (default: reports).",
    )
    parser.add_argument(
        "--rc-dir",
        type=Path,
        default=None,
        help="Explicit RC directory to summarise (overrides pattern discovery).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Optional glob pattern(s) relative to --root for selecting RC directories.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all matching RC directories instead of just the latest.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.root.exists():
        raise ValueError(f"Root directory '{args.root}' does not exist.")
    rc_dirs = _discover_rc_dirs(args.root.resolve(), args.pattern, args.all, args.rc_dir)
    write_summaries(rc_dirs)


if __name__ == "__main__":  # pragma: no cover
    main()
