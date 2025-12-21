#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from meta.completeness import CompletenessResult, evaluate_eval_run

REGIMES: Sequence[str] = ("full", "calm", "crisis")
PORTFOLIOS: Sequence[str] = ("ew", "mv")
OVERLAY_FORENSICS_COLUMNS: Sequence[str] = (
    "window_end",
    "window_id",
    "regime",
    "portfolio",
    "design",
    "shrinker",
    "edge_mode",
    "changed",
    "skip_reason_primary",
    "skip_reason_detail",
    "gate_mode",
    "delta_frac_used",
    "lambda1_base",
    "lambda1_treat",
    "delta_lambda1",
    "mp_edge",
    "edge_margin",
    "realized_var",
    "mse_base",
    "mse_treat",
    "qlike_base",
    "qlike_treat",
)
PERF_COLUMNS: Sequence[str] = (
    "rc_run",
    "regime",
    "portfolio",
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
    "n_effective_mse",
    "n_effective_es",
    "n_effective_qlike",
    "comparison_valid_dm",
    "comparison_valid_delta",
    "cap_active",
    "cap_sources",
    "window_coverage",
)
DET_COLUMNS: Sequence[str] = (
    "rc_run",
    "regime",
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
    "resolved_config_path",
    "cap_active",
    "cap_sources",
    "window_coverage",
)
SKIP_COLUMNS: Sequence[str] = (
    "regime",
    "portfolio",
    "estimator",
    "skip_reason",
    "windows",
    "skip_count",
    "skip_share",
)


@dataclass(frozen=True, slots=True)
class SummaryArtifacts:
    performance: pd.DataFrame
    detection: pd.DataFrame
    skip_stats: pd.DataFrame
    completeness: CompletenessResult | None = None
    run_eligibility: tuple["RunEligibility", ...] = ()


@dataclass(frozen=True, slots=True)
class RunEligibility:
    run_dir: Path
    display_path: str
    completeness: CompletenessResult
    mv_skip_on_missing_solver: bool
    excluded_from_summary: bool


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _empty_perf_df() -> pd.DataFrame:
    return pd.DataFrame(columns=PERF_COLUMNS)


def _empty_det_df() -> pd.DataFrame:
    return pd.DataFrame(columns=DET_COLUMNS)


def _empty_skip_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SKIP_COLUMNS)


def _is_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "run.json").exists() or (path / "run_manifest.json").exists():
        return True
    return any((path / regime).is_dir() for regime in REGIMES)


def _discover_design_dirs(rc_dir: Path) -> list[Path]:
    candidates = [
        child
        for child in rc_dir.iterdir()
        if child.is_dir() and child.name != "summary" and _is_run_dir(child)
    ]
    return sorted(candidates)


def _read_mv_skip_on_missing_solver(run_dir: Path) -> bool:
    payload = _read_json(run_dir / "run.json")
    config = payload.get("config")
    if isinstance(config, dict) and "mv_skip_on_missing_solver" in config:
        return bool(config.get("mv_skip_on_missing_solver"))
    return bool(payload.get("mv_skip_on_missing_solver", False))


def _aggregate_completeness(eligible_runs: Sequence[RunEligibility]) -> tuple[bool, list[str], float | None]:
    if not eligible_runs:
        return False, [], None
    cap_active = any(run.completeness.cap_active for run in eligible_runs)
    cap_sources = sorted({src for run in eligible_runs for src in run.completeness.cap_sources})
    coverages = [
        run.completeness.window_coverage
        for run in eligible_runs
        if run.completeness.window_coverage is not None
    ]
    coverage = min(coverages) if coverages else None
    return cap_active, cap_sources, coverage


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


def _series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(_series_or_default(df, column, np.nan), errors="coerce")


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


def _load_resolved_config(design_dir: Path, detail_df: pd.DataFrame) -> tuple[dict[str, Any], Path | None]:
    config_path = design_dir / "resolved_config.json"
    if not config_path.exists() and "resolved_config_path" in detail_df.columns:
        candidates = detail_df["resolved_config_path"].dropna().astype(str)
        if not candidates.empty:
            config_path = Path(candidates.iloc[0])
    payload = _read_json(config_path) if config_path.exists() else {}
    return payload, config_path if config_path.exists() else None


def _overlay_forensics_for_design(design_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    metrics_path = design_dir / "metrics_detail.csv"
    diag_path = design_dir / "diagnostics_detail.csv"
    metrics_df = _read_csv(metrics_path)
    diag_df = _read_csv(diag_path)

    if not metrics_path.exists():
        warnings.append(f"overlay_forensics missing metrics_detail.csv in {design_dir.name}")
    elif metrics_df.empty:
        warnings.append(f"overlay_forensics metrics_detail.csv empty in {design_dir.name}")
    if not diag_path.exists():
        warnings.append(f"overlay_forensics missing diagnostics_detail.csv in {design_dir.name}")
    elif diag_df.empty:
        warnings.append(f"overlay_forensics diagnostics_detail.csv empty in {design_dir.name}")
    if metrics_df.empty or diag_df.empty:
        return pd.DataFrame(columns=OVERLAY_FORENSICS_COLUMNS), warnings

    if "window_id" not in metrics_df.columns or "regime" not in metrics_df.columns:
        warnings.append(f"overlay_forensics metrics_detail missing window_id/regime in {design_dir.name}")
        return pd.DataFrame(columns=OVERLAY_FORENSICS_COLUMNS), warnings
    if "window_id" not in diag_df.columns or "regime" not in diag_df.columns:
        warnings.append(f"overlay_forensics diagnostics_detail missing window_id/regime in {design_dir.name}")
        return pd.DataFrame(columns=OVERLAY_FORENSICS_COLUMNS), warnings

    metrics_df = metrics_df.copy()
    diag_df = diag_df.copy()
    metrics_df["regime"] = metrics_df["regime"].astype(str).str.strip().str.lower()
    diag_df["regime"] = diag_df["regime"].astype(str).str.strip().str.lower()
    metrics_df["window_id"] = pd.to_numeric(metrics_df["window_id"], errors="coerce")
    diag_df["window_id"] = pd.to_numeric(diag_df["window_id"], errors="coerce")
    metrics_df["portfolio"] = (
        _series_or_default(metrics_df, "portfolio", "").astype(str).str.strip().str.lower()
    )
    metrics_df["estimator_norm"] = (
        _series_or_default(metrics_df, "estimator", "").astype(str).str.strip().str.lower()
    )

    metrics_subset = metrics_df[metrics_df["estimator_norm"].isin({"overlay", "baseline"})]
    if metrics_subset.empty:
        warnings.append(f"overlay_forensics found no overlay/baseline rows in {design_dir.name}")
        return pd.DataFrame(columns=OVERLAY_FORENSICS_COLUMNS), warnings

    pivot = metrics_subset.pivot_table(
        index=["window_id", "regime", "portfolio"],
        columns="estimator_norm",
        values=["realised_var", "sq_error", "qlike"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{est}" for metric, est in pivot.columns]
    pivot = pivot.reset_index()

    config_payload, config_path = _load_resolved_config(design_dir, diag_df)
    shrinker = str(config_payload.get("shrinker") or "")
    edge_mode = str(config_payload.get("edge_mode") or "")
    design = str(config_payload.get("group_design") or "")
    if not design and "group_design" in diag_df.columns:
        design_series = diag_df["group_design"].dropna().astype(str).str.strip()
        if not design_series.empty:
            design = design_series.iloc[0]
    if not design:
        design = design_dir.name
    if not shrinker:
        warnings.append(f"overlay_forensics missing shrinker in {design_dir.name}")
    if not edge_mode:
        warnings.append(f"overlay_forensics missing edge_mode in {design_dir.name}")
    if config_path is None:
        warnings.append(f"overlay_forensics missing resolved_config.json in {design_dir.name}")

    diag_df["design"] = design
    diag_df["shrinker"] = shrinker
    diag_df["edge_mode"] = edge_mode
    if "window_start" in diag_df.columns:
        diag_df["window_end"] = diag_df["window_start"]
    else:
        diag_df["window_end"] = ""

    diag_df["changed"] = (
        pd.to_numeric(_series_or_default(diag_df, "changed_flag", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    diag_df["skip_reason_primary"] = (
        _series_or_default(diag_df, "reason_code", "").fillna("").astype(str)
    )
    detail_series = _series_or_default(diag_df, "baseline_errors", "").fillna("").astype(str)
    mv_skip = _series_or_default(diag_df, "mv_skip_reason", "").fillna("").astype(str)
    diag_df["skip_reason_detail"] = detail_series.where(detail_series != "", mv_skip)
    diag_df["gate_mode"] = _series_or_default(diag_df, "gating_mode", "").fillna("").astype(str)
    diag_df["delta_frac_used"] = _numeric_series(diag_df, "gating_delta_frac")
    diag_df["lambda1_base"] = _numeric_series(diag_df, "lambda1_base")
    diag_df["lambda1_treat"] = _numeric_series(diag_df, "lambda1_treat")
    diag_df["delta_lambda1"] = _numeric_series(diag_df, "delta_lambda1")
    diag_df["mp_edge"] = _numeric_series(diag_df, "mp_edge")
    mp_edge_margin = _numeric_series(diag_df, "mp_edge_margin")
    edge_mean = _numeric_series(diag_df, "edge_margin_mean")
    diag_df["edge_margin"] = mp_edge_margin.where(mp_edge_margin.notna(), edge_mean)

    merged = pivot.merge(diag_df, on=["window_id", "regime"], how="left")

    realised_base = _numeric_series(merged, "realised_var_baseline")
    realised_treat = _numeric_series(merged, "realised_var_overlay")
    merged["realized_var"] = realised_treat.where(realised_treat.notna(), realised_base)
    merged["mse_base"] = _numeric_series(merged, "sq_error_baseline")
    merged["mse_treat"] = _numeric_series(merged, "sq_error_overlay")
    merged["qlike_base"] = _numeric_series(merged, "qlike_baseline")
    merged["qlike_treat"] = _numeric_series(merged, "qlike_overlay")

    if "window_end" not in merged.columns:
        merged["window_end"] = ""

    filtered = merged[merged["changed"].fillna(0).astype(int) == 1].copy()
    string_cols = {
        "window_end",
        "regime",
        "portfolio",
        "design",
        "shrinker",
        "edge_mode",
        "skip_reason_primary",
        "skip_reason_detail",
        "gate_mode",
    }
    for col in OVERLAY_FORENSICS_COLUMNS:
        if col not in filtered.columns:
            if col in string_cols:
                filtered[col] = ""
            else:
                filtered[col] = np.nan
    filtered = filtered[list(OVERLAY_FORENSICS_COLUMNS)]
    sort_cols = [col for col in ["design", "edge_mode", "shrinker", "regime", "window_id", "portfolio"] if col in filtered.columns]
    if sort_cols and not filtered.empty:
        filtered = filtered.sort_values(sort_cols, kind="mergesort")
    return filtered, warnings


def _build_overlay_forensics(rc_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    design_dirs = [child for child in rc_dir.iterdir() if child.is_dir() and child.name != "summary"]
    if (rc_dir / "metrics_detail.csv").exists() or (rc_dir / "diagnostics_detail.csv").exists():
        design_dirs = [rc_dir]
    frames: list[pd.DataFrame] = []
    for design_dir in design_dirs:
        frame, warn = _overlay_forensics_for_design(design_dir)
        warnings.extend(warn)
        if not frame.empty:
            frames.append(frame)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=OVERLAY_FORENSICS_COLUMNS)
    if not combined.empty:
        combined = combined[list(OVERLAY_FORENSICS_COLUMNS)]
    else:
        combined = combined.reindex(columns=list(OVERLAY_FORENSICS_COLUMNS))
    sort_cols = [col for col in ["design", "edge_mode", "shrinker", "regime", "window_id", "portfolio"] if col in combined.columns]
    if sort_cols and not combined.empty:
        combined = combined.sort_values(sort_cols, kind="mergesort")
    return combined, warnings


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

    completeness = evaluate_eval_run(
        rc_dir,
        label=rc_dir.name,
        require_manifest=False,
        allow_unknown_coverage=True,
        run_type="rc",
    )
    root_detail = _read_csv(rc_dir / "diagnostics_detail.csv")
    design_dirs = _discover_design_dirs(rc_dir)
    root_has_metrics = any((rc_dir / regime / "metrics.csv").exists() for regime in REGIMES)
    use_design_dirs = (not root_has_metrics) and bool(design_dirs)
    run_dirs = design_dirs if use_design_dirs else [rc_dir]
    run_eligibility: list[RunEligibility] = []
    for run_dir in run_dirs:
        run_comp = evaluate_eval_run(
            run_dir,
            label=run_dir.name,
            require_manifest=False,
            allow_unknown_coverage=True,
            run_type="rc",
        )
        mv_skip = _read_mv_skip_on_missing_solver(run_dir)
        excluded = run_comp.cap_active or mv_skip
        run_eligibility.append(
            RunEligibility(
                run_dir=run_dir,
                display_path=_display_path(run_dir),
                completeness=run_comp,
                mv_skip_on_missing_solver=mv_skip,
                excluded_from_summary=excluded,
            )
        )
    eligible_runs = [run for run in run_eligibility if not run.excluded_from_summary]
    eligible_dirs = [run.run_dir for run in eligible_runs]
    if not eligible_dirs:
        return SummaryArtifacts(
            performance=_empty_perf_df(),
            detection=_empty_det_df(),
            skip_stats=_empty_skip_df(),
            completeness=completeness,
            run_eligibility=tuple(run_eligibility),
        )
    if use_design_dirs:
        summary_cap_active, summary_cap_sources, summary_window_coverage = _aggregate_completeness(
            eligible_runs
        )
    else:
        run_comp = run_eligibility[0].completeness if run_eligibility else completeness
        summary_cap_active = bool(run_comp.cap_active) if run_comp else False
        summary_cap_sources = list(run_comp.cap_sources) if run_comp else []
        summary_window_coverage = run_comp.window_coverage if run_comp else None

    data_dirs = eligible_dirs if use_design_dirs else [rc_dir]
    perf_records: list[dict[str, object]] = []
    det_records: list[dict[str, object]] = []
    skip_frames: list[pd.DataFrame] = []

    for regime in REGIMES:
        metrics_path = rc_dir / regime / "metrics.csv"
        dm_path = rc_dir / regime / "dm.csv"
        diag_path = rc_dir / regime / "diagnostics.csv"
        skip_path = rc_dir / regime / "skip_stats.csv"

        use_design_dirs_for_regime = not metrics_path.exists() and any((d / regime).exists() for d in data_dirs)

        if use_design_dirs_for_regime:
            metrics_df = _concat_if_exists(d / regime / "metrics.csv" for d in data_dirs)
            dm_df = _concat_if_exists(d / regime / "dm.csv" for d in data_dirs)
            diag_df = _concat_if_exists(d / regime / "diagnostics.csv" for d in data_dirs)
            detail_df = _concat_if_exists(d / regime / "diagnostics_detail.csv" for d in data_dirs)
            skip_df = _concat_if_exists(d / regime / "skip_stats.csv" for d in data_dirs)
        else:
            metrics_df = _read_csv(metrics_path)
            dm_df = _read_csv(dm_path)
            diag_df = _read_csv(diag_path)
            skip_df = _read_csv(skip_path)
            detail_df = _load_detail(rc_dir, regime, root_detail)

        diag_row = _aggregate_diag_row(diag_df)
        if not skip_df.empty:
            skip_df["regime"] = skip_df.get("regime", regime)
            skip_frames.append(skip_df)

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
                "n_effective_mse": float("nan"),
                "n_effective_es": float("nan"),
                "n_effective_qlike": float("nan"),
                "comparison_valid_dm": float("nan"),
                "comparison_valid_delta": float("nan"),
                "cap_active": summary_cap_active,
                "cap_sources": ",".join(summary_cap_sources),
                "window_coverage": summary_window_coverage,
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
                        "n_effective_mse": _numeric(overlay_row, "n_effective_mse"),
                        "n_effective_es": _numeric(overlay_row, "n_effective_es"),
                        "n_effective_qlike": _numeric(overlay_row, "n_effective_qlike"),
                        "comparison_valid_delta": _numeric(overlay_row, "comparison_valid"),
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
                        "comparison_valid_dm": _numeric(dm_row, "comparison_valid"),
                    }
                )

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
            "cap_active": summary_cap_active,
            "cap_sources": ",".join(summary_cap_sources),
            "window_coverage": summary_window_coverage,
        }
        det_records.append(det_record)

    perf_df = pd.DataFrame(perf_records)
    det_df = pd.DataFrame(det_records)
    skip_df_all = pd.concat(skip_frames, ignore_index=True) if skip_frames else pd.DataFrame()
    if perf_df.empty:
        perf_df = _empty_perf_df()
    if det_df.empty:
        det_df = _empty_det_df()
    if skip_df_all.empty:
        skip_df_all = _empty_skip_df()
    return SummaryArtifacts(
        performance=perf_df,
        detection=det_df,
        skip_stats=skip_df_all,
        completeness=completeness,
        run_eligibility=tuple(run_eligibility),
    )


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
        skip_path = summary_dir / "summary_skip_stats.csv"
        artifacts.performance.to_csv(perf_path, index=False)
        artifacts.detection.to_csv(det_path, index=False)
        artifacts.skip_stats.to_csv(skip_path, index=False)
        outputs[directory] = artifacts
        print(f"[make_summary] Wrote {_display_path(perf_path)}")
        print(f"[make_summary] Wrote {_display_path(det_path)}")
        print(f"[make_summary] Wrote {_display_path(skip_path)}")

        overlay_df, overlay_warnings = _build_overlay_forensics(directory)
        overlay_path = summary_dir / "overlay_forensics.csv"
        overlay_df.to_csv(overlay_path, index=False)
        print(f"[make_summary] Wrote {_display_path(overlay_path)}")

        kill_data, limitations = _evaluate_kill_criteria(
            artifacts.performance, artifacts.detection, directory.name
        )
        comp = artifacts.completeness
        run_eligibility = artifacts.run_eligibility
        single_run = (
            comp is not None
            and len(run_eligibility) == 1
            and run_eligibility[0].run_dir.resolve() == directory.resolve()
        )
        if comp is not None:
            kill_data["completeness"] = comp.as_dict()
            if single_run:
                if not comp.is_complete:
                    limitations.append(comp.incomplete_reason or "run marked incomplete")
                if comp.window_coverage is not None and comp.window_coverage < 1.0:
                    limitations.append(
                        f"window coverage {comp.window_coverage:.3g} < 1.0; excluded from aggregates."
                    )
                if comp.cap_active:
                    reason = ", ".join(comp.cap_sources) if comp.cap_sources else "cap_active=true"
                    limitations.append(f"run capped ({reason}); excluded from aggregates.")
        limitations.extend(overlay_warnings)
        if overlay_df.empty:
            limitations.append("overlay_forensics.csv empty or missing changed windows (see diagnostics_detail.csv).")
        limitations.append(
            "Overlay forensics: see summary/overlay_forensics.csv for changed-window diagnostics and loss deltas."
        )
        invalid_rows = artifacts.performance[
            (artifacts.performance.get("comparison_valid_delta") == 0)
            | (artifacts.performance.get("comparison_valid_dm") == 0)
        ]
        if not invalid_rows.empty:
            limitations.append(
                "Some comparisons marked invalid due to insufficient aligned windows (see summary_perf.csv)."
            )

        kill_path = summary_dir / "kill_criteria.json"
        kill_path.write_text(json.dumps(kill_data, indent=2, sort_keys=True), encoding="utf-8")
        limitation_lines: list[str] = []
        capped_runs = [run for run in run_eligibility if run.completeness.cap_active]
        if capped_runs:
            limitation_lines.append("## Excluded smoke-only runs (capped)")
            for run in capped_runs:
                cap_note = ", ".join(run.completeness.cap_sources) if run.completeness.cap_sources else "cap_active=true"
                limitation_lines.append(f"- {run.display_path} (cap_sources: {cap_note})")
        mv_skip_runs = [run for run in run_eligibility if run.mv_skip_on_missing_solver]
        if mv_skip_runs:
            if limitation_lines:
                limitation_lines.append("")
            limitation_lines.append("## Smoke-only: MV skip-on-missing-solver enabled")
            for run in mv_skip_runs:
                suffix = "excluded from headline summaries" if run.excluded_from_summary else "not excluded"
                limitation_lines.append(f"- {run.display_path} ({suffix})")
        if limitations:
            deduped = list(dict.fromkeys(limitations))
            if limitation_lines:
                limitation_lines.append("")
            limitation_lines.append("## Other limitations")
            limitation_lines.extend(f"- {item}" for item in deduped)
        if not limitation_lines:
            text = "No critical limitations detected under current criteria."
        else:
            text = "\n".join(limitation_lines)
        limitations_path = summary_dir / "limitations.md"
        limitations_path.write_text(text, encoding="utf-8")
        print(f"[make_summary] Wrote {_display_path(kill_path)}")
        print(f"[make_summary] Wrote {_display_path(limitations_path)}")

        if comp is not None:
            comp_path = summary_dir / "completeness.json"
            comp_path.write_text(json.dumps(comp.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
            print(f"[make_summary] Wrote {_display_path(comp_path)}")
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
