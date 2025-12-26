#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import platform
import pstats
import sys
import time
from collections import Counter
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from eval.balance import build_balanced_window
from eval.clean import apply_nan_policy
from experiments.daily.grouping import GroupingError
from experiments.eval.config import DEFAULT_THRESHOLDS_PATH, ResolveResult, resolve_eval_config
import experiments.eval.run as eval_run
from fjs.overlay import OverlayConfig, detect_spikes


WINDOW_DETAIL_REQUIRED_COLUMNS = [
    "window_idx",
    "fit_start",
    "fit_end",
    "horizon_start",
    "horizon_end",
    "n_obs",
    "n_assets",
    "injected",
    "injected_mu",
    "detected_initial",
    "accepted",
]

PRE_GATE_COLUMNS = [
    "pre_gate_raw_outliers_found",
    "pre_gate_mp_edge_margin",
    "pre_gate_leakage_offcomp",
    "pre_gate_stability_eta_pass",
    "pre_gate_bracket_status",
    "pre_gate_coarse_candidates",
]

GATING_COLUMNS = [
    "gating_mode",
    "gating_rejected",
    "gating_soft_cap",
    "gating_delta_frac_used",
    "gating_capped",
]

GUARD_KEYS = [
    "edge_buffer",
    "stability_fail",
    "off_component_ratio",
    "energy_floor",
    "neg_mu",
    "eps",
    "tvec_compute_error",
    "tvec_no_real_root",
    "tvec_no_admissible_root",
    "tvec_singularity",
    "tvec_target_zero",
    "tvec_off_component",
    "mu_nonfinite",
]

GATE_REASON_KEYS = [
    "accepted",
    "inadmissible_root",
    "edge_margin",
    "nonisolated",
    "stability",
    "alignment",
    "delta_frac_min",
    "delta_frac_max",
    "soft_cap",
    "hard_cap",
]

THREAD_ENV_KEYS = [
    "EXEC_MODE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]


@dataclass(frozen=True)
class WindowSample:
    window_id: int
    matrix: np.ndarray
    group_labels: np.ndarray
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    hold_start: pd.Timestamp
    hold_end: pd.Timestamp


@dataclass(frozen=True)
class InjectionBasis:
    direction: np.ndarray
    series: np.ndarray


def _parse_float_list(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            values.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"invalid {name} value '{token}'") from exc
    if not values:
        raise ValueError(f"{name} grid must contain at least one value")
    return values


def _normalise_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        vec = np.ones_like(vec, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
    return vec / norm


def _standardise_series(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    series = series - float(np.mean(series))
    std = float(np.std(series))
    if not np.isfinite(std) or std <= 0.0:
        return np.ones_like(series, dtype=np.float64)
    return series / std


def _make_injection_basis(matrix: np.ndarray, rng: np.random.Generator) -> InjectionBasis:
    n_obs, n_assets = matrix.shape
    direction = _normalise_vector(rng.normal(size=n_assets).astype(np.float64))
    series = _standardise_series(rng.normal(size=n_obs))
    return InjectionBasis(direction=direction, series=series)


def _apply_injection(matrix: np.ndarray, basis: InjectionBasis, mu: float) -> np.ndarray:
    if mu < 0.0:
        raise ValueError("mu must be non-negative for injection.")
    scale = math.sqrt(mu)
    injection = np.outer(basis.series, basis.direction) * scale
    return matrix + injection


def _window_detection_stats(
    matrix: np.ndarray,
    labels: np.ndarray,
    overlay_cfg: OverlayConfig,
) -> tuple[int, int, dict[str, Any]]:
    stats: dict[str, Any] = {}
    detections = detect_spikes(matrix, labels, config=overlay_cfg, stats=stats)
    gating = stats.get("gating", {})
    initial = gating.get("initial")
    accepted = gating.get("accepted")
    if initial is None:
        initial = len(detections)
    if accepted is None:
        accepted = len(detections)
    return int(initial), int(accepted), stats


def _score_samples(
    samples: Sequence[WindowSample],
    overlay_cfg: OverlayConfig,
    *,
    bases: Sequence[InjectionBasis] | None = None,
    indices: Sequence[int] | None = None,
    mu: float | None = None,
) -> dict[str, float]:
    if indices is None:
        indices = list(range(len(samples)))
    n_windows = int(len(indices))
    n_detected = 0
    n_accepted = 0
    for idx in indices:
        sample = samples[int(idx)]
        matrix = sample.matrix
        if mu is not None and bases is not None:
            matrix = _apply_injection(matrix, bases[int(idx)], mu)
        detected_count, accepted_count, _ = _window_detection_stats(matrix, sample.group_labels, overlay_cfg)
        n_detected += int(detected_count > 0)
        n_accepted += int(accepted_count > 0)
    detection_rate = float(n_detected) / float(n_windows) if n_windows else float("nan")
    acceptance_rate = float(n_accepted) / float(n_windows) if n_windows else float("nan")
    return {
        "n_windows": n_windows,
        "n_detected": n_detected,
        "n_accepted": n_accepted,
        "detection_rate": detection_rate,
        "acceptance_rate": acceptance_rate,
    }


def _build_curve_dataframe(rows: Sequence[dict[str, float]]) -> pd.DataFrame:
    required = [
        "mu",
        "detection_rate",
        "acceptance_rate",
        "n_windows",
        "n_detected",
        "n_accepted",
    ]
    df = pd.DataFrame(rows)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"curve rows missing required columns: {missing}")
    return df.loc[:, required]


def _make_overlay_config(config: eval_run.EvalConfig) -> OverlayConfig:
    return OverlayConfig(
        shrinker=config.shrinker,
        q_max=int(config.q_max) if config.q_max is not None else None,
        max_detections=int(config.q_max) if config.q_max is not None else None,
        edge_mode=str(config.edge_mode),
        seed=config.overlay_seed if config.overlay_seed is not None else config.seed,
        a_grid=int(config.overlay_a_grid),
        delta_frac=getattr(config, "overlay_delta_frac", None),
        require_isolated=bool(config.require_isolated),
        cs_drop_top_frac=config.cs_drop_top_frac,
        ewma_halflife=float(config.ewma_halflife),
        gate_mode=str(config.gate_mode) if config.gate_mode else "strict",
        gate_soft_max=config.gate_soft_max,
        gate_delta_calibration=str(config.gate_delta_calibration)
        if config.gate_delta_calibration
        else None,
        gate_delta_frac_min=config.gate_delta_frac_min,
        gate_delta_frac_max=config.gate_delta_frac_max,
        gate_stability_min=config.gate_stability_min,
        gate_alignment_min=config.gate_alignment_min,
        gate_accept_nonisolated=bool(config.gate_accept_nonisolated),
        coarse_candidate=bool(getattr(config, "coarse_candidate", False)),
    )


def _select_window_indices(
    total: int,
    max_windows: int | None,
    mode: str,
    seed: int,
) -> list[int]:
    if total <= 0:
        return []
    if max_windows is None or max_windows >= total:
        return list(range(total))
    if max_windows <= 0:
        raise ValueError("max_windows must be positive when provided.")
    mode_norm = (mode or "first").strip().lower()
    if mode_norm == "first":
        return list(range(int(max_windows)))
    if mode_norm == "random":
        rng = np.random.default_rng(int(seed))
        return sorted(rng.choice(total, size=int(max_windows), replace=False).tolist())
    raise ValueError(f"Unsupported window sampling mode: {mode}.")


def _extract_window_diagnostics(stats: dict[str, Any]) -> dict[str, Any]:
    pre_gate = stats.get("pre_gate", {}) if stats else {}
    gating = stats.get("gating", {}) if stats else {}
    diagnostics = stats.get("diagnostics", {}) if stats else {}
    row: dict[str, Any] = {
        "pre_gate_raw_outliers_found": pre_gate.get("raw_outliers_found"),
        "pre_gate_mp_edge_margin": pre_gate.get("mp_edge_margin"),
        "pre_gate_leakage_offcomp": pre_gate.get("leakage_offcomp"),
        "pre_gate_stability_eta_pass": pre_gate.get("stability_eta_pass"),
        "pre_gate_bracket_status": pre_gate.get("bracket_status"),
        "pre_gate_coarse_candidates": pre_gate.get("coarse_candidates"),
        "gating_mode": gating.get("mode"),
        "gating_rejected": gating.get("rejected"),
        "gating_soft_cap": gating.get("soft_cap"),
        "gating_delta_frac_used": gating.get("delta_frac_used"),
        "gating_capped": gating.get("capped"),
    }

    reasons = gating.get("reasons", {}) if isinstance(gating, dict) else {}
    for key in GATE_REASON_KEYS:
        value = reasons.get(key, 0) if isinstance(reasons, dict) else 0
        try:
            row[f"gate_reason_{key}"] = int(value)
        except (TypeError, ValueError):
            row[f"gate_reason_{key}"] = 0

    unknown_count = 0
    for key in GUARD_KEYS:
        value = diagnostics.get(key, 0) if isinstance(diagnostics, dict) else 0
        try:
            row[f"guard_{key}"] = int(value)
        except (TypeError, ValueError):
            row[f"guard_{key}"] = 0
    if isinstance(diagnostics, dict):
        for key, value in diagnostics.items():
            if key in GUARD_KEYS:
                continue
            try:
                unknown_count += int(value)
            except (TypeError, ValueError):
                continue
    row["guard_unknown"] = int(unknown_count)
    return row


def _build_windows_detail_dataframe(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in WINDOW_DETAIL_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in PRE_GATE_COLUMNS + GATING_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    guard_cols = [f"guard_{key}" for key in GUARD_KEYS] + ["guard_unknown"]
    gate_reason_cols = [f"gate_reason_{key}" for key in GATE_REASON_KEYS]
    for col in guard_cols + gate_reason_cols:
        if col not in df.columns:
            df[col] = 0
    ordered = WINDOW_DETAIL_REQUIRED_COLUMNS + PRE_GATE_COLUMNS + GATING_COLUMNS + guard_cols + gate_reason_cols
    extras = [col for col in df.columns if col not in ordered]
    return df.loc[:, ordered + extras]


def _build_gating_reasons_dataframe(detail_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["stage", "reason", "count", "injected_mu"]
    if detail_df.empty:
        return pd.DataFrame(columns=columns)

    df = detail_df.copy()
    if "injected_mu" not in df.columns:
        df["injected_mu"] = np.nan
    df["_mu"] = pd.to_numeric(df["injected_mu"], errors="coerce").fillna(0.0)

    for key in GUARD_KEYS:
        col = f"guard_{key}"
        if col not in df.columns:
            df[col] = 0
    if "guard_unknown" not in df.columns:
        df["guard_unknown"] = 0
    for key in GATE_REASON_KEYS:
        col = f"gate_reason_{key}"
        if col not in df.columns:
            df[col] = 0

    rows: list[dict[str, Any]] = []
    for mu_val, group in df.groupby("_mu"):
        mu_float = float(mu_val)
        for key in GUARD_KEYS:
            col = f"guard_{key}"
            count = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
            rows.append({"stage": "pre_gate", "reason": key, "count": count, "injected_mu": mu_float})
        unknown_count = int(pd.to_numeric(group["guard_unknown"], errors="coerce").fillna(0).sum())
        rows.append({"stage": "pre_gate", "reason": "unknown", "count": unknown_count, "injected_mu": mu_float})
        for key in GATE_REASON_KEYS:
            col = f"gate_reason_{key}"
            count = int(pd.to_numeric(group[col], errors="coerce").fillna(0).sum())
            rows.append({"stage": "post_gate", "reason": key, "count": count, "injected_mu": mu_float})
    return pd.DataFrame(rows, columns=columns)


def _thread_env_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def _write_profile_summary(profile: cProfile.Profile, out_path: Path) -> None:
    stream = StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.sort_stats("cumulative")
    stream.write("==== Top cumulative (30) ===\n")
    stats.print_stats(30)
    stream.write("\n==== fjs.dealias / fjs.mp ===\n")
    stats.print_stats("fjs.dealias")
    stats.print_stats("fjs.mp")
    out_path.write_text(stream.getvalue(), encoding="utf-8")


def _dominant_tvec_reasons(gating_df: pd.DataFrame) -> bool:
    if gating_df.empty:
        return False
    subset = gating_df[(gating_df["stage"] == "pre_gate") & (gating_df["injected_mu"] == 0.0)]
    if subset.empty:
        return False
    total = int(subset["count"].sum())
    if total <= 0:
        return False
    tvec_reasons = {
        "tvec_compute_error",
        "tvec_no_real_root",
        "tvec_no_admissible_root",
        "tvec_singularity",
        "tvec_off_component",
    }
    tvec_total = int(subset[subset["reason"].isin(tvec_reasons)]["count"].sum())
    return (tvec_total / float(total)) >= 0.5


def _write_debug_window(
    sample: WindowSample,
    stats: dict[str, Any],
    overlay_cfg: OverlayConfig,
    thresholds_path: Path | None,
    out_path: Path,
) -> None:
    overlay_payload = {field.name: getattr(overlay_cfg, field.name) for field in fields(OverlayConfig)}
    thresholds_payload = None
    if thresholds_path is not None and thresholds_path.exists():
        try:
            thresholds_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            thresholds_payload = None
    payload = {
        "overlay_config": overlay_payload,
        "thresholds_path": str(thresholds_path) if thresholds_path else None,
        "thresholds": thresholds_payload,
        "stats": stats,
        "fit_start": str(sample.fit_start),
        "fit_end": str(sample.fit_end),
        "horizon_start": str(sample.hold_start),
        "horizon_end": str(sample.hold_end),
    }
    np.savez_compressed(
        out_path,
        matrix=sample.matrix,
        group_labels=sample.group_labels,
        metadata=np.array(json.dumps(payload, default=str)),
    )


def _resolve_eval_config_or_fail(args: argparse.Namespace) -> ResolveResult:
    config_args = {
        "returns_csv": args.returns_csv,
        "factors_csv": args.factors_csv,
        "window": args.window,
        "horizon": args.horizon,
        "start": args.start,
        "end": args.end,
        "assets_top": args.assets_top,
        "config": args.config,
        "thresholds": args.thresholds,
        "group_design": args.group_design,
        "use_factor_prewhiten": args.use_factor_prewhiten,
        "coarse_candidate": args.coarse_candidate,
    }
    resolved = resolve_eval_config(config_args)
    config = resolved.config
    if config.config_path is None or not Path(config.config_path).exists():
        raise FileNotFoundError("Resolved config file is missing; provide --config.")
    return resolved


def _collect_windows(
    config: eval_run.EvalConfig,
    raw_returns: pd.DataFrame,
    residuals: pd.DataFrame,
    vol_proxy_full: pd.Series,
    *,
    factor_tracking_required: bool,
    residual_index_set: set[pd.Timestamp],
    skip_counts: Counter[str] | None = None,
) -> list[WindowSample]:
    windows: list[WindowSample] = []
    vol_proxy_past = vol_proxy_full.shift(1)
    total_days = raw_returns.shape[0]
    start_indices = range(0, total_days - config.window - config.horizon + 1)

    def _skip(reason: str) -> None:
        if skip_counts is not None:
            skip_counts[reason] += 1

    for start in start_indices:
        fit_end = start + config.window
        hold_end = fit_end + config.horizon
        fit_labels = list(raw_returns.index[start:fit_end])
        hold_labels = list(raw_returns.index[fit_end:hold_end])
        if len(fit_labels) < config.window or len(hold_labels) < config.horizon:
            _skip("insufficient_labels")
            continue
        fit_base = raw_returns.loc[fit_labels]
        overlay_allowed = True
        if factor_tracking_required:
            needed = fit_labels + hold_labels
            overlay_allowed = all(label in residual_index_set for label in needed)
        fit = residuals.loc[fit_labels] if overlay_allowed else fit_base
        if not overlay_allowed:
            _skip("factor_tracking_unavailable")
            continue

        train_end = pd.to_datetime(fit_labels[-1])
        hold_start = pd.to_datetime(hold_labels[0])
        calm_cut, crisis_cut = eval_run._vol_thresholds(vol_proxy_past, train_end, config)
        try:
            fit_grouped, group_labels = eval_run._build_grouped_window(
                fit,
                config=config,
                calm_threshold=calm_cut,
                crisis_threshold=crisis_cut,
                vol_proxy=vol_proxy_past,
            )
        except GroupingError:
            _skip("grouping_error")
            continue

        fit_grouped = fit_grouped.replace([np.inf, -np.inf], np.nan)
        nan_result = apply_nan_policy(
            fit_grouped,
            group_labels,
            max_missing_asset=float(config.max_missing_asset),
            max_missing_group_row=float(config.max_missing_group_row),
        )
        fit_clean = nan_result.frame.replace([np.inf, -np.inf], np.nan)
        if fit_clean.shape[0] == 0 or fit_clean.shape[1] == 0:
            _skip("nan_policy_empty")
            continue
        balance_result = build_balanced_window(
            fit_clean,
            nan_result.labels,
            min_replicates=eval_run._required_replicates(config.group_design, config),
        )
        if balance_result.reason in {"empty_after_balance", "insufficient_reps"}:
            _skip(f"balance_{balance_result.reason}")
            continue
        fit_balanced = balance_result.frame.replace([np.inf, -np.inf], np.nan)
        if fit_balanced.shape[0] == 0 or fit_balanced.shape[1] == 0:
            _skip("empty_matrix")
            continue
        if fit_balanced.isna().any().any():
            valid_mask = fit_balanced.notna().all(axis=1)
            if not bool(valid_mask.all()):
                keep_positions = np.where(valid_mask.to_numpy(dtype=bool))[0]
                fit_balanced = fit_balanced.iloc[keep_positions]
                balance_labels = balance_result.labels[keep_positions]
            else:
                balance_labels = balance_result.labels
        else:
            balance_labels = balance_result.labels
        if fit_balanced.shape[0] == 0 or fit_balanced.shape[1] == 0:
            _skip("empty_matrix")
            continue
        matrix = fit_balanced.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(matrix).all():
            _skip("nonfinite_matrix")
            continue
        windows.append(
            WindowSample(
                window_id=start,
                matrix=matrix,
                group_labels=np.asarray(balance_labels, dtype=np.intp),
                fit_start=pd.to_datetime(fit_labels[0]),
                fit_end=pd.to_datetime(fit_labels[-1]),
                hold_start=pd.to_datetime(hold_labels[0]),
                hold_end=pd.to_datetime(hold_labels[-1]),
            )
        )
    return windows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weak-spike injection evaluation on real residuals")
    parser.add_argument("--returns-csv", type=Path, required=True)
    parser.add_argument("--factors-csv", type=Path, default=None)
    parser.add_argument("--window", type=int, default=126)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--assets-top", type=int, default=150)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--group-design", type=str, default="week")
    parser.add_argument("--use-factor-prewhiten", type=int, choices=[0, 1], default=1)
    parser.add_argument("--coarse-candidate", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mu-grid", type=str, default="3,4,5")
    parser.add_argument("--inject-frac-min", type=float, default=0.05)
    parser.add_argument("--inject-frac-max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--window-sampling", type=str, choices=["first", "random"], default="first")
    parser.add_argument("--window-sampling-seed", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="write cProfile summary to output dir")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/inject_spike"))
    args = parser.parse_args(argv)
    args.mu_values = _parse_float_list(args.mu_grid, "mu")
    if any(mu < 0 for mu in args.mu_values):
        raise ValueError("mu values must be non-negative.")
    if args.inject_frac_min <= 0 or args.inject_frac_max <= 0:
        raise ValueError("Injection fractions must be positive.")
    if args.inject_frac_max < args.inject_frac_min:
        raise ValueError("inject-frac-max must be >= inject-frac-min")
    if args.inject_frac_min > 1 or args.inject_frac_max > 1:
        raise ValueError("Injection fractions must be <= 1.")
    if args.window_sampling_seed is None:
        args.window_sampling_seed = args.seed
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resolved = _resolve_eval_config_or_fail(args)
    config = resolved.config
    if plt is None:
        raise RuntimeError("matplotlib is required for injection plots")

    start_utc = datetime.now(tz=timezone.utc)
    start_perf = time.perf_counter()
    thread_env = _thread_env_snapshot()
    thresholds_path = Path(args.thresholds) if args.thresholds else DEFAULT_THRESHOLDS_PATH
    if thresholds_path is not None and not thresholds_path.exists():
        thresholds_path = None

    profiler: cProfile.Profile | None = cProfile.Profile() if args.profile else None

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = args.out.resolve()
    run_dir = out_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_payload = dict(resolved.resolved)
    resolved_payload["inject_spike"] = {
        "mu_grid": list(args.mu_values),
        "inject_frac_min": float(args.inject_frac_min),
        "inject_frac_max": float(args.inject_frac_max),
        "seed": int(args.seed),
        "max_windows": args.max_windows,
        "window_sampling": str(args.window_sampling),
        "window_sampling_seed": int(args.window_sampling_seed),
        "run_id": run_id,
    }
    resolved_config_path = run_dir / "resolved_config.json"
    resolved_config_path.write_text(
        json.dumps(resolved_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    panel, raw_returns, whitening, telemetry, factor_entry = eval_run._prepare_returns(config)
    residuals = whitening.residuals.sort_index()
    raw_returns = raw_returns.sort_index()
    residual_index_set = set(residuals.index)
    factor_tracking_required = bool(config.use_factor_prewhiten and telemetry.mode_effective != "off")
    vol_proxy_full = eval_run._compute_vol_proxy(residuals, span=config.vol_ewma_span)

    skip_counts: Counter[str] = Counter()
    samples = _collect_windows(
        config,
        raw_returns,
        residuals,
        vol_proxy_full,
        factor_tracking_required=factor_tracking_required,
        residual_index_set=residual_index_set,
        skip_counts=skip_counts,
    )
    if not samples:
        raise RuntimeError("No valid windows available for injection analysis.")

    windows_available = len(samples)
    sample_indices = _select_window_indices(
        windows_available,
        args.max_windows,
        args.window_sampling,
        int(args.window_sampling_seed),
    )
    samples = [samples[idx] for idx in sample_indices]
    if not samples:
        raise RuntimeError("No windows selected after sampling.")

    overlay_cfg = _make_overlay_config(config)
    rng = np.random.default_rng(args.seed)
    bases = [_make_injection_basis(sample.matrix, rng) for sample in samples]

    inject_frac = float(rng.uniform(args.inject_frac_min, args.inject_frac_max))
    n_injected = max(1, int(round(inject_frac * len(samples))))
    inject_indices = sorted(rng.choice(len(samples), size=n_injected, replace=False).tolist())
    inject_index_set = set(inject_indices)

    mu_values = sorted({float(mu) for mu in args.mu_values})
    curve_rows: list[dict[str, float]] = []
    window_rows: list[dict[str, Any]] = []
    debug_candidate: tuple[WindowSample, dict[str, Any]] | None = None
    debug_window_path: Path | None = None

    if profiler is not None:
        profiler.enable()

    baseline_stats = {
        "n_windows": len(samples),
        "n_detected": 0,
        "n_accepted": 0,
    }
    for sample in samples:
        detected_count, accepted_count, stats = _window_detection_stats(
            sample.matrix,
            sample.group_labels,
            overlay_cfg,
        )
        if debug_candidate is None:
            diag = stats.get("diagnostics", {}) if isinstance(stats, dict) else {}
            if diag.get("tvec_compute_error", 0) or diag.get("tvec_off_component", 0):
                debug_candidate = (sample, stats)
        baseline_stats["n_detected"] += int(detected_count > 0)
        baseline_stats["n_accepted"] += int(accepted_count > 0)
        row = {
            "window_idx": sample.window_id,
            "fit_start": sample.fit_start,
            "fit_end": sample.fit_end,
            "horizon_start": sample.hold_start,
            "horizon_end": sample.hold_end,
            "n_obs": sample.matrix.shape[0],
            "n_assets": sample.matrix.shape[1],
            "injected": 0,
            "injected_mu": None,
            "detected_initial": int(detected_count),
            "accepted": int(accepted_count),
        }
        row.update(_extract_window_diagnostics(stats))
        window_rows.append(row)

    baseline_stats["detection_rate"] = (
        float(baseline_stats["n_detected"]) / float(baseline_stats["n_windows"])
        if baseline_stats["n_windows"]
        else float("nan")
    )
    baseline_stats["acceptance_rate"] = (
        float(baseline_stats["n_accepted"]) / float(baseline_stats["n_windows"])
        if baseline_stats["n_windows"]
        else float("nan")
    )
    curve_rows.append({"mu": 0.0, **baseline_stats})

    for mu in mu_values:
        if mu <= 0.0:
            continue
        mu_stats = {
            "n_windows": len(inject_indices),
            "n_detected": 0,
            "n_accepted": 0,
        }
        for idx in inject_indices:
            sample = samples[int(idx)]
            matrix = _apply_injection(sample.matrix, bases[int(idx)], mu)
            detected_count, accepted_count, stats = _window_detection_stats(
                matrix,
                sample.group_labels,
                overlay_cfg,
            )
            if debug_candidate is None:
                diag = stats.get("diagnostics", {}) if isinstance(stats, dict) else {}
                if diag.get("tvec_compute_error", 0) or diag.get("tvec_off_component", 0):
                    debug_candidate = (sample, stats)
            mu_stats["n_detected"] += int(detected_count > 0)
            mu_stats["n_accepted"] += int(accepted_count > 0)
            row = {
                "window_idx": sample.window_id,
                "fit_start": sample.fit_start,
                "fit_end": sample.fit_end,
                "horizon_start": sample.hold_start,
                "horizon_end": sample.hold_end,
                "n_obs": sample.matrix.shape[0],
                "n_assets": sample.matrix.shape[1],
                "injected": 1,
                "injected_mu": float(mu),
                "detected_initial": int(detected_count),
                "accepted": int(accepted_count),
            }
            row.update(_extract_window_diagnostics(stats))
            window_rows.append(row)
        mu_stats["detection_rate"] = (
            float(mu_stats["n_detected"]) / float(mu_stats["n_windows"])
            if mu_stats["n_windows"]
            else float("nan")
        )
        mu_stats["acceptance_rate"] = (
            float(mu_stats["n_accepted"]) / float(mu_stats["n_windows"])
            if mu_stats["n_windows"]
            else float("nan")
        )
        mu_stats["mu"] = float(mu)
        curve_rows.append(mu_stats)

    if profiler is not None:
        profiler.disable()

    curve_df = _build_curve_dataframe(curve_rows).sort_values("mu")
    curve_csv = run_dir / "curve.csv"
    curve_df.to_csv(curve_csv, index=False)

    windows_detail_df = _build_windows_detail_dataframe(window_rows)
    windows_detail_csv = run_dir / "windows_detail.csv"
    windows_detail_df.to_csv(windows_detail_csv, index=False)

    gating_reasons_df = _build_gating_reasons_dataframe(windows_detail_df)
    gating_reasons_csv = run_dir / "gating_reasons.csv"
    gating_reasons_df.to_csv(gating_reasons_csv, index=False)

    max_detected = int(curve_df["n_detected"].max()) if not curve_df.empty else 0
    max_accepted = int(curve_df["n_accepted"].max()) if not curve_df.empty else 0
    flat_zero = bool(max_detected == 0 and max_accepted == 0)
    dominant_tvec = _dominant_tvec_reasons(gating_reasons_df)
    if (flat_zero or dominant_tvec) and debug_candidate is not None:
        debug_window_path = run_dir / "debug_window.npz"
        sample, stats = debug_candidate
        _write_debug_window(sample, stats, overlay_cfg, thresholds_path, debug_window_path)

    profile_txt = None
    if profiler is not None:
        profile_txt = run_dir / "profile.txt"
        _write_profile_summary(profiler, profile_txt)

    curve_plot = run_dir / "curve.png"
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(
        curve_df["mu"],
        curve_df["detection_rate"],
        marker="o",
        label="Detection rate (pre-gate)",
    )
    ax.plot(
        curve_df["mu"],
        curve_df["acceptance_rate"],
        marker="s",
        label="Acceptance rate (post-gate)",
    )
    ax.set_xlabel("Injected spike μ")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(curve_plot, dpi=200)
    plt.close(fig)

    selected_windows = pd.DataFrame(
        [
            {
                "window_id": sample.window_id,
                "fit_start": sample.fit_start,
                "fit_end": sample.fit_end,
                "hold_start": sample.hold_start,
                "hold_end": sample.hold_end,
                "n_obs": sample.matrix.shape[0],
                "n_assets": sample.matrix.shape[1],
                "injected": int(idx in inject_index_set),
            }
            for idx, sample in enumerate(samples)
        ]
    )
    selected_windows_csv = run_dir / "selected_windows.csv"
    selected_windows.to_csv(selected_windows_csv, index=False)

    returns_hash = None
    factors_hash = None
    try:
        returns_hash = eval_run._sha256_path(config.returns_csv)
    except Exception:
        returns_hash = None
    if config.factors_csv:
        try:
            factors_hash = eval_run._sha256_path(config.factors_csv)
        except Exception:
            factors_hash = None

    window_date_range = {
        "fit_start_min": min(sample.fit_start for sample in samples),
        "fit_end_max": max(sample.fit_end for sample in samples),
        "hold_start_min": min(sample.hold_start for sample in samples),
        "hold_end_max": max(sample.hold_end for sample in samples),
    }
    asset_counts = [sample.matrix.shape[1] for sample in samples]
    obs_counts = [sample.matrix.shape[0] for sample in samples]
    window_candidates = max(raw_returns.shape[0] - config.window - config.horizon + 1, 0)

    end_utc = datetime.now(tz=timezone.utc)
    runtime_sec = float(time.perf_counter() - start_perf)

    run_metadata = {
        "run_id": run_id,
        "timestamp_utc": end_utc.isoformat(),
        "git_sha": eval_run._current_git_sha(),
        "git_dirty": eval_run._git_dirty(),
        "out_dir": str(run_dir),
        "config": eval_run._serialise_config(config),
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_hash": eval_run._sha256_path(resolved_config_path),
        "data": {
            "returns_csv": str(config.returns_csv),
            "returns_sha256": returns_hash,
            "factors_csv": str(config.factors_csv) if config.factors_csv else None,
            "factors_sha256": factors_hash,
            "panel": getattr(panel, "meta", None),
        },
        "design": {
            "group_design": config.group_design,
            "edge_mode": config.edge_mode,
            "prewhiten": config.prewhiten,
            "prewhiten_mode_effective": telemetry.mode_effective,
            "use_factor_prewhiten": bool(config.use_factor_prewhiten),
            "coarse_candidate": bool(config.coarse_candidate),
        },
        "windows": {
            "windows_candidate": window_candidates,
            "windows_available": windows_available,
            "windows_used": len(samples),
            "window_date_range": window_date_range,
            "assets_top": config.assets_top,
            "assets_min": min(asset_counts),
            "assets_max": max(asset_counts),
            "obs_min": min(obs_counts),
            "obs_max": max(obs_counts),
            "skip_reasons": dict(skip_counts),
            "n_changed": int(baseline_stats["n_accepted"]),
            "baseline_windows": len(samples),
            "injected_windows": n_injected,
            "non_injected_windows": len(samples) - n_injected,
            "sampling": {
                "max_windows": args.max_windows,
                "window_sampling": str(args.window_sampling),
                "window_sampling_seed": int(args.window_sampling_seed),
            },
        },
        "injection": {
            "seed": int(args.seed),
            "mu_grid": mu_values,
            "inject_frac_min": float(args.inject_frac_min),
            "inject_frac_max": float(args.inject_frac_max),
            "inject_frac": inject_frac,
            "n_injected": n_injected,
            "basis_fixed_per_window": True,
            "mu_definition": "rank-1 covariance spike eigenvalue (variance) applied as sqrt(mu) * z_t v^T",
            "signal_definition": "z_t standardised to mean 0, std 1; v unit-norm; same (z_t, v) reused for each mu",
        },
        "metrics": {
            "baseline_detection_rate": baseline_stats["detection_rate"],
            "baseline_acceptance_rate": baseline_stats["acceptance_rate"],
            "baseline_n_windows": baseline_stats["n_windows"],
            "baseline_n_detected": baseline_stats["n_detected"],
            "baseline_n_accepted": baseline_stats["n_accepted"],
            "detection_rate_definition": "fraction of windows with >=1 candidate before gating",
            "acceptance_rate_definition": "fraction of windows where overlay would be applied after gating",
        },
        "outputs": {
            "curve_csv": str(curve_csv),
            "curve_plot": str(curve_plot),
            "selected_windows_csv": str(selected_windows_csv),
            "windows_detail_csv": str(windows_detail_csv),
            "gating_reasons_csv": str(gating_reasons_csv),
            "debug_window_npz": str(debug_window_path) if debug_window_path else None,
            "profile_txt": str(profile_txt) if profile_txt else None,
            "resolved_config_json": str(resolved_config_path),
        },
        "runtime": {
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "wall_sec": runtime_sec,
            "python_version": sys.version,
            "platform": platform.platform(),
            "exec_mode": thread_env.get("EXEC_MODE"),
            "workers": 1,
            "thread_env": thread_env,
        },
        "gating_summary": gating_reasons_df.to_dict(orient="records"),
        "factors": (
            {
                "key": factor_entry.key,
                "path": str(factor_entry.path),
                "sha256": factor_entry.sha256,
                "start_date": factor_entry.start_date,
                "end_date": factor_entry.end_date,
                "source": factor_entry.source,
                "note": factor_entry.note,
            }
            if factor_entry is not None
            else None
        ),
        "cap_active": False,
    }
    run_json_path = run_dir / "run.json"
    run_json_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    print(
        f"[inject] windows={len(samples)} "
        f"baseline_detect={baseline_stats['detection_rate']:.3f} "
        f"baseline_accept={baseline_stats['acceptance_rate']:.3f} "
        + ", ".join(
            f"μ={row['mu']}: det={row['detection_rate']:.2f} acc={row['acceptance_rate']:.2f}"
            for row in curve_rows
            if row["mu"] > 0
        )
    )


if __name__ == "__main__":
    main()
