from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.equity_panel.run import _infer_skip_reason  # type: ignore
from fjs.balanced_nested import mean_squares_nested
from fjs.dealias import dealias_search
from fjs.gating import count_isolated_outliers, lookup_calibrated_delta, select_top_k
from fjs.robust import edge_from_scatter, huber_scatter, tyler_scatter


DEFAULT_CONFIG = {
    "n_assets": 200,
    "n_assets_options": None,
    "years": 2,
    "weeks_options": [6, 7, 8],
    "replicates": 5,
    "trials_per_scenario": {"null": 220, "moderate": 60, "strong": 40},
    "spikes": {"null": 0.0, "moderate": 3.0, "strong": 6.0},
    "noise_variance": 1.0,
    "signal_to_noise": 0.35,
    "edge_modes": ["tyler"],
    "edge_huber_c": 1.5,
    "delta": 0.35,
    "delta_frac_min": 0.05,
    "eps": 0.008,
    "stability_eta_deg": 0.2,
    "a_grid": 96,
    "cs_drop_top_frac": 0.1,
    "cs_sensitivity_frac": 0.0,
    "off_component_leak_cap": 25.0,
    "energy_min_abs": 2e-7,
    "allow_nonisolated": False,
    "nonisolated_stability_min": 0.015,
    "nonisolated_edge_min": 0.015,
    "nonisolated_q_max": 2,
    "require_isolated": True,
    "q_max": 2,
    "calibration_path": "calibration/nested_edge_delta_thresholds.json",
    "calibration_design": "nested",
    "target_fpr": 0.02,
    "delta_frac_grid": None,
    "calibration_out": None,
    "run_name": None,
    "seed": 0,
    "out_dir": "reports/synthetic_nested_killtest",
}


@dataclass(frozen=True)
class TrialResult:
    scenario: str
    trial: int
    weeks_common: int
    n_obs: int
    p_assets: int
    detected: bool
    skip_reason: str
    isolated_spikes: int
    edge_mode: str
    delta_frac_used: float
    calibration_missing: bool


def _wilson_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""

    if trials <= 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = successes / trials
    denom = 1.0 + (z * z) / trials
    centre = phat + (z * z) / (2.0 * trials)
    adj = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * trials)) / trials)
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return (max(0.0, low), min(1.0, high))


def _current_git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, timeout=5
            )
            .strip()
        )
    except Exception:
        return None


def _nan_safe(val: float, default: float) -> float:
    return float(val) if (val is not None and np.isfinite(val)) else default


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG.copy()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Config must be a mapping.")
    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in data.items() if v is not None})
    return merged


def _resolve_n_assets(config: Mapping[str, Any]) -> list[int]:
    raw = config.get("n_assets_options")
    if raw is None:
        raw = config.get("n_assets")
    if isinstance(raw, (list, tuple)):
        values = [int(val) for val in raw]
    else:
        values = [int(raw)]
    values = sorted({val for val in values if val > 0})
    if not values:
        raise ValueError("n_assets must contain at least one positive value.")
    return values


def simulate_nested_panel(
    rng: np.random.Generator,
    *,
    n_assets: int,
    years: int,
    weeks: int,
    replicates: int,
    spike_strength: float,
    signal_to_noise: float,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (observations, year_labels, week_labels)."""

    total_obs = years * weeks * replicates
    observations = np.zeros((total_obs, n_assets), dtype=np.float64)
    year_labels = np.zeros(total_obs, dtype=np.intp)
    week_labels = np.zeros(total_obs, dtype=np.intp)

    spike_dir = rng.normal(size=n_assets)
    spike_dir /= max(float(np.linalg.norm(spike_dir)), 1e-8)
    aux_dir = rng.normal(size=n_assets)
    aux_dir /= max(float(np.linalg.norm(aux_dir)), 1e-8)
    noise_scale = np.sqrt(noise_variance)

    idx = 0
    for year in range(years):
        year_effect = rng.normal() if spike_strength > 0.0 else 0.0
        for week in range(weeks):
            week_effect = rng.normal() if spike_strength > 0.0 else 0.0
            for _ in range(replicates):
                obs = noise_scale * rng.normal(size=n_assets)
                if year_effect:
                    obs += signal_to_noise * year_effect * aux_dir
                if week_effect:
                    obs += 0.25 * signal_to_noise * week_effect * aux_dir
                if spike_strength > 0.0:
                    direction = 1.0 if year == 0 else -1.0
                    obs += float(spike_strength) * direction * spike_dir
                observations[idx] = obs
                year_labels[idx] = year
                week_labels[idx] = week
                idx += 1
    return observations, year_labels, week_labels


def _edge_scale(observations: np.ndarray, edge_mode: str, edge_huber_c: float) -> tuple[float, float, float]:
    p_dim = observations.shape[1]
    n_obs = observations.shape[0]
    edge_scm = float("nan")
    edge_selected = float("nan")
    edge_tyler = float("nan")
    try:
        scatter_scm = np.cov(observations, rowvar=False, ddof=1)
        scatter_scm = 0.5 * (scatter_scm + scatter_scm.T)
        edge_scm = edge_from_scatter(scatter_scm, p_dim, n_obs)
    except Exception:
        pass
    if edge_mode == "tyler":
        try:
            scatter_tyler = tyler_scatter(observations)
            scatter_tyler = 0.5 * (scatter_tyler + scatter_tyler.T)
            edge_tyler = edge_from_scatter(scatter_tyler, p_dim, n_obs)
            edge_selected = edge_tyler
        except Exception:
            pass
    elif edge_mode == "huber":
        try:
            scatter_huber = huber_scatter(observations, edge_huber_c)
            scatter_huber = 0.5 * (scatter_huber + scatter_huber.T)
            edge_selected = edge_from_scatter(scatter_huber, p_dim, n_obs)
        except Exception:
            pass
    else:
        edge_selected = edge_scm

    if np.isfinite(edge_scm) and edge_scm > 0 and np.isfinite(edge_selected) and edge_selected > 0:
        scale = float(edge_selected / edge_scm)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
    else:
        scale = 1.0
    return scale, edge_scm, edge_selected


def _gate_nested_detections(
    detections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    delta_frac_used: float,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Apply overlay-like gating to nested detections."""

    min_edge = float(config.get("min_edge_margin", 0.0) or 0.0)
    stability_min = float(config.get("stability_eta_deg", 0.0) or 0.0)
    delta_frac_min = config.get("delta_frac_min")
    delta_frac_max = config.get("delta_frac_max")
    allow_nonisolated = bool(config.get("allow_nonisolated", False))
    q_max = int(config.get("q_max", 0) or 0)
    reasons: Counter[str] = Counter()
    accepted: list[Mapping[str, Any]] = []

    for det in detections:
        if not bool(det.get("admissible_root", True)):
            reasons["inadmissible_root"] += 1
            continue
        edge_margin = det.get("edge_margin")
        try:
            edge_val = float(edge_margin)
        except (TypeError, ValueError):
            edge_val = float("-inf")
        if edge_val < min_edge:
            reasons["edge_margin"] += 1
            continue
        pre_count = det.get("pre_outlier_count")
        if not allow_nonisolated and (pre_count is None or int(pre_count) != 1):
            reasons["nonisolated"] += 1
            continue
        try:
            stability = float(det.get("stability_margin", 0.0))
        except (TypeError, ValueError):
            stability = float("-inf")
        if stability < stability_min:
            reasons["stability"] += 1
            continue
        delta_used = det.get("delta_frac", delta_frac_used)
        try:
            delta_val = float(delta_used)
        except (TypeError, ValueError):
            delta_val = float("nan")
        if delta_frac_min is not None and np.isfinite(delta_val) and delta_val < float(delta_frac_min):
            reasons["delta_frac_min"] += 1
            continue
        if delta_frac_max is not None and np.isfinite(delta_val) and delta_val > float(delta_frac_max):
            reasons["delta_frac_max"] += 1
            continue
        accepted.append(det)

    if q_max > 0 and len(accepted) > q_max:
        selected, discarded = select_top_k(accepted, q_max)
        reasons["q_max_trim"] += len(discarded)
        accepted = selected

    return accepted, dict(reasons)


def run_trials(config: Mapping[str, Any]) -> tuple[list[TrialResult], dict[str, Any]]:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    records: list[TrialResult] = []
    diag_summary: dict[str, Any] = {}

    spikes_cfg = config.get("spikes", {})
    if isinstance(spikes_cfg, Mapping):
        scenarios = list(spikes_cfg.items())
    elif isinstance(spikes_cfg, Sequence):
        scenarios = [(str(val), float(val)) for val in spikes_cfg]
    else:
        raise ValueError("spikes must be mapping or sequence.")

    years = int(config["years"])
    weeks_options = list(config["weeks_options"])
    replicates = int(config["replicates"])
    n_assets_values = _resolve_n_assets(config)
    trials_cfg = config.get("trials_per_scenario", 0)
    if isinstance(trials_cfg, Mapping):
        trials_default = int(trials_cfg.get("default", 0))
    else:
        trials_default = int(trials_cfg)
    edge_modes = [str(mode).lower() for mode in config.get("edge_modes", ["tyler"])]

    for edge_mode in edge_modes:
        skip_reasons_accum: Counter[str] = Counter()
        gate_rejections_accum: Counter[str] = Counter()
        for scenario_label, spike_strength in scenarios:
            if isinstance(trials_cfg, Mapping):
                trials_per = int(trials_cfg.get(scenario_label, trials_default))
            else:
                trials_per = trials_default
            for trial in range(trials_per):
                weeks = int(rng.choice(weeks_options))
                for n_assets in n_assets_values:
                    observations, year_labels, week_labels = simulate_nested_panel(
                        rng,
                        n_assets=n_assets,
                        years=years,
                        weeks=weeks,
                        replicates=replicates,
                        spike_strength=float(spike_strength),
                        signal_to_noise=float(config["signal_to_noise"]),
                        noise_variance=float(config["noise_variance"]),
                    )
                    n_obs = observations.shape[0]
                    try:
                        (ms1, ms2, ms3), meta = mean_squares_nested(
                            observations,
                            year_labels,
                            week_labels,
                            replicates,
                        )
                    except ValueError:
                        records.append(
                            TrialResult(
                                scenario=str(scenario_label),
                                trial=trial,
                                weeks_common=weeks,
                                n_obs=n_obs,
                                p_assets=n_assets,
                                detected=False,
                                skip_reason="prep_failure",
                                isolated_spikes=0,
                                edge_mode=edge_mode,
                                delta_frac_used=float(config["delta_frac_min"]),
                                calibration_missing=True,
                            )
                        )
                        continue

                    sigma1 = (
                        (ms1 - ms2) / float(meta.J * meta.replicates)
                    ).astype(np.float64, copy=False)
                    sigma2 = ((ms2 - ms3) / float(meta.replicates)).astype(
                        np.float64, copy=False
                    )
                    sigma3 = ms3.astype(np.float64, copy=False)

                    design_c = meta.c.astype(np.float64, copy=False)
                    design_d = meta.d.astype(np.float64, copy=False)
                    design_N = float(meta.N)
                    design_order = [[1, 2, 3], [2, 3], [3]]
                    design_override = {
                        "c": design_c,
                        "C": np.ones_like(design_c, dtype=np.float64),
                        "d": design_d,
                        "N": design_N,
                        "order": design_order,
                    }

                    stats_local = {
                        "MS1": ms1.astype(np.float64, copy=False),
                        "MS2": ms2.astype(np.float64, copy=False),
                        "MS3": ms3.astype(np.float64, copy=False),
                        "Sigma1_hat": sigma1,
                        "Sigma2_hat": sigma2,
                        "Sigma3_hat": sigma3,
                        "I": meta.I,
                        "J": meta.J,
                        "n": meta.n,
                        "replicates": meta.replicates,
                    }

                    delta_frac_calib = lookup_calibrated_delta(
                        edge_mode=edge_mode,
                        p=n_assets,
                        t=n_obs,
                        calibration_path=config["calibration_path"],
                        design=config.get("calibration_design"),
                    )
                    calibration_missing = delta_frac_calib is None
                    base_delta_frac = float(config["delta_frac_min"])
                    delta_frac_used = (
                        base_delta_frac
                        if delta_frac_calib is None
                        else max(base_delta_frac, float(delta_frac_calib))
                    )

                    edge_scale, edge_scm_val, edge_sel_val = _edge_scale(
                        observations,
                        edge_mode=edge_mode,
                        edge_huber_c=float(config["edge_huber_c"]),
                    )

                    diag_local: dict[str, int] = {}
                    detections = dealias_search(
                        observations,
                        np.arange(observations.shape[0], dtype=np.intp),
                        target_r=0,
                        delta=float(config["delta"]),
                        delta_frac=float(delta_frac_used),
                        eps=float(config["eps"]),
                        energy_min_abs=float(config["energy_min_abs"]),
                        stability_eta_deg=float(config["stability_eta_deg"]),
                        use_tvector=False,
                        nonnegative_a=False,
                        a_grid=int(config["a_grid"]),
                        cs_drop_top_frac=float(config["cs_drop_top_frac"]),
                        cs_sensitivity_frac=float(config["cs_sensitivity_frac"]),
                        scan_basis="sigma",
                        off_component_leak_cap=float(config["off_component_leak_cap"]),
                        diagnostics=diag_local,
                        stats=stats_local,
                        design=design_override,
                        edge_scale=edge_scale,
                        edge_mode=edge_mode,
                    )
                    detections = list(detections or [])
                    for det in detections:
                        det["delta_frac"] = float(delta_frac_used)
                    gated, gate_reasons = _gate_nested_detections(
                        detections,
                        config,
                        delta_frac_used=delta_frac_used,
                    )
                    gate_rejections_accum.update(gate_reasons)
                    diag_local["gating_rejections"] = gate_reasons
                    isolated_count = count_isolated_outliers(gated, None, None)
                    window_skip_reason: str | None = None
                    candidate_pool = list(gated)

                    if bool(config["require_isolated"]) and isolated_count == 0:
                        window_skip_reason = "no_isolated_spike"
                        candidate_pool = []

                    if not candidate_pool and not window_skip_reason:
                        window_skip_reason = _infer_skip_reason(
                            diag_local,
                            calibration_missing=calibration_missing,
                            isolated_spikes=int(isolated_count),
                        )

                    detected = bool(candidate_pool)
                    skip_reasons_accum[window_skip_reason or ""] += 1
                    records.append(
                        TrialResult(
                            scenario=str(scenario_label),
                            trial=trial,
                            weeks_common=weeks,
                            n_obs=n_obs,
                            p_assets=n_assets,
                            detected=detected,
                            skip_reason=window_skip_reason or "",
                            isolated_spikes=int(isolated_count),
                            edge_mode=edge_mode,
                            delta_frac_used=float(delta_frac_used),
                            calibration_missing=bool(calibration_missing),
                        )
                    )
        diag_summary[edge_mode] = {
            "skip_reasons": dict(skip_reasons_accum),
            "gating_rejections": dict(gate_rejections_accum),
        }

    return records, diag_summary


def summarise_results(results: Sequence[TrialResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    summaries: list[dict[str, Any]] = []
    for (edge_mode, scenario, p_assets), group in df.groupby(
        ["edge_mode", "scenario", "p_assets"]
    ):
        detections = int(group["detected"].sum()) if not group.empty else 0
        ci_low, ci_high = _wilson_interval(detections, int(group.shape[0]), alpha=0.05)
        skip_counts = (
            group["skip_reason"].replace("", np.nan).dropna().value_counts().to_dict()
            if not group.empty
            else {}
        )
        summaries.append(
            {
                "edge_mode": edge_mode,
                "scenario": scenario,
                "p_assets": int(p_assets),
                "trials": int(group.shape[0]),
                "detections": detections,
                "detection_rate": float(group["detected"].mean()) if not group.empty else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "weeks_mean": float(group["weeks_common"].mean()) if not group.empty else float("nan"),
                "n_obs_mean": float(group["n_obs"].mean()) if not group.empty else float("nan"),
                "skip_reason_top": max(skip_counts, key=skip_counts.get) if skip_counts else "",
                "skip_reason_counts": skip_counts,
                "calibration_missing_share": float(
                    np.mean(group["calibration_missing"]) if not group.empty else 0.0
                ),
            }
        )
    return pd.DataFrame(summaries)


def write_summary_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Nested synthetic kill-test", ""]
    for _, row in summary_df.iterrows():
        ci_low = row.get("ci_low", float("nan"))
        ci_high = row.get("ci_high", float("nan"))
        lines.append(
            f"- **{row['edge_mode']} | {row['scenario']} | p={int(row['p_assets'])}**: "
            f"detection_rate={row['detection_rate']:.3f} "
            f"[{ci_low:.3f}, {ci_high:.3f}] over {int(row['trials'])} trials; "
            f"skip_top={row['skip_reason_top'] or 'n/a'}; "
            f"calib_missing_share={row['calibration_missing_share']:.3f}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _candidate_metrics(summary_df: pd.DataFrame, target_fpr: float) -> dict[str, Any]:
    null_rows = summary_df[summary_df["scenario"] == "null"]
    moderate_rows = summary_df[summary_df["scenario"] == "moderate"]
    strong_rows = summary_df[summary_df["scenario"] == "strong"]

    null_ci_high = float(null_rows["ci_high"].max()) if not null_rows.empty else float("nan")
    null_ci_low = float(null_rows["ci_low"].min()) if not null_rows.empty else float("nan")
    null_trials = int(null_rows["trials"].sum()) if not null_rows.empty else 0
    null_detects = int(null_rows["detections"].sum()) if not null_rows.empty else 0
    null_rate = float(null_detects / null_trials) if null_trials > 0 else float("nan")

    power_mod_trials = int(moderate_rows["trials"].sum()) if not moderate_rows.empty else 0
    power_mod_detects = int(moderate_rows["detections"].sum()) if not moderate_rows.empty else 0
    power_mod = (
        float(power_mod_detects / power_mod_trials)
        if power_mod_trials > 0
        else float("nan")
    )
    power_mod_ci = float(moderate_rows["ci_high"].max()) if not moderate_rows.empty else float("nan")

    power_strong_trials = int(strong_rows["trials"].sum()) if not strong_rows.empty else 0
    power_strong_detects = int(strong_rows["detections"].sum()) if not strong_rows.empty else 0
    power_strong = (
        float(power_strong_detects / power_strong_trials)
        if power_strong_trials > 0
        else float("nan")
    )
    power_strong_ci = float(strong_rows["ci_high"].max()) if not strong_rows.empty else float("nan")

    meets = np.isfinite(null_ci_high) and null_ci_high <= target_fpr
    return {
        "null_rate": null_rate,
        "null_ci_high": null_ci_high,
        "null_ci_low": null_ci_low,
        "null_trials": null_trials,
        "power_moderate": power_mod,
        "power_moderate_ci_high": power_mod_ci,
        "power_moderate_trials": power_mod_trials,
        "power_strong": power_strong,
        "power_strong_ci_high": power_strong_ci,
        "power_strong_trials": power_strong_trials,
        "ci_method": "wilson(alpha=0.05)",
        "meets_target": bool(meets),
    }


def _select_best_candidate(
    candidates: list[dict[str, Any]], target_fpr: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for cand in candidates:
        cand["metrics"] = _candidate_metrics(cand["summary"], target_fpr)

    def _power_score(c: dict[str, Any]) -> float:
        power = c["metrics"].get("power_moderate")
        if power is None or not np.isfinite(power):
            return -1.0
        return float(power)

    meeting = [c for c in candidates if c["metrics"]["meets_target"]]
    if meeting:
        best = max(meeting, key=lambda c: (_power_score(c), -float(c["delta_frac"])))
    else:
        best = min(
            candidates,
            key=lambda c: _nan_safe(c["metrics"].get("null_ci_high"), float("inf")),
        )
    return best, candidates


def _write_calibration_file(
    path: Path,
    *,
    selection: dict[str, Any],
    sweep: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_run_name = config.get("run_name")
    run_name = (
        (str(raw_run_name).strip() if raw_run_name else None)
        or os.environ.get("RUN_NAME")
        or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_nested_calib")
    )
    git_sha = _current_git_sha() or "unknown"
    generated_at = datetime.now(timezone.utc).isoformat()
    timestamp_utc = generated_at.replace("+00:00", "Z")
    design = str(config.get("calibration_design") or "nested").lower()

    n_assets_values = _resolve_n_assets(config)
    years = int(config.get("years", 0) or 0)
    replicates = int(config.get("replicates", 0) or 0)
    weeks_options = [int(w) for w in config.get("weeks_options", [])]
    n_obs_values = sorted({years * int(w) * replicates for w in weeks_options if years and replicates})
    thresholds: dict[str, dict[str, Any]] = {}
    operating_points: list[dict[str, Any]] = []
    seen_ops: set[tuple[str, int, int]] = set()

    for edge_mode in config.get("edge_modes", ["tyler"]):
        edge_key = str(edge_mode).lower()
        thresholds[edge_key] = {}
        for p_assets in n_assets_values:
            null_rows = summary_df[
                (summary_df["edge_mode"] == edge_key)
                & (summary_df["scenario"] == "null")
                & (summary_df["p_assets"] == p_assets)
            ]
            null_trials = int(null_rows["trials"].sum()) if not null_rows.empty else 0
            null_detects = int(null_rows["detections"].sum()) if not null_rows.empty else 0
            null_rate = float(null_detects / null_trials) if null_trials > 0 else float("nan")
            null_ci_low = float(null_rows["ci_low"].min()) if not null_rows.empty else float("nan")
            null_ci_high = float(null_rows["ci_high"].max()) if not null_rows.empty else float("nan")
            n_obs_mean = float(null_rows["n_obs_mean"].mean()) if not null_rows.empty else float("nan")
            for n_obs in n_obs_values:
                weeks_common = None
                if years > 0 and replicates > 0 and n_obs > 0:
                    weeks_common = int(round(n_obs / float(replicates * years)))
                entry = {
                    "delta": float(config["delta"]),
                    "delta_frac": float(selection["delta_frac"]),
                    "stability_eta_deg": float(config["stability_eta_deg"]),
                    "target_fpr": float(config.get("target_fpr", 0.02)),
                    "fpr": null_rate,
                    "fpr_ci": [null_ci_low, null_ci_high],
                    "trials_null": null_trials,
                    "run_name": run_name,
                    "git_sha": git_sha,
                    "generated_at": generated_at,
                    "design": design,
                    "edge_mode": edge_key,
                    "n_obs": int(n_obs),
                    "n_obs_mean": n_obs_mean,
                    "p_assets": int(p_assets),
                }
                if weeks_common is not None:
                    entry["weeks_common"] = weeks_common
                if years:
                    entry["years"] = years
                if replicates:
                    entry["replicates"] = replicates
                thresholds[edge_key][f"{int(p_assets)}x{int(n_obs)}"] = entry
                op_id = (edge_key, int(p_assets), int(n_obs))
                if op_id not in seen_ops:
                    seen_ops.add(op_id)
                    operating_points.append(
                        {
                            "edge_mode": edge_key,
                            "p": int(p_assets),
                            "t": int(n_obs),
                            "weeks": weeks_common,
                            "years": years or None,
                            "replicates": replicates or None,
                        }
                    )

    config_payload = {
        key: value
        for key, value in dict(config).items()
        if not str(key).startswith("_")
    }
    config_path = config_payload.get("config_path") or config_payload.get("calibration_config")
    config_path = config.get("_config_path") or config_path
    config_hash = None
    if config_path:
        try:
            config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        except FileNotFoundError:
            config_hash = None

    selection_meta = dict(selection)
    selection_meta.setdefault("target_fpr", float(config.get("target_fpr", 0.02)))
    selection_meta.setdefault("null_rate", selection.get("null_rate"))
    selection_meta.setdefault("null_ci_low", selection.get("null_ci_low"))
    selection_meta.setdefault("null_ci_high", selection.get("null_ci_high"))
    selection_meta.setdefault("null_trials", selection.get("null_trials"))
    selection_meta.setdefault("ci_method", selection.get("ci_method", "wilson(alpha=0.05)"))
    selection_meta.setdefault("power_moderate", selection.get("power_moderate"))
    selection_meta.setdefault("power_moderate_ci_high", selection.get("power_moderate_ci_high"))
    selection_meta.setdefault("power_moderate_trials", selection.get("power_moderate_trials"))
    selection_meta.setdefault("power_strong", selection.get("power_strong"))
    selection_meta.setdefault("power_strong_ci_high", selection.get("power_strong_ci_high"))
    selection_meta.setdefault("power_strong_trials", selection.get("power_strong_trials"))

    payload = {
        "schema_version": "1.0",
        "design": design,
        "alpha": float(config.get("target_fpr", 0.02)),
        "target_fpr": float(config.get("target_fpr", 0.02)),
        "achieved_fpr": selection_meta.get("null_rate"),
        "achieved_fpr_ci_low": selection_meta.get("null_ci_low"),
        "achieved_fpr_ci_high": selection_meta.get("null_ci_high"),
        "generated_at": generated_at,
        "timestamp_utc": timestamp_utc,
        "run_name": run_name,
        "git_sha": git_sha,
        "config": config_payload,
        "selection": selection_meta,
        "sweep": sweep,
        "thresholds": thresholds,
        "design_thresholds": {design: {"thresholds": copy.deepcopy(thresholds)}},
        "metadata": {
            "run_name": run_name,
            "timestamp_utc": timestamp_utc,
            "git_sha": git_sha,
            "config_path": str(config_path) if config_path else None,
            "config_hash": config_hash,
            "config_edge_modes": config_payload.get("edge_modes", []),
            "target_fpr": float(config.get("target_fpr", 0.02)),
            "achieved_fpr": selection_meta.get("null_rate"),
            "achieved_fpr_ci_low": selection_meta.get("null_ci_low"),
            "achieved_fpr_ci_high": selection_meta.get("null_ci_high"),
            "ci_method": selection_meta.get("ci_method", "wilson(alpha=0.05)"),
            "null_trials": selection_meta.get("null_trials"),
            "power_moderate": selection_meta.get("power_moderate"),
            "power_moderate_ci_high": selection_meta.get("power_moderate_ci_high"),
            "power_moderate_trials": selection_meta.get("power_moderate_trials"),
            "power_strong": selection_meta.get("power_strong"),
            "power_strong_ci_high": selection_meta.get("power_strong_ci_high"),
            "power_strong_trials": selection_meta.get("power_strong_trials"),
            "trials_per_scenario": config_payload.get("trials_per_scenario"),
            "operating_points": operating_points,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nested design synthetic kill-test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/synthetic/config.nested.killtest.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=None,
        help="FPR target for calibration / selection (default from config).",
    )
    parser.add_argument(
        "--delta-frac-grid",
        type=str,
        default=None,
        help="Comma-separated delta_frac candidates for sweep.",
    )
    parser.add_argument(
        "--calibration-out",
        type=Path,
        default=None,
        help="Optional path to write calibration JSON.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name to embed in calibration metadata.",
    )
    parser.add_argument(
        "--calibration-design",
        type=str,
        default=None,
        help="Design label for calibration lookup (e.g., nested).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    if args.target_fpr is not None:
        config["target_fpr"] = args.target_fpr
    if args.delta_frac_grid is not None:
        config["delta_frac_grid"] = args.delta_frac_grid
    if args.calibration_out is not None:
        config["calibration_out"] = str(args.calibration_out)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.calibration_design is not None:
        config["calibration_design"] = args.calibration_design
    out_dir = Path(args.out) if args.out is not None else Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_raw = config.get("delta_frac_grid")
    if isinstance(grid_raw, str):
        grid = [float(x) for x in grid_raw.split(",") if x.strip()]
    elif grid_raw is None:
        grid = [float(config["delta_frac_min"])]
    else:
        grid = [float(x) for x in grid_raw]
    if not grid:
        grid = [float(config["delta_frac_min"])]

    target_fpr = float(config.get("target_fpr", 0.02))
    candidates: list[dict[str, Any]] = []
    start = time.time()
    for delta_val in grid:
        candidate_cfg = config.copy()
        candidate_cfg["delta_frac_min"] = float(delta_val)
        records, diag_summary = run_trials(candidate_cfg)
        summary_df = summarise_results(records)
        candidates.append(
            {
                "delta_frac": float(delta_val),
                "records": records,
                "summary": summary_df,
                "diagnostics": diag_summary,
                "config": candidate_cfg,
            }
        )
    elapsed = time.time() - start

    best, scored = _select_best_candidate(candidates, target_fpr)
    summary_df = best["summary"]
    records = best["records"]
    diag_summary = best["diagnostics"]
    best_config = best["config"]

    df = pd.DataFrame([r.__dict__ for r in records])
    df.to_csv(out_dir / "nested_killtest_trials.csv", index=False)

    summary_df.to_csv(out_dir / "summary.csv", index=False)
    write_summary_markdown(summary_df, out_dir / "summary.md")

    sweep_table = []
    for cand in scored:
        row = {
            "delta_frac": cand["delta_frac"],
            **cand["metrics"],
        }
        sweep_table.append(row)
    pd.DataFrame(sweep_table).to_csv(out_dir / "sweep.csv", index=False)

    selection_meta = {
        "delta_frac": float(best["delta_frac"]),
        **best["metrics"],
    }

    config_payload = {
        key: value
        for key, value in dict(best_config).items()
        if not str(key).startswith("_")
    }
    config_payload["config_path"] = str(args.config)
    config_payload["config_hash"] = config_hash
    resolved_path = out_dir / "resolved_config.json"
    resolved_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    resolved_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _current_git_sha() or "unknown",
        "elapsed_seconds": elapsed,
        "config": config_payload,
        "config_path": str(args.config),
        "config_hash": config_hash,
        "resolved_config_path": str(resolved_path),
        "resolved_config_hash": resolved_hash,
        "diagnostics": diag_summary,
        "selection": selection_meta,
        "sweep": sweep_table,
        "artifacts": {
            "trials": str(out_dir / "nested_killtest_trials.csv"),
            "summary": str(out_dir / "summary.csv"),
            "summary_md": str(out_dir / "summary.md"),
            "sweep": str(out_dir / "sweep.csv"),
            "resolved_config": str(resolved_path),
        },
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    calib_path_cfg = config.get("calibration_out")
    if calib_path_cfg:
        _write_calibration_file(
            Path(calib_path_cfg),
            selection=selection_meta,
            sweep=sweep_table,
            summary_df=summary_df,
            config=best_config,
        )

    print(
        f"[nested-killtest] wrote {out_dir} (best delta_frac={best['delta_frac']:.4f}, "
        f"null_ci_high={selection_meta.get('null_ci_high'):.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
