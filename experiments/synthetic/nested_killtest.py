from __future__ import annotations

import argparse
import json
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
    "years": 2,
    "weeks_options": [6, 7, 8],
    "replicates": 5,
    "trials_per_scenario": 80,
    "spikes": {"null": 0.0, "moderate": 3.0, "strong": 6.0},
    "noise_variance": 1.0,
    "signal_to_noise": 0.35,
    "edge_modes": ["tyler"],
    "edge_huber_c": 1.5,
    "delta": 0.35,
    "delta_frac_min": 0.05,
    "eps": 1.0,
    "stability_eta_deg": 0.3,
    "a_grid": 96,
    "cs_drop_top_frac": 0.1,
    "cs_sensitivity_frac": 0.0,
    "off_component_leak_cap": 0.3,
    "energy_min_abs": 2e-7,
    "allow_nonisolated": False,
    "nonisolated_stability_min": 0.015,
    "nonisolated_edge_min": 0.015,
    "nonisolated_q_max": 2,
    "require_isolated": True,
    "use_tvector": True,
    "q_max": 2,
    "calibration_path": "calibration/edge_delta_thresholds.json",
    "seed": 0,
    "out_dir": "reports/synthetic_nested_killtest",
}


@dataclass(frozen=True)
class TrialResult:
    scenario: str
    trial: int
    weeks_common: int
    n_obs: int
    detected: bool
    skip_reason: str
    isolated_spikes: int
    edge_mode: str
    delta_frac_used: float
    calibration_missing: bool


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return DEFAULT_CONFIG.copy()
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Config must be a mapping.")
    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in data.items() if v is not None})
    return merged


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
    n_assets = int(config["n_assets"])
    trials_per = int(config["trials_per_scenario"])
    edge_modes = [str(mode).lower() for mode in config.get("edge_modes", ["tyler"])]

    for edge_mode in edge_modes:
        skip_reasons_accum: Counter[str] = Counter()
        for scenario_label, spike_strength in scenarios:
            for trial in range(trials_per):
                weeks = int(rng.choice(weeks_options))
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
                            detected=False,
                            skip_reason="prep_failure",
                            isolated_spikes=0,
                            edge_mode=edge_mode,
                            delta_frac_used=float(config["delta_frac_min"]),
                            calibration_missing=True,
                        )
                    )
                    continue

                sigma1 = ((ms1 - ms2) / float(meta.J * meta.replicates)).astype(np.float64, copy=False)
                sigma2 = ((ms2 - ms3) / float(meta.replicates)).astype(np.float64, copy=False)
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
                )
                calibration_missing = delta_frac_calib is None
                base_delta_frac = float(config["delta_frac_min"])
                delta_frac_used = (
                    base_delta_frac
                    if delta_frac_calib is None
                    else max(base_delta_frac, float(delta_frac_calib))
                )
                use_tvector = bool(config.get("use_tvector", config.get("require_isolated", False)))

                edge_scale, edge_scm_val, edge_sel_val = _edge_scale(
                    observations, edge_mode=edge_mode, edge_huber_c=float(config["edge_huber_c"])
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
                    use_tvector=use_tvector,
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
                isolated_count = count_isolated_outliers(detections, None, None)
                window_skip_reason: str | None = None
                candidate_pool = list(detections)

                allow_noniso = bool(config["allow_nonisolated"])
                require_isolated = bool(config.get("require_isolated", False))

                if require_isolated:
                    if isolated_count == 0:
                        window_skip_reason = "no_isolated_spike"
                        candidate_pool = []
                    else:
                        filtered_iso: list[dict[str, Any]] = []
                        for det in candidate_pool:
                            if not isinstance(det, Mapping):
                                continue
                            try:
                                pre_val = int(det.get("pre_outlier_count", 0))
                            except (TypeError, ValueError):
                                pre_val = 0
                            if pre_val == 1:
                                filtered_iso.append(det)
                        if filtered_iso:
                            candidate_pool = filtered_iso
                        else:
                            window_skip_reason = "no_isolated_spike"
                            candidate_pool = []

                if allow_noniso and candidate_pool:
                    filtered = []
                    for det in candidate_pool:
                        try:
                            stab = float(det.get("stability_margin", 0.0))
                        except Exception:
                            stab = 0.0
                        try:
                            edge_val = float(det.get("edge_margin", 0.0))
                        except Exception:
                            edge_val = 0.0
                        if (
                            np.isfinite(stab)
                            and stab >= float(config["nonisolated_stability_min"])
                            and np.isfinite(edge_val)
                            and edge_val >= float(config["nonisolated_edge_min"])
                        ):
                            filtered.append(det)
                    if filtered:
                        candidate_pool = filtered
                    else:
                        window_skip_reason = window_skip_reason or "nested_guard"
                        candidate_pool = []

                if candidate_pool and int(config["q_max"]) > 0 and len(candidate_pool) > int(config["q_max"]):
                    candidate_pool, _ = select_top_k(candidate_pool, int(config["q_max"]))

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
                        detected=detected,
                        skip_reason=window_skip_reason or "",
                        isolated_spikes=int(isolated_count),
                        edge_mode=edge_mode,
                        delta_frac_used=float(delta_frac_used),
                        calibration_missing=bool(calibration_missing),
                    )
                )
        diag_summary[edge_mode] = dict(skip_reasons_accum)

    return records, diag_summary


def summarise_results(results: Sequence[TrialResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in results])
    summaries: list[dict[str, Any]] = []
    for (edge_mode, scenario), group in df.groupby(["edge_mode", "scenario"]):
        skip_counts = (
            group["skip_reason"].replace("", np.nan).dropna().value_counts().to_dict()
            if not group.empty
            else {}
        )
        summaries.append(
            {
                "edge_mode": edge_mode,
                "scenario": scenario,
                "trials": int(group.shape[0]),
                "detection_rate": float(group["detected"].mean()) if not group.empty else 0.0,
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
        lines.append(
            f"- **{row['edge_mode']} | {row['scenario']}**: "
            f"detection_rate={row['detection_rate']:.3f} over {int(row['trials'])} trials; "
            f"skip_top={row['skip_reason_top'] or 'n/a'}; "
            f"calib_missing_share={row['calibration_missing_share']:.3f}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


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
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = Path(args.out) if args.out is not None else Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    records, diag_summary = run_trials(config)
    elapsed = time.time() - start

    df = pd.DataFrame([r.__dict__ for r in records])
    df.to_csv(out_dir / "nested_killtest_trials.csv", index=False)

    summary_df = summarise_results(records)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    write_summary_markdown(summary_df, out_dir / "summary.md")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "config": config,
        "diagnostics": diag_summary,
        "artifacts": {
            "trials": str(out_dir / "nested_killtest_trials.csv"),
            "summary": str(out_dir / "summary.csv"),
            "summary_md": str(out_dir / "summary.md"),
        },
    }
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[nested-killtest] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
