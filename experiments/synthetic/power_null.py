# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.synthetic_oneway.run import simulate_panel, simulate_multi_spike
from fjs.dealias import dealias_search
from fjs.robust import edge_from_scatter, huber_scatter, tyler_scatter
from finance.eval import oos_variance_forecast
from evaluation.evaluate import qlike as eval_qlike


DEFAULT_CONFIG = {
    "n_assets": 40,
    "n_groups": 60,
    "replicates": 3,
    "noise_variance": 1.0,
    "signal_to_noise": 0.35,
    "delta": 0.05,
    "delta_frac": 0.02,
    "eps": 0.03,
    "stability_eta_deg": 0.4,
    "a_grid": 120,
    "cs_drop_top_frac": 0.05,
    "cs_sensitivity_frac": 0.0,
    "trials_null": 200,
    "trials_power": 150,
    "spike_grid": [2.0, 3.5, 5.0, 6.5],
    "two_spike": False,
    "output_dir": "experiments/synthetic/outputs",
}

GATING_SETTINGS: Mapping[str, Mapping[str, object]] = {
    "default": {"enable": True, "require_isolated": True, "q_max": 2},
    "loose": {"enable": True, "require_isolated": False, "q_max": 3},
}


def _edge_scale_for_mode(y: np.ndarray, mode: str, huber_c: float = 1.5) -> tuple[float, float, float]:
    p = y.shape[1]
    n = y.shape[0]
    cov = np.cov(y, rowvar=False, ddof=1)
    cov = 0.5 * (cov + cov.T)
    edge_scm = edge_from_scatter(cov, p, n)
    if mode == "scm":
        return 1.0, edge_scm, edge_scm
    if mode == "tyler":
        try:
            scatter = tyler_scatter(y)
            edge_alt = edge_from_scatter(scatter, p, n)
        except Exception:
            return 1.0, edge_scm, edge_scm
    elif mode == "huber":
        try:
            scatter = huber_scatter(y, huber_c)
            edge_alt = edge_from_scatter(scatter, p, n)
        except Exception:
            return 1.0, edge_scm, edge_scm
    else:
        return 1.0, edge_scm, edge_scm
    if not (np.isfinite(edge_scm) and edge_scm > 0 and np.isfinite(edge_alt) and edge_alt > 0):
        return 1.0, edge_scm, edge_alt
    scale = float(edge_alt / edge_scm)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale, edge_scm, edge_alt


@dataclass
class TrialResult:
    edge_mode: str
    gating: str
    scenario: str
    spike_strength: float | None
    detected: bool
    delta_mse_vs_lw: float | None
    delta_qlike_vs_lw: float | None


def _detections_for_mode(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    edge_mode: str,
    gating: Mapping[str, object],
    config: Mapping[str, object],
) -> list[dict]:
    scale, _, _ = _edge_scale_for_mode(y, edge_mode)
    detections = dealias_search(
        y,
        groups,
        target_r=0,
        delta=float(config["delta"]),
        delta_frac=float(config["delta_frac"]),
        eps=float(config["eps"]),
        stability_eta_deg=float(config["stability_eta_deg"]),
        cs_drop_top_frac=float(config["cs_drop_top_frac"]),
        cs_sensitivity_frac=float(config["cs_sensitivity_frac"]),
        a_grid=int(config["a_grid"]),
        edge_scale=scale,
        edge_mode=edge_mode,
    )
    if gating.get("enable", True):
        require_iso = bool(gating.get("require_isolated", True))
        if require_iso:
            detections = [det for det in detections if int(det.get("pre_outlier_count", 0)) == 1]
        q_max = int(gating.get("q_max", 0))
        if q_max > 0 and len(detections) > q_max:
            detections = sorted(detections, key=lambda d: float(d.get("target_energy", 0.0)) * float(d.get("stability_margin", 0.0)), reverse=True)[:q_max]
    return detections


def _simulate_null(config: Mapping[str, object], *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return simulate_panel(
        rng,
        n_assets=int(config["n_assets"]),
        n_groups=int(config["n_groups"]),
        replicates=int(config["replicates"]),
        spike_strength=0.0,
        noise_variance=float(config["noise_variance"]),
        signal_to_noise=float(config["signal_to_noise"]),
    )


def _simulate_power(
    config: Mapping[str, object],
    *,
    rng: np.random.Generator,
    strength: float,
    two_spike: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if two_spike:
        observations, groups, _ = simulate_multi_spike(
            rng,
            n_assets=int(config["n_assets"]),
            n_groups=int(config["n_groups"]),
            replicates=int(config["replicates"]),
            spike_strengths=[strength, max(strength / 2.0, 0.5)],
            noise_variance=float(config["noise_variance"]),
            signal_to_noise=float(config["signal_to_noise"]),
        )
        return observations, groups
    observations, groups = simulate_panel(
        rng,
        n_assets=int(config["n_assets"]),
        n_groups=int(config["n_groups"]),
        replicates=int(config["replicates"]),
        spike_strength=strength,
        noise_variance=float(config["noise_variance"]),
        signal_to_noise=float(config["signal_to_noise"]),
    )
    return observations, groups


def run_trials(
    *,
    config: Mapping[str, object],
    edge_modes: Iterable[str],
    trials_null: int,
    trials_power: int,
    spike_grid: Iterable[float],
    two_spike: bool,
    rng: np.random.Generator,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    gating_labels = list(GATING_SETTINGS.keys())

    for _ in range(trials_null):
        y, groups = _simulate_null(config, rng=rng)
        for gating_label in gating_labels:
            gating = GATING_SETTINGS[gating_label]
            for mode in edge_modes:
                detections = _detections_for_mode(y, groups, edge_mode=mode, gating=gating, config=config)
                results.append(
                    TrialResult(
                        edge_mode=mode,
                        gating=gating_label,
                        scenario="null",
                        spike_strength=None,
                        detected=bool(detections),
                        delta_mse_vs_lw=None,
                        delta_qlike_vs_lw=None,
                    )
                )

    weights = np.full(int(config["n_assets"]), 1.0 / float(int(config["n_assets"])))

    for strength in spike_grid:
        for _ in range(trials_power):
            y_fit, groups = _simulate_power(config, rng=rng, strength=float(strength), two_spike=two_spike)
            # Hold-out sample from the same distribution for realised variance
            y_hold, _ = _simulate_power(config, rng=rng, strength=float(strength), two_spike=two_spike)
            for gating_label in gating_labels:
                gating = GATING_SETTINGS[gating_label]
                for mode in edge_modes:
                    detections = _detections_for_mode(y_fit, groups, edge_mode=mode, gating=gating, config=config)
                    detected = bool(detections)
                    forecast_de, realised_de = oos_variance_forecast(
                        y_fit,
                        y_hold,
                        weights,
                        estimator="dealias",
                        detections=detections,
                    )
                    forecast_lw, realised_lw = oos_variance_forecast(
                        y_fit,
                        y_hold,
                        weights,
                        estimator="lw",
                    )
                    mse_de = (forecast_de - realised_de) ** 2
                    mse_lw = (forecast_lw - realised_lw) ** 2
                    ql_de = float(eval_qlike([forecast_de], [realised_de])[0])
                    ql_lw = float(eval_qlike([forecast_lw], [realised_lw])[0])
                    results.append(
                        TrialResult(
                            edge_mode=mode,
                            gating=gating_label,
                            scenario="power",
                            spike_strength=float(strength),
                            detected=detected,
                            delta_mse_vs_lw=float(mse_de - mse_lw),
                            delta_qlike_vs_lw=float(ql_de - ql_lw),
                        )
                    )
    return results


def summarise_results(results: list[TrialResult]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in results])

    for (scenario, edge_mode, gating, strength), group in df.groupby(
        ["scenario", "edge_mode", "gating", "spike_strength"], dropna=False
    ):
        detection_rate = float(group["detected"].mean()) if not group.empty else float("nan")
        delta_mse = float(group["delta_mse_vs_lw"].mean()) if group["delta_mse_vs_lw"].notna().any() else float("nan")
        delta_qlike = float(group["delta_qlike_vs_lw"].mean()) if group["delta_qlike_vs_lw"].notna().any() else float("nan")
        records.append(
            {
                "scenario": scenario,
                "edge_mode": edge_mode,
                "gating": gating,
                "spike_strength": strength if pd.notna(strength) else "",
                "detection_rate": detection_rate,
                "delta_mse_vs_lw": delta_mse,
                "delta_qlike_vs_lw": delta_qlike,
            }
        )
    return pd.DataFrame(records)


def plot_fpr_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    subset = summary[(summary["scenario"] == "null")]
    if subset.empty:
        return
    pivot = subset.pivot(index="gating", columns="edge_mode", values="detection_rate")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", vmin=0.0, vmax=0.2)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Null FPR by edge_mode and gating")
    for (i, j), value in np.ndenumerate(pivot.to_numpy()):
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="False positive rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_power_curves(summary: pd.DataFrame, out_path: Path) -> None:
    subset = summary[(summary["scenario"] == "power")]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (edge_mode, gating), group in subset.groupby(["edge_mode", "gating"]):
        group_sorted = group.sort_values("spike_strength")
        ax.plot(
            group_sorted["spike_strength"].astype(float),
            group_sorted["detection_rate"],
            marker="o",
            label=f"{edge_mode} ({gating})",
        )
    ax.set_xlabel("Spike strength")
    ax.set_ylabel("Detection rate")
    ax.set_title("Power curves")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic null/power diagnostics for edge modes")
    parser.add_argument("--design", type=str, default="oneway", help="Design placeholder (currently oneway only).")
    parser.add_argument(
        "--edge-modes",
        nargs="+",
        default=["scm", "tyler"],
        help="Edge modes to evaluate (subset of {scm, tyler, huber}).",
    )
    parser.add_argument("--trials-null", type=int, default=None, help="Number of null trials (default 200).")
    parser.add_argument("--trials-power", type=int, default=None, help="Number of power trials per strength (default 150).")
    parser.add_argument(
        "--spike-grid",
        type=float,
        nargs="+",
        default=None,
        help="Spike strength grid for power analysis (default [2.0,3.5,5.0,6.5]).",
    )
    parser.add_argument("--two-spike", action="store_true", help="Inject two spikes instead of one in the power design.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store summary and plots (default experiments/synthetic/outputs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    if args.trials_null is not None:
        config["trials_null"] = int(args.trials_null)
    if args.trials_power is not None:
        config["trials_power"] = int(args.trials_power)
    if args.spike_grid is not None and args.spike_grid:
        config["spike_grid"] = [float(x) for x in args.spike_grid]
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    edge_modes = [mode.lower() for mode in args.edge_modes]
    valid_modes = {"scm", "tyler", "huber"}
    for mode in edge_modes:
        if mode not in valid_modes:
            raise ValueError(f"Unsupported edge mode '{mode}'. Valid options: {sorted(valid_modes)}")

    results = run_trials(
        config=config,
        edge_modes=edge_modes,
        trials_null=int(config["trials_null"]),
        trials_power=int(config["trials_power"]),
        spike_grid=[float(x) for x in config["spike_grid"]],
        two_spike=bool(args.two_spike),
        rng=rng,
    )
    summary = summarise_results(results)
    summary_path = output_dir / "power_null_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_fpr_heatmap(summary, output_dir / "fpr_heatmap.png")
    plot_power_curves(summary, output_dir / "power_curves.png")

    meta = {
        "config": config,
        "edge_modes": [mode.lower() for mode in args.edge_modes],
        "two_spike": bool(args.two_spike),
        "design": args.design,
    }
    (output_dir / "power_null_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
