# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fjs.balanced import mean_squares
from fjs.dealias import dealias_search
from fjs.spectra import plot_spike_timeseries

DEFAULT_CONFIG = {
    "n_assets": 60,
    "n_groups": 60,
    "replicates": 2,
    "noise_variance": 1.0,
    "signal_to_noise": 0.5,
    "spike_strength": 6.0,
    "mc": 200,
    "snr_grid": [4.0, 6.0, 8.0],
    "output_dir": "figures/synthetic",
}


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return DEFAULT_CONFIG.copy()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    config = DEFAULT_CONFIG.copy()
    config.update(data)
    return config


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def simulate_panel(
    rng: np.random.Generator,
    *,
    n_assets: int,
    n_groups: int,
    replicates: int,
    spike_strength: float,
    noise_variance: float,
    signal_to_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.zeros((n_groups * replicates, n_assets), dtype=np.float64)
    groups = np.repeat(np.arange(n_groups, dtype=np.intp), replicates)

    signal_dir = rng.normal(size=n_assets)
    signal_dir /= np.linalg.norm(signal_dir)
    aux_dir = rng.normal(size=n_assets)
    aux_dir /= np.linalg.norm(aux_dir)

    noise_scale = np.sqrt(noise_variance)
    spike_scale = np.sqrt(spike_strength)

    idx = 0
    for _ in range(n_groups):
        factor = spike_scale * rng.normal()
        group_effect = factor * signal_dir
        for _ in range(replicates):
            common_noise = signal_to_noise * noise_scale * rng.normal() * aux_dir
            idio_noise = noise_scale * rng.normal(size=n_assets)
            observations[idx] = group_effect + common_noise + idio_noise
            idx += 1
    return observations, groups


def mp_upper_edge(noise_variance: float, n_assets: int, n_groups: int) -> float:
    aspect_ratio = n_assets / max(n_groups - 1, 1)
    sqrt_ratio = np.sqrt(aspect_ratio)
    return noise_variance * (1.0 + sqrt_ratio) ** 2


def histogram_s1(
    eigenvalues: Sequence[float],
    edge: float,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eigenvalues, bins=50, alpha=0.7, color="C0", label="Eigenvalues")
    ax.axvline(edge, color="C1", linestyle="--", linewidth=1.5, label="MP edge")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Frequency")
    ax.set_title("S1: Spectrum of $\\Sigma_1$")
    ax.legend()
    fig.savefig(out_dir / "s1_histogram.png", bbox_inches="tight")
    fig.savefig(out_dir / "s1_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


def bias_table_s3(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    df.to_csv(out_dir / "bias_table.csv", index=False)


def summary_to_json(summary: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)


def s1_monte_carlo(config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    eigenvalues_accum: list[float] = []
    top_eigs: list[float] = []

    for _ in range(int(config["mc"])):
        y_mat, groups = simulate_panel(
            rng,
            n_assets=config["n_assets"],
            n_groups=config["n_groups"],
            replicates=config["replicates"],
            spike_strength=config["spike_strength"],
            noise_variance=config["noise_variance"],
            signal_to_noise=config["signal_to_noise"],
        )
        stats = mean_squares(y_mat, groups)
        eigenvalues = np.linalg.eigvalsh(stats["MS1"].astype(np.float64))
        eigenvalues_accum.extend(eigenvalues.tolist())
        top_eigs.append(float(eigenvalues[-1]))

    noise_estimate = float(np.median(eigenvalues_accum))
    edge = mp_upper_edge(noise_estimate, config["n_assets"], config["n_groups"])
    histogram_s1(eigenvalues_accum, edge=edge, out_dir=Path(config["output_dir"]))

    return {
        "s1_noise_estimate": noise_estimate,
        "s1_edge": edge,
        "s1_top_mean": float(np.mean(top_eigs)),
    }


def s3_bias(config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snr_grid = config.get("snr_grid", [4.0, 6.0, 8.0])
    mc = int(config["mc"])
    delta = float(config.get("delta", 0.3))
    eps = float(config.get("eps", 0.05))

    for spike in snr_grid:
        aliased_vals: list[float] = []
        dealiased_vals: list[float] = []
        detects = 0

        for _ in range(mc):
            y_mat, groups = simulate_panel(
                rng,
                n_assets=config["n_assets"],
                n_groups=config["n_groups"],
                replicates=config["replicates"],
                spike_strength=spike,
                noise_variance=config["noise_variance"],
                signal_to_noise=config["signal_to_noise"],
            )
            stats = mean_squares(y_mat, groups)
            eigenvalues = np.linalg.eigvalsh(stats["MS1"].astype(np.float64))
            aliased_vals.append(float(eigenvalues[-1]))

            detections = dealias_search(
                y_mat,
                groups,
                target_r=0,
                a_grid=72,
                delta=delta,
                eps=eps,
            )
            if detections:
                detects += 1
                dealiased_vals.append(float(detections[0]["mu_hat"]))
            else:
                dealiased_vals.append(np.nan)

        aliased_arr = np.asarray(aliased_vals)
        dealiased_arr = np.asarray(dealiased_vals)

        row = {
            "spike_strength": spike,
            "aliased_mean": float(np.mean(aliased_arr)),
            "aliased_bias": float(np.mean(aliased_arr) - spike),
            "dealiased_mean": float(np.nanmean(dealiased_arr)),
            "dealiased_bias": float(np.nanmean(dealiased_arr) - spike),
            "detection_rate": detects / mc,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    bias_table_s3(df, Path(config["output_dir"]))
    return df


def plot_bias_timeseries(
    prefixes: Sequence[int],
    aliased: Sequence[float],
    dealiased: Sequence[float],
    spike: float,
    output_dir: Path,
) -> None:
    plot_spike_timeseries(
        prefixes,
        aliased,
        dealiased,
        out_path=output_dir / f"bias_timeseries_mu{int(spike)}.png",
        title=f"S3 Bias tracking µ={spike}",
        true_value=spike,
        xlabel="Prefix",
        ylabel="Estimate",
    )


def run_experiment(
    config_path: Path | str | None = None,
    *,
    seed: int | None = None,
) -> None:
    config = load_config(Path(config_path) if config_path else None)
    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)

    rng = np.random.default_rng(seed if seed is not None else config.get("seed", 0))

    summary = s1_monte_carlo(config, rng)
    bias_df = s3_bias(config, rng)

    summary["s3_bias"] = bias_df.to_dict(orient="records")
    summary_to_json(summary, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic de-aliasing experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run_experiment(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
