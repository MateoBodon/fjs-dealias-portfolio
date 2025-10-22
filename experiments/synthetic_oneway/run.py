from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fjs.balanced import mean_squares
from fjs.dealias import dealias_search
from fjs.spectra import plot_spectrum_with_edges, plot_spike_timeseries, topk_eigh


def load_config(path: Path | str) -> dict[str, Any]:
    """Load the synthetic experiment configuration from disk."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    return data


def _simulate_balanced_panel(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(config.get("seed", 0)))
    n_assets = int(config.get("n_assets", 60))
    n_groups = int(config.get("n_samples", 60))
    replicates = int(config.get("replicates", 2))
    spike_strength = float(config.get("spike_strength", 3.0))
    noise_variance = float(config.get("noise_variance", 1.0))
    signal_to_noise = float(config.get("signal_to_noise", 0.5))

    signal_dir = rng.normal(size=n_assets)
    signal_dir /= np.linalg.norm(signal_dir)
    aux_dir = rng.normal(size=n_assets)
    aux_dir /= np.linalg.norm(aux_dir)

    observations = np.zeros((n_groups * replicates, n_assets), dtype=np.float64)
    groups = np.repeat(np.arange(n_groups, dtype=np.int64), replicates)

    spike_scale = np.sqrt(spike_strength)
    noise_scale = np.sqrt(noise_variance)

    idx = 0
    for _ in range(n_groups):
        factor = spike_scale * rng.normal()
        group_effect = factor * signal_dir
        for _ in range(replicates):
            common_noise = signal_to_noise * noise_scale * rng.normal() * aux_dir
            idiosyncratic_noise = noise_scale * rng.normal(size=n_assets)
            observations[idx] = group_effect + common_noise + idiosyncratic_noise
            idx += 1

    return observations, groups


def _mp_edges(
    noise_variance: float, n_assets: int, sample_count: int
) -> tuple[float, float]:
    aspect_ratio = n_assets / sample_count
    sqrt_ratio = np.sqrt(aspect_ratio)
    upper = noise_variance * (1.0 + sqrt_ratio) ** 2
    lower = noise_variance * max(0.0, (1.0 - sqrt_ratio) ** 2)
    return lower, upper


def _series_bias(
    observations: np.ndarray,
    groups: np.ndarray,
    true_spike: float,
    *,
    delta: float,
    eps: float,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    prefixes: list[int] = []
    aliased: list[float] = []
    dealiased: list[float] = []

    for end in range(step, unique_groups.size + 1, step):
        mask = groups < end
        ms_stats = mean_squares(observations[mask], groups[mask])
        eigenvalues, _ = topk_eigh(ms_stats["MS1"], 1)
        lambda_hat = float(eigenvalues[0])
        detections = dealias_search(
            observations[mask],
            groups[mask],
            target_r=0,
            a_grid=90,
            delta=delta,
            eps=eps,
        )
        mu_hat = detections[0]["mu_hat"] if detections else np.nan

        prefixes.append(end)
        aliased.append(lambda_hat)
        dealiased.append(mu_hat)

    return np.asarray(prefixes), np.asarray(aliased), np.asarray(dealiased)


def _output_directory(config: dict[str, Any]) -> Path:
    if "output_dir" in config:
        return Path(config["output_dir"])
    return Path(__file__).with_name("outputs")


def run_experiment(config_path: Path | str | None = None) -> None:
    """Execute the synthetic MANOVA experiment pipeline."""

    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("config.yaml")
    )
    config = load_config(path)

    observations, groups = _simulate_balanced_panel(config)
    ms_stats = mean_squares(observations, groups)
    ms1 = ms_stats["MS1"].astype(np.float64)

    eigenvalues = np.linalg.eigvalsh(ms1)
    n_assets = ms1.shape[0]
    total_samples = observations.shape[0]
    lower_edge, upper_edge = _mp_edges(
        float(config.get("noise_variance", 1.0)), n_assets, total_samples
    )

    output_dir = _output_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectrum_png = output_dir / "spectrum.png"
    spectrum_pdf = output_dir / "spectrum.pdf"
    plot_spectrum_with_edges(
        eigenvalues,
        edges=(lower_edge, upper_edge),
        out_path=spectrum_png,
        title="Empirical MS1 spectrum",
    )
    plot_spectrum_with_edges(
        eigenvalues,
        edges=(lower_edge, upper_edge),
        out_path=spectrum_pdf,
        title="Empirical MS1 spectrum",
    )

    signal_strength = float(config.get("spike_strength", 3.0))
    prefixes, aliased, dealiased = _series_bias(
        observations,
        groups,
        signal_strength,
        delta=0.3,
        eps=0.05,
        step=max(5, int(groups.max() + 1) // 20 or 1),
    )

    spike_png = output_dir / "spike_timeseries.png"
    spike_pdf = output_dir / "spike_timeseries.pdf"
    plot_spike_timeseries(
        prefixes,
        aliased,
        dealiased,
        out_path=spike_png,
        title="Outlier tracking",
        true_value=signal_strength,
    )
    plot_spike_timeseries(
        prefixes,
        aliased,
        dealiased,
        out_path=spike_pdf,
        title="Outlier tracking",
        true_value=signal_strength,
    )

    table = pd.DataFrame(
        {
            "prefix_groups": prefixes,
            "aliased": aliased,
            "dealiased": dealiased,
            "aliased_bias": aliased - signal_strength,
            "dealiased_bias": dealiased - signal_strength,
        }
    )
    table_path = output_dir / "bias_table.csv"
    table.to_csv(table_path, index=False)


def main() -> None:
    """Entry point for CLI execution."""

    run_experiment()


if __name__ == "__main__":
    main()
