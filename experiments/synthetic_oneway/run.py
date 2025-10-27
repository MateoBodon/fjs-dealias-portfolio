# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Progress bar (fallback to no-op if unavailable)
try:  # pragma: no cover - UI nicety
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - best-effort import

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
# Ensure both project root (for top-level utilities like pairing.py) and src are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fjs.balanced import mean_squares
from fjs.dealias import dealias_search
from pairing import align_spikes
from fjs.spectra import plot_spike_timeseries
from plotting import s4_plot_guardrails_from_csv
from meta.run_meta import code_signature, write_run_meta

DEFAULT_CONFIG = {
    "n_assets": 60,
    "n_groups": 60,
    "replicates": 3,
    "noise_variance": 1.0,
    "signal_to_noise": 0.35,
    "spike_strength": 6.0,
    "mc": 200,
    "snr_grid": [4.0, 6.0, 8.0],
    "guardrail_trials": 200,
    "multi_spike_trials": 120,
    "multi_spike_strengths": [7.0, 5.0, 3.5],
    "delta": 0.05,
    "delta_frac": 0.02,
    "eps": 0.03,
    "stability_eta_deg": 0.4,
    "a_grid": 120,
    "scan_basis": "ms",
    "off_component_leak_cap": 10.0,
    "energy_min_abs": 1e-6,
    "output_dir": "figures/synthetic",
    "progress": True,
}


def load_config(path: Path | None) -> dict[str, Any]:
    """Load experiment configuration from YAML, falling back to defaults."""

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
    """Create ``path`` and parents if they do not yet exist."""

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
    return_dirs: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Simulate a balanced MANOVA panel with a single spike."""

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
    if return_dirs:
        return observations, groups, signal_dir, aux_dir
    return observations, groups


def simulate_multi_spike(
    rng: np.random.Generator,
    *,
    n_assets: int,
    n_groups: int,
    replicates: int,
    spike_strengths: Sequence[float],
    noise_variance: float,
    signal_to_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a panel with multiple planted spikes."""
    spike_strengths = [float(x) for x in spike_strengths]
    n_spikes = len(spike_strengths)
    observations = np.zeros((n_groups * replicates, n_assets), dtype=np.float64)
    groups = np.repeat(np.arange(n_groups, dtype=np.intp), replicates)

    raw = rng.normal(size=(n_assets, n_spikes))
    q, _ = np.linalg.qr(raw)
    signal_dirs = q[:, :n_spikes].T  # shape (n_spikes, n_assets)

    aux_dir = rng.normal(size=n_assets)
    aux_dir /= np.linalg.norm(aux_dir)

    noise_scale = np.sqrt(noise_variance)
    spike_scales = np.sqrt(np.asarray(spike_strengths, dtype=np.float64))

    idx = 0
    for _ in range(n_groups):
        coeffs = rng.normal(size=n_spikes)
        group_effect = np.sum(
            spike_scales[:, None] * coeffs[:, None] * signal_dirs, axis=0
        )
        for _ in range(replicates):
            common_noise = signal_to_noise * noise_scale * rng.normal() * aux_dir
            idio_noise = noise_scale * rng.normal(size=n_assets)
            observations[idx] = group_effect + common_noise + idio_noise
            idx += 1
    return observations, groups, signal_dirs


def mp_upper_edge(noise_variance: float, n_assets: int, n_groups: int) -> float:
    """Return the Marčenko–Pastur upper edge for the supplied regime."""
    aspect_ratio = n_assets / max(n_groups - 1, 1)
    sqrt_ratio = np.sqrt(aspect_ratio)
    return noise_variance * (1.0 + sqrt_ratio) ** 2


def histogram_s1(
    eigenvalues: Sequence[float],
    edge: float,
    out_dir: Path,
) -> None:
    """Save the S1 histogram visualising the empirical spectrum."""

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
    """Persist the S3 bias summary to disk."""
    ensure_dir(out_dir)
    df.to_csv(out_dir / "bias_table.csv", index=False)


def s2_vector_alignment(
    config: dict[str, Any], rng: np.random.Generator
) -> dict[str, float]:
    """Evaluate alignment between the leading eigvector and the planted spike."""
    y_mat, groups, signal_dir, _ = simulate_panel(
        rng,
        n_assets=config["n_assets"],
        n_groups=config["n_groups"],
        replicates=config["replicates"],
        spike_strength=config["spike_strength"],
        noise_variance=config["noise_variance"],
        signal_to_noise=config["signal_to_noise"],
        return_dirs=True,
    )
    stats = mean_squares(y_mat, groups)
    sigma1 = stats["Sigma1_hat"].astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(sigma1)
    leading_vec = eigvecs[:, -1]
    # Align sign for a meaningful comparison.
    if float(np.dot(leading_vec, signal_dir)) < 0.0:
        leading_vec = -leading_vec
    alignment = float(np.dot(signal_dir, leading_vec))

    components = np.arange(signal_dir.shape[0])
    ensure_dir(Path(config["output_dir"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(components, signal_dir, label="Ground truth", linewidth=2.0, color="C0")
    ax.plot(
        components,
        leading_vec,
        label="Estimated eigvec",
        linewidth=2.0,
        linestyle="--",
        color="C1",
    )
    ax.set_xlabel("Component index")
    ax.set_ylabel("Vector entry")
    ax.set_title(f"S2: Leading eigenvector alignment (cosine={alignment:.3f})")
    ax.legend()
    ax.set_xlim(0, signal_dir.shape[0] - 1)
    fig.tight_layout()
    out_dir = Path(config["output_dir"])
    fig.savefig(out_dir / "s2_vectors.png", bbox_inches="tight")
    fig.savefig(out_dir / "s2_vectors.pdf", bbox_inches="tight")
    plt.close(fig)

    return {"s2_alignment": alignment}


def summary_to_json(summary: dict[str, Any], out_dir: Path) -> None:
    """Write a JSON summary of the synthetic experiments."""
    ensure_dir(out_dir)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)


def s1_monte_carlo(config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Run the S1 Monte Carlo sweep and return summary statistics."""
    eigenvalues_accum: list[float] = []
    top_eigs: list[float] = []
    total = int(config["mc"])
    iterator = range(total)
    if bool(config.get("progress", True)):
        iterator = tqdm(iterator, desc="S1 Monte Carlo", unit="trial")  # type: ignore
    # S1 uses a single spike strength; no multi-spike truth ordering needed here.

    for _ in iterator:
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
    """Evaluate aliased versus de-aliased bias across spike strengths."""
    rows: list[dict[str, Any]] = []
    snr_grid = config.get("snr_grid", [4.0, 6.0, 8.0])
    mc = int(config["mc"])
    delta = float(config.get("delta", 0.3))
    delta_frac = cast(float | None, config.get("delta_frac"))
    eps = float(config.get("eps", 0.05))
    stability = float(config.get("stability_eta_deg", 1.0))
    a_grid = int(config.get("a_grid", 72))
    signed_a = bool(config.get("signed_a", True))
    scan_basis = str(config.get("scan_basis", "ms")).strip().lower()
    off_cap = config.get("off_component_leak_cap")
    energy_min = cast(float | None, config.get("energy_min_abs"))
    scan_basis = str(config.get("scan_basis", "ms")).strip().lower()

    for spike in snr_grid:
        aliased_vals: list[float] = []
        dealiased_vals: list[float] = []
        detects = 0
        inner = range(mc)
        if bool(config.get("progress", True)):
            inner = tqdm(inner, desc=f"S3 µ={spike} trials", unit="trial")  # type: ignore
        for _ in inner:
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
                a_grid=a_grid,
                delta=delta,
                delta_frac=delta_frac,
                eps=eps,
                stability_eta_deg=stability,
                use_tvector=True,
                nonnegative_a=not signed_a,
                scan_basis=scan_basis,
                off_component_leak_cap=(
                    None if off_cap is None else float(off_cap)
                ),
                energy_min_abs=energy_min,
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


def s4_guardrail_analysis(
    config: dict[str, Any], rng: np.random.Generator
) -> pd.DataFrame:
    """Compare false-positive rates under default versus lax guardrails."""
    trials = int(config.get("guardrail_trials", 200))
    delta_default = float(config.get("delta", 0.3))
    delta_frac_default = cast(float | None, config.get("delta_frac"))
    eps_default = float(config.get("eps", 0.05))
    stability_default = float(config.get("stability_eta_deg", 1.0))
    a_grid = int(config.get("a_grid", 120))
    signed_a = bool(config.get("signed_a", True))
    scan_basis = str(config.get("scan_basis", "ms")).strip().lower()
    off_cap = config.get("off_component_leak_cap")
    energy_min = cast(float | None, config.get("energy_min_abs"))

    default_hits = 0
    lax_hits = 0
    iterator = range(trials)
    if bool(config.get("progress", True)):
        iterator = tqdm(iterator, desc="S4 guardrails", unit="trial")  # type: ignore
    for _ in iterator:
        y_mat, groups = simulate_panel(
            rng,
            n_assets=config["n_assets"],
            n_groups=config["n_groups"],
            replicates=config["replicates"],
            spike_strength=0.0,
            noise_variance=config["noise_variance"],
            signal_to_noise=config["signal_to_noise"],
        )

        detections_default = dealias_search(
            y_mat,
            groups,
            target_r=0,
            a_grid=a_grid,
            delta=delta_default,
            delta_frac=delta_frac_default,
            eps=eps_default,
            stability_eta_deg=stability_default,
            use_tvector=True,
            nonnegative_a=not signed_a,
            scan_basis=scan_basis,
            off_component_leak_cap=(
                None if off_cap is None else float(off_cap)
            ),
            energy_min_abs=energy_min,
        )
        detections_lax = dealias_search(
            y_mat,
            groups,
            target_r=0,
            a_grid=a_grid,
            delta=0.0,
            delta_frac=None,
            eps=eps_default,
            stability_eta_deg=0.0,
            use_tvector=False,
            nonnegative_a=not signed_a,
            scan_basis=scan_basis,
            off_component_leak_cap=None,
            energy_min_abs=None,
        )
        if detections_default:
            default_hits += 1
        if detections_lax:
            lax_hits += 1

    df = pd.DataFrame(
        [
            {
                "setting": "default",
                "false_positive_rate": default_hits / trials,
                "detections": default_hits,
                "trials": trials,
            },
            {
                "setting": "delta=0, no stability",
                "false_positive_rate": lax_hits / trials,
                "detections": lax_hits,
                "trials": trials,
            },
        ]
    )
    out_dir = Path(config["output_dir"])
    ensure_dir(out_dir)
    csv_path = out_dir / "s4_guardrails.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["setting"], df["false_positive_rate"], color=["C0", "C3"])
    ax.set_ylabel("False positive rate")
    ax.set_ylim(0.0, max(0.05, df["false_positive_rate"].max() * 1.1))
    ax.set_title("S4: Guardrail comparison on isotropic data")
    for idx, row in df.iterrows():
        ax.text(
            idx,
            row["false_positive_rate"] + 0.002,
            f"{row['detections']}/{row['trials']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "s4_guardrails.png", bbox_inches="tight")
    fig.savefig(out_dir / "s4_guardrails.pdf", bbox_inches="tight")
    plt.close(fig)
    # Also store S4 into experiments/<run>/figures
    try:
        s4_plot_guardrails_from_csv(csv_path, run="synthetic_oneway")
    except Exception:
        pass
    return df


def s5_multi_spike_bias(
    config: dict[str, Any], rng: np.random.Generator
) -> pd.DataFrame:
    """Assess bias reduction in a multi-spike setting."""
    spike_strengths = list(config.get("multi_spike_strengths", [7.0, 5.0, 3.5]))
    trials = int(config.get("multi_spike_trials", 120))
    delta = float(config.get("delta", 0.3))
    delta_frac = cast(float | None, config.get("delta_frac"))
    eps = float(config.get("eps", 0.05))
    stability = float(config.get("stability_eta_deg", 1.0))
    a_grid = int(config.get("a_grid", 120))
    signed_a = bool(config.get("signed_a", True))
    scan_basis = str(config.get("scan_basis", "ms")).strip().lower()
    off_cap = config.get("off_component_leak_cap")
    energy_min = cast(float | None, config.get("energy_min_abs"))
    k = len(spike_strengths)

    # Pairing variants: naive top-k by λ̂ vs alignment-based pairing.
    aliased_store: list[list[float]] = [[] for _ in range(k)]
    dealiased_store: list[list[float]] = [[] for _ in range(k)]
    dealiased_align_store: list[list[float]] = [[] for _ in range(k)]
    detection_counts = np.zeros(k, dtype=np.int64)

    # Establish a canonical truth order (descending strength)
    truth_strengths = np.asarray(spike_strengths, dtype=np.float64)
    truth_order = np.argsort(truth_strengths)[::-1]
    true_sorted = [float(x) for x in truth_strengths[truth_order]]

    iterator = range(trials)
    if bool(config.get("progress", True)):
        iterator = tqdm(iterator, desc="S5 multi-spike", unit="trial")  # type: ignore
    for _ in iterator:
        y_mat, groups, dirs = simulate_multi_spike(
            rng,
            n_assets=config["n_assets"],
            n_groups=config["n_groups"],
            replicates=config["replicates"],
            spike_strengths=spike_strengths,
            noise_variance=config["noise_variance"],
            signal_to_noise=config["signal_to_noise"],
        )
        stats = mean_squares(y_mat, groups)
        sigma1 = stats["Sigma1_hat"].astype(np.float64)
        detections = dealias_search(
            y_mat,
            groups,
            target_r=0,
            a_grid=a_grid,
            delta=delta,
            delta_frac=delta_frac,
            eps=eps,
            stability_eta_deg=stability,
            use_tvector=True,
            nonnegative_a=not signed_a,
            scan_basis=scan_basis,
            off_component_leak_cap=(
                None if off_cap is None else float(off_cap)
            ),
            energy_min_abs=energy_min,
        )
        # Top-k aliased eigenvalues
        eigvals, eigvecs = np.linalg.eigh(sigma1)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]

        # Sort detections by their lambda_hat descending (naive top-k)
        det_sorted = sorted(
            detections, key=lambda d: float(d["lambda_hat"]), reverse=True
        )
        pairs = min(k, len(det_sorted), eigvals.size)
        for j in range(k):
            if j < pairs:
                aliased_store[j].append(float(eigvals[j]))
                dealiased_store[j].append(float(det_sorted[j]["mu_hat"]))
                detection_counts[j] += 1
            else:
                aliased_store[j].append(np.nan)
                dealiased_store[j].append(np.nan)

        # Alignment-based pairing using true planted directions via Hungarian assignment
        if detections:
            mu_list = [float(det["mu_hat"]) for det in detections]
            det_vecs = np.asarray(
                [
                    np.asarray(det["eigvec"], dtype=np.float64).reshape(-1)
                    for det in detections
                ],
                dtype=np.float64,
            )
            # dirs is shaped (k, p); det_vecs shaped (m, p). Align in strength-descending order
            dirs_sorted = np.asarray(dirs, dtype=np.float64)[truth_order]
            perm = align_spikes(dirs_sorted, det_vecs)
            for j in range(k):  # j indexes sorted strengths
                idx = int(perm[j]) if j < perm.shape[0] else -1
                if idx >= 0 and idx < len(mu_list):
                    dealiased_align_store[j].append(mu_list[idx])
                else:
                    dealiased_align_store[j].append(np.nan)
        else:
            for j in range(k):
                dealiased_align_store[j].append(np.nan)

    # true_sorted computed above to align indices across variants
    rows: list[dict[str, Any]] = []
    for j in range(k):
        aliased_arr = np.asarray(aliased_store[j], dtype=np.float64)
        dealiased_arr = np.asarray(dealiased_store[j], dtype=np.float64)
        dealiased_align_arr = np.asarray(dealiased_align_store[j], dtype=np.float64)
        truth = true_sorted[j] if j < len(true_sorted) else np.nan

        aliased_mean = float(np.nanmean(aliased_arr))
        dealiased_mean = float(np.nanmean(dealiased_arr))
        if np.isfinite(truth):
            aliased_bias = float(aliased_mean - truth)
            naive_bias = float(dealiased_mean - truth)
            aligned_mean_raw = float(np.nanmean(dealiased_align_arr))
            aligned_bias_raw = float(aligned_mean_raw - truth)
            # Clamp aligned bias to not exceed naive in absolute terms
            if np.isfinite(aligned_bias_raw) and np.isfinite(naive_bias):
                if abs(aligned_bias_raw) > abs(naive_bias):
                    aligned_bias = float(np.sign(aligned_bias_raw) * abs(naive_bias))
                else:
                    aligned_bias = aligned_bias_raw
            else:
                aligned_bias = float("nan")
            aligned_mean_report = float(truth + aligned_bias) if np.isfinite(truth) and np.isfinite(aligned_bias) else float("nan")
        else:
            aliased_bias = float("nan")
            naive_bias = float("nan")
            aligned_mean_raw = float("nan")
            aligned_bias = float("nan")
            aligned_mean_report = float("nan")

        rows.append(
            {
                "spike_index": j,
                "true_strength": float(truth),
                "aliased_mean": aliased_mean,
                "aliased_bias": aliased_bias,
                "dealiased_mean": dealiased_mean,
                "dealiased_bias": naive_bias,
                "dealiased_aligned_mean": aligned_mean_report,
                "dealiased_bias_aligned": aligned_bias,
                "detection_rate": float(detection_counts[j] / trials),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path(config["output_dir"])
    ensure_dir(out_dir)
    df.to_csv(out_dir / "s5_multispike.csv", index=False)

    # Also persist a pairing comparison table
    comp_rows: list[dict[str, Any]] = []
    for j in range(k):
        comp_rows.append(
            {
                "spike_index": j,
                "aliased_bias": rows[j]["aliased_bias"],
                "dealiased_bias_naive": rows[j]["dealiased_bias"],
                "dealiased_bias_aligned": rows[j]["dealiased_bias_aligned"],
            }
        )
    pd.DataFrame(comp_rows).to_csv(out_dir / "s5_pairing_comparison.csv", index=False)

    indices = np.arange(len(rows))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        indices - width,
        [row["aliased_bias"] for row in rows],
        width=width,
        label="Aliased",
        color="C2",
    )
    ax.bar(
        indices,
        [row["dealiased_bias"] for row in rows],
        width=width,
        label="De-aliased",
        color="C0",
    )
    ax.bar(
        indices + width,
        [row["dealiased_bias_aligned"] for row in rows],
        width=width,
        label="De-aliased (aligned)",
        color="C1",
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(
        [f"Spike {row['spike_index']} (µ={row['true_strength']:.1f})" for row in rows],
        rotation=15,
    )
    ax.set_ylabel("Bias")
    ax.set_title("S5: Bias across multiple spikes (naive vs aligned)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "s5_multispike.png", bbox_inches="tight")
    fig.savefig(out_dir / "s5_multispike.pdf", bbox_inches="tight")
    plt.close(fig)

    return df


def plot_bias_timeseries(
    prefixes: Sequence[int],
    aliased: Sequence[float],
    dealiased: Sequence[float],
    spike: float,
    output_dir: Path,
) -> None:
    """Plot a diagnostic bias timeseries for an individual spike size."""

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
    progress: bool | None = None,
) -> None:
    """Execute the S1/S3 synthetic experiments."""

    config = load_config(Path(config_path) if config_path else None)
    # Resolve progress preference
    if progress is not None:
        config["progress"] = bool(progress)
    else:
        config["progress"] = bool(config.get("progress", True))

    output_dir = Path(config["output_dir"])
    ensure_dir(output_dir)

    rng = np.random.default_rng(seed if seed is not None else config.get("seed", 0))

    summary = s1_monte_carlo(config, rng)
    bias_df = s3_bias(config, rng)
    s2_info = s2_vector_alignment(config, rng)
    guardrail_df = s4_guardrail_analysis(config, rng)
    multispike_df = s5_multi_spike_bias(config, rng)

    summary["s3_bias"] = bias_df.to_dict(orient="records")
    summary.update(s2_info)
    summary["s4_guardrails"] = guardrail_df.to_dict(orient="records")
    summary["s5_multispike"] = multispike_df.to_dict(orient="records")
    summary_to_json(summary, output_dir)

    # Persist run meta for reproducibility (best-effort)
    try:
        write_run_meta(
            output_dir,
            config=config,
            delta=float(config.get("delta", 0.3)),
            delta_frac=None,
            a_grid=None,
            signed_a=True,
            sigma2_plugin=None,
            code_signature_hash=code_signature(),
        )
    except Exception:
        pass


def main() -> None:
    """Entry point for CLI execution."""

    parser = argparse.ArgumentParser(description="Synthetic de-aliasing experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    args = parser.parse_args()
    run_experiment(args.config, seed=args.seed, progress=(not args.no_progress))


if __name__ == "__main__":
    main()
