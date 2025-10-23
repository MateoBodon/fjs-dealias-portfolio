from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

# Use non-interactive backend suitable for CI/tests
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Reuse existing core plotters where they match requirements
from fjs.spectra import plot_spectrum_with_edges, plot_spike_timeseries
from evaluation.evaluate import plot_coverage_error


def _figures_dir_for_run(run: str | Path) -> Path:
    """Return figures directory for a given run under experiments/<run>/figures."""

    p = Path("experiments") / str(run) / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def e1_plot_spectrum_with_mp(
    eigenvalues: Sequence[float],
    mp_edges: Iterable[float] | None,
    *,
    run: str | Path,
    title: str | None = None,
) -> Path:
    """E1: Plot spectrum with MP edge and mark outliers.

    Saves PDF to experiments/<run>/figures/E1_spectrum.pdf
    """

    out_dir = _figures_dir_for_run(run)
    base = out_dir / "E1_spectrum"
    # Mark as outliers any eigenvalues above MP upper edge (if provided)
    highlight = (max(list(mp_edges)) if mp_edges is not None else None)
    # Save PDF (and a PNG for convenience)
    plot_spectrum_with_edges(
        eigenvalues,
        edges=mp_edges,
        out_path=base.with_suffix(".pdf"),
        title=title,
        highlight_threshold=highlight,
    )
    # Also emit PNG alongside PDF (no extra style)
    plot_spectrum_with_edges(
        eigenvalues,
        edges=mp_edges,
        out_path=base.with_suffix(".png"),
        title=title,
        highlight_threshold=highlight,
    )
    return base.with_suffix(".pdf")


def e2_plot_spike_timeseries(
    time_index: Sequence[float],
    aliased_series: Sequence[float],
    dealiased_series: Sequence[float],
    *,
    run: str | Path,
    title: str | None = None,
    xlabel: str = "Window",
    ylabel: str = "Spike magnitude",
) -> Path:
    """E2: Plot aliased vs de-aliased spike time-series.

    Saves PDF to experiments/<run>/figures/E2_spike_timeseries.pdf
    """

    out_dir = _figures_dir_for_run(run)
    base = out_dir / "E2_spike_timeseries"
    # Single chart per figure; reuse helper to ensure consistent look
    plot_spike_timeseries(
        time_index,
        aliased_series,
        dealiased_series,
        out_path=base.with_suffix(".pdf"),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    plot_spike_timeseries(
        time_index,
        aliased_series,
        dealiased_series,
        out_path=base.with_suffix(".png"),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    return base.with_suffix(".pdf")


def e3_plot_var_mse(
    errors_by_method: Mapping[str, Iterable[float]],
    *,
    run: str | Path,
    title: str = "E3: Variance forecast MSE (mean)",
    ylabel: str = "Squared error",
) -> Path:
    """E3: Single-chart Var-MSE comparison across methods (bar of means).

    Saves PDF to experiments/<run>/figures/E3_variance_mse.pdf
    """

    out_dir = _figures_dir_for_run(run)
    base = out_dir / "E3_variance_mse"

    # Prepare data
    methods: list[str] = []
    means: list[float] = []
    for k, vals in errors_by_method.items():
        arr = np.asarray(list(vals), dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        methods.append(k)
        means.append(float(np.mean(arr)))
    if not methods:
        # Create an empty figure to satisfy CI/tests expecting a file
        fig, _ = plt.subplots(figsize=(7, 4))
        fig.suptitle(title)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
        return base.with_suffix(".pdf")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(methods, means, color=[f"C{i}" for i in range(len(methods))])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".pdf")


def e4_plot_var_coverage(
    coverage_errors: Mapping[str, float],
    *,
    run: str | Path,
) -> Path:
    """E4: VaR(95%) coverage error plot.

    Saves PDF to experiments/<run>/figures/E4_var95_coverage_error.pdf
    """

    out_dir = _figures_dir_for_run(run)
    base = out_dir / "E4_var95_coverage_error"
    # Reuse the existing single-chart plotter and direct it to our figures dir
    plot_coverage_error(coverage_errors, base)
    return base.with_suffix(".pdf")


def s4_plot_guardrails_from_csv(
    csv_path: str | Path,
    *,
    run: str | Path,
    title: str = "S4: Guardrails on isotropic data",
) -> Path:
    """S4: Plot guardrail false-positive comparison from a CSV.

    The CSV is expected to have columns: setting, false_positive_rate, detections, trials.
    Saves PDF to experiments/<run>/figures/S4_guardrails.pdf
    """

    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"setting", "false_positive_rate"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: setting, false_positive_rate")

    out_dir = _figures_dir_for_run(run)
    base = out_dir / "S4_guardrails"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["setting"], df["false_positive_rate"], color=["C0", "C3"])
    ax.set_ylabel("False positive rate")
    ax.set_title(title)
    for tick in ax.get_xticklabels():
        tick.set_rotation(10)
        tick.set_horizontalalignment("right")

    # Annotate with counts if present
    if {"detections", "trials"}.issubset(set(df.columns)):
        for idx, row in df.iterrows():
            try:
                txt = f"{int(row['detections'])}/{int(row['trials'])}"
                ax.text(idx, float(row["false_positive_rate"]) + 1e-3, txt, ha="center", va="bottom", fontsize=9)
            except Exception:
                pass

    fig.tight_layout()
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".pdf")

