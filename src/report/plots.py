from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_dm_bars",
    "plot_edge_margin_hist",
    "plot_detection_rate",
    "plot_ablation_heatmap",
    "plot_alignment_angles",
]

DEFAULT_FIG_ROOT = Path("figures")
EW_LABEL = "Equal Weight"
MV_PREFIX = "Min-Variance"


def _single_run_tag(df: pd.DataFrame) -> str:
    if "run" not in df or df["run"].nunique() != 1:
        raise ValueError("Expected a single run in the provided DataFrame.")
    return df["run"].iloc[0]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_dm_bars(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> Path:
    if df.empty:
        raise ValueError("DM p-value DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "plots"
    _ensure_dir(output_dir)

    estimators = sorted(df["estimator"].unique())
    ew_values = []
    mv_values = []
    for estimator in estimators:
        group = df[df["estimator"] == estimator]
        ew_series = group[group["strategy"] == EW_LABEL]["dm_p"]
        if ew_series.empty:
            ew_series = group[group["strategy"].str.startswith(EW_LABEL, na=False)]["dm_p"]
        mv_series = group[group["strategy"].str.startswith(MV_PREFIX, na=False)]["dm_p"]
        ew_values.append(float(ew_series.iloc[0]) if not ew_series.empty else np.nan)
        mv_values.append(float(mv_series.iloc[0]) if not mv_series.empty else np.nan)

    x = np.arange(len(estimators))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, ew_values, width, label="EW")
    ax.bar(x + width / 2, mv_values, width, label="MV")
    ax.set_xticks(x)
    ax.set_xticklabels(estimators, rotation=20, ha="right")
    ax.set_ylabel("DM p-value")
    ax.set_title("Diebold–Mariano p-values by estimator")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "dm_pvals.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_edge_margin_hist(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> Path:
    if df.empty:
        raise ValueError("Edge margin DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "plots"
    _ensure_dir(output_dir)

    column = "edge_margin"
    if column not in df.columns:
        column = "top_edge_margin"
    if column not in df.columns:
        raise ValueError("Expected 'edge_margin' or 'top_edge_margin' column.")

    values = df[column].dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(len(values), 20) or 10, color="C0", edgecolor="black")
    ax.set_xlabel("Edge margin")
    ax.set_ylabel("Count")
    ax.set_title("Edge margin distribution")
    fig.tight_layout()

    path = output_dir / "edge_margin_hist.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_detection_rate(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> Path:
    if df.empty:
        raise ValueError("Detection rate DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "plots"
    _ensure_dir(output_dir)

    grouped = df.groupby("estimator")["detection_rate"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(grouped.index))
    ax.bar(positions, grouped.values, color="C1")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(grouped.index, rotation=20, ha="right")
    ax.set_title("Detection rate by estimator")
    fig.tight_layout()

    path = output_dir / "detection_rate.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_alignment_angles(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> Path:
    if df.empty:
        raise ValueError("Alignment DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "plots"
    _ensure_dir(output_dir)

    column = "angle_min_deg"
    if column not in df.columns:
        raise ValueError("Expected 'angle_min_deg' column in DataFrame.")

    values = df[column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("No finite alignment angles available for plotting.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(len(values), 18) or 10, color="C2", edgecolor="black")
    ax.set_xlabel("Minimum principal angle (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Alignment of accepted detections vs PCA subspace")
    ax.set_xlim(0, 90)
    fig.tight_layout()

    path = output_dir / "alignment_angles.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ablation_heatmap(df: pd.DataFrame, *, root: Path = DEFAULT_FIG_ROOT) -> Path:
    if df.empty:
        raise ValueError("Ablation DataFrame is empty.")

    run_tag = _single_run_tag(df)
    output_dir = root / run_tag / "plots"
    _ensure_dir(output_dir)

    df = df.copy()
    if "mse_gain" not in df.columns and {"mse_alias", "mse_de"}.issubset(df.columns):
        df["mse_gain"] = df["mse_alias"] - df["mse_de"]
    if "delta_mse_mv" not in df.columns and {"mse_mv", "mse_de"}.issubset(df.columns):
        df["delta_mse_mv"] = df["mse_mv"] - df["mse_de"]

    required = {"delta_frac", "eps", "mse_gain"}
    if not required.issubset(df.columns):
        raise ValueError("Ablation DataFrame must include 'delta_frac', 'eps', and 'mse_gain' columns.")

    agg_kwargs = {"aggfunc": lambda series: float(np.nanmean(series)) if len(series) else float("nan")}
    pivot_gain = df.pivot_table(index="delta_frac", columns="eps", values="mse_gain", **agg_kwargs)
    pivot_cov = (
        df.pivot_table(index="delta_frac", columns="eps", values="detection_rate", **agg_kwargs)
        if "detection_rate" in df.columns
        else None
    )
    pivot_mv = (
        df.pivot_table(index="delta_frac", columns="eps", values="delta_mse_mv", **agg_kwargs)
        if "delta_mse_mv" in df.columns
        else None
    )

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pivot_gain.values, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot_gain.columns)))
    ax.set_xticklabels(pivot_gain.columns)
    ax.set_yticks(range(len(pivot_gain.index)))
    ax.set_yticklabels(pivot_gain.index)
    ax.set_xlabel("eps")
    ax.set_ylabel("delta_frac")
    ax.set_title("Ablation ΔMSE (alias - de) with coverage")

    for i, delta_val in enumerate(pivot_gain.index):
        for j, eps_val in enumerate(pivot_gain.columns):
            gain = pivot_gain.iloc[i, j]
            mv_val = pivot_mv.iloc[i, j] if pivot_mv is not None and (delta_val in pivot_mv.index and eps_val in pivot_mv.columns) else float("nan")
            cov_val = pivot_cov.iloc[i, j] if pivot_cov is not None and (delta_val in pivot_cov.index and eps_val in pivot_cov.columns) else float("nan")
            parts: list[str] = []
            if not np.isnan(gain):
                parts.append(f"ΔEW {gain:.2e}")
            else:
                parts.append("ΔEW n/a")
            if not np.isnan(mv_val):
                parts.append(f"ΔMV {mv_val:.2e}")
            else:
                parts.append("ΔMV n/a")
            if not np.isnan(cov_val):
                parts.append(f"Cov {cov_val * 100:4.0f}%")
            else:
                parts.append("Cov n/a")
            ax.text(
                j,
                i,
                "\n".join(parts),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ΔMSE alias - de")
    fig.tight_layout()

    path = output_dir / "ablation_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
