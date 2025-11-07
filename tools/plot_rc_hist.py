#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_series(path: Path, metric: str, regime: str | None) -> pd.Series:
    frame = pd.read_csv(path)
    series = pd.to_numeric(frame[metric], errors="coerce")
    if regime and "regime" in frame.columns:
        mask = frame["regime"].astype(str).str.lower().eq(regime.lower())
        series = series[mask]
    return series.dropna()


def plot_histogram(
    diagnostics_path: Path,
    metric: str,
    out_path: Path,
    *,
    title: str | None = None,
    bins: int = 40,
    regime: str | None = None,
) -> None:
    series = _load_series(diagnostics_path, metric, regime)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    if series.empty:
        plt.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12)
        plt.axis("off")
    else:
        plt.hist(series, bins=bins, color="#1f77b4", alpha=0.75, edgecolor="black")
        plt.grid(True, axis="y", alpha=0.3, linestyle="--")
        plt.xlabel(metric.replace("_", " "))
        plt.ylabel("count")
    label = title or f"{metric} histogram"
    if regime:
        label = f"{label} ({regime})"
    plt.title(label)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot histograms from RC diagnostics.")
    parser.add_argument("--diagnostics", type=Path, required=True, help="Path to diagnostics_detail.csv.")
    parser.add_argument("--metric", type=str, required=True, help="Metric column to visualise.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title.")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins (default: 40).")
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Optional regime filter (full/calm/crisis).",
    )
    args = parser.parse_args()
    plot_histogram(
        diagnostics_path=args.diagnostics,
        metric=args.metric,
        out_path=args.out,
        title=args.title,
        bins=max(int(args.bins), 1),
        regime=args.regime,
    )


if __name__ == "__main__":
    main()
