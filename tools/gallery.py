"""Render simple figures for RC reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_gallery(
    *,
    metrics_path: Path,
    var_path: Path,
    output_dir: Path,
    regime: str,
) -> None:
    metrics = pd.read_csv(metrics_path)
    var = pd.read_csv(var_path) if var_path.exists() else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - fallback when matplotlib missing
        summary = metrics.pivot(index="estimator", columns="portfolio", values="delta_mse_vs_rie")
        (output_dir / f"delta_mse_{regime}.txt").write_text(summary.to_string(), encoding="utf-8")
        if not var.empty:
            coverage = var.pivot(index="estimator", columns="portfolio", values="violation_rate")
            (output_dir / f"var_coverage_{regime}.txt").write_text(coverage.to_string(), encoding="utf-8")
        return

    # ΔMSE bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics.pivot(index="estimator", columns="portfolio", values="delta_mse_vs_rie").plot.bar(ax=ax, rot=0)
    ax.set_ylabel("ΔMSE vs RIE")
    ax.set_title(f"ΔMSE — {regime}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"delta_mse_{regime}.png")
    plt.close(fig)

    if not var.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        var.pivot(index="estimator", columns="portfolio", values="violation_rate").plot.bar(ax=ax, rot=0)
        ax.set_ylabel("Violation rate")
        ax.set_title(f"VaR Violations — {regime}")
        ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="Target 5%")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"var_violations_{regime}.png")
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gallery figures from evaluation artifacts.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics CSV.")
    parser.add_argument("--var", type=Path, required=True, help="Path to VaR CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for figures.")
    parser.add_argument("--regime", type=str, default="full", help="Regime label.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_gallery(
        metrics_path=args.metrics,
        var_path=args.var,
        output_dir=args.out,
        regime=args.regime,
    )


if __name__ == "__main__":
    main()
