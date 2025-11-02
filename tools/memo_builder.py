"""Generate a compact RC memo from evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_memo(
    *,
    metrics_path: Path,
    dm_path: Path,
    var_path: Path,
    output_path: Path,
    regime: str,
) -> None:
    metrics = pd.read_csv(metrics_path)
    dm = pd.read_csv(dm_path) if dm_path.exists() else pd.DataFrame()
    var = pd.read_csv(var_path) if var_path.exists() else pd.DataFrame()

    lines: list[str] = [f"# RC Memo — {regime}", ""]
    if not metrics.empty:
        lines.append("## ΔMSE Summary")
        pivot = metrics.pivot(index="estimator", columns="portfolio", values="delta_mse_vs_rie")
        lines.append(_to_table(pivot))
        lines.append("")
        best = metrics.sort_values("delta_mse_vs_rie").groupby("portfolio").first().reset_index()
        for _, row in best.iterrows():
            lines.append(
                f"- **{row['portfolio'].upper()}** best ΔMSE: {row['estimator']} ({row['delta_mse_vs_rie']:.4e})"
            )
        lines.append("")

    if not dm.empty:
        lines.append("## Diebold–Mariano Tests")
        lines.append(_to_table(dm, index=False))
        lines.append("")

    if not var.empty:
        lines.append("## VaR Coverage")
        var_table = var.pivot(index="estimator", columns="portfolio", values="violation_rate")
        lines.append(_to_table(var_table))
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _to_table(frame: pd.DataFrame, *, index: bool = True) -> str:
    try:
        return frame.to_markdown(index=index)
    except ImportError:  # pandas tabulate optional
        return frame.to_string(index=index)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RC memo from evaluation artifacts.")
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics CSV.")
    parser.add_argument("--dm", type=Path, required=True, help="Path to DM CSV.")
    parser.add_argument("--var", type=Path, required=True, help="Path to VaR CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Output memo markdown path.")
    parser.add_argument("--regime", type=str, default="full", help="Regime label.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_memo(
        metrics_path=args.metrics,
        dm_path=args.dm,
        var_path=args.var,
        output_path=args.out,
        regime=args.regime,
    )


if __name__ == "__main__":
    main()
