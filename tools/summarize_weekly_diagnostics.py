"""Summarize weekly gating diagnostics into a markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _format_skip_summary(df: pd.DataFrame, top_k: int) -> list[str]:
    if "skip_reason" not in df.columns:
        return []
    series = df["skip_reason"].fillna("")
    series = series[series != ""]
    if series.empty:
        return []
    counts = series.value_counts()
    total = float(series.shape[0])
    lines: list[str] = []
    for reason, count in counts.head(top_k).items():
        share = count / total if total else 0.0
        lines.append(f"- {reason}: {count} ({share:.2%})")
    return lines


def _render_stat_table(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    lines = ["| metric | min | median | max |", "| --- | --- | --- | --- |"]
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        lines.append(
            f"| {col} | {series.min():.6g} | {series.median():.6g} | {series.max():.6g} |"
        )
    return lines


def _guardrail_totals(df: pd.DataFrame) -> list[str]:
    guard_cols = [col for col in df.columns if col.startswith("guard_")]
    lines: list[str] = []
    for col in guard_cols:
        total = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
        if total:
            lines.append(f"- {col.replace('guard_', '')}: {total}")
    return lines


def summarize(input_path: Path, output_path: Path, top_k: int = 5) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {input_path}")
    df = pd.read_csv(input_path)
    total_windows = int(df.shape[0])
    detection_rate = float(df.get("accepted", pd.Series(dtype=float)).astype(bool).mean()) if total_windows else 0.0

    lines = ["# Weekly Gating Diagnostics", ""]
    lines.append(f"- Input: {input_path}")
    lines.append(f"- Windows: {total_windows}")
    lines.append(f"- Detection rate: {detection_rate:.2%}")

    skip_lines = _format_skip_summary(df, top_k)
    lines.append("- Top skip reasons: " + ("none" if not skip_lines else ""))
    lines.extend(skip_lines)
    lines.append("")

    stat_lines = _render_stat_table(
        df,
        [
            "delta_frac_used",
            "lambda_top_over_edge",
            "edge_used",
            "candidate_pool",
            "raw_detections",
            "isolated_spikes",
        ],
    )
    if stat_lines:
        lines.append("## Gate Stats (min/median/max)")
        lines.extend(stat_lines)
        lines.append("")

    guard_lines = _guardrail_totals(df)
    if guard_lines:
        lines.append("## Guardrail Triggers")
        lines.extend(guard_lines)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gating_diagnostics.csv")
    parser.add_argument("--input", type=Path, required=True, help="Path to gating_diagnostics.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write summary markdown (default: alongside input as weekly_diagnostics.md)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top skip reasons to list")
    args = parser.parse_args()

    output_path = args.output or args.input.with_name("weekly_diagnostics.md")
    summarize(args.input, output_path, top_k=int(args.top_k))


if __name__ == "__main__":
    main()
