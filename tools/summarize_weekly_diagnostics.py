"""Summarize weekly gating diagnostics into a markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

def _reason_series(df: pd.DataFrame) -> pd.Series:
    col = "skip_reason_primary" if "skip_reason_primary" in df.columns else "skip_reason"
    if col not in df.columns:
        return pd.Series(dtype=str)
    series = df[col].fillna("").astype(str)
    return series[series != ""]


def _format_reason_summary(df: pd.DataFrame, top_k: int) -> list[str]:
    series = _reason_series(df)
    if series.empty:
        return []
    counts = series.value_counts()
    total = float(series.shape[0])
    lines = ["## Primary Skip Reasons"]
    for reason, count in counts.head(top_k).items():
        share = count / total if total else 0.0
        lines.append(f"- {reason}: {count} ({share:.2%})")
    lines.append("")
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


def _format_reason_examples(df: pd.DataFrame, top_k: int, example_k: int = 3) -> list[str]:
    reason_col = "skip_reason_primary" if "skip_reason_primary" in df.columns else "skip_reason"
    if reason_col not in df.columns:
        return []
    reason_values = df[reason_col].fillna("").astype(str)
    series = reason_values[reason_values != ""]
    if series.empty:
        return []
    counts = series.value_counts()
    lines: list[str] = ["## Example Windows by Reason"]
    example_columns = {
        "raw_detections": "raw",
        "candidate_pool": "cand",
        "isolated_spikes": "isolated",
        "gating_q_max": "q",
        "delta_frac_used": "delta",
        "edge_mode": "edge",
        "lambda_top_over_edge": "lambda_over_edge",
    }

    def _fmt_float(val: float, precision: int = 4) -> str:
        if pd.isna(val):
            return "nan"
        return f"{float(val):.{precision}f}"

    for reason, _ in counts.head(top_k).items():
        mask = reason_values == reason
        subset = df[mask].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            by=["accepted", "raw_detections", "candidate_pool", "window_index"],
            ascending=[True, False, False, True],
        ).head(example_k)
        lines.append(f"### {reason}")
        for _, row in subset.iterrows():
            fields = [f"w={int(row.get('window_index', -1))}"]
            fields.append(f"accepted={bool(row.get('accepted', False))}")
            for col, label in example_columns.items():
                if col not in row:
                    continue
                value = row[col]
                if col == "delta_frac_used":
                    fields.append(f"{label}={_fmt_float(value)}")
                elif col == "lambda_top_over_edge":
                    fields.append(f"{label}={_fmt_float(value, precision=3)}")
                else:
                    try:
                        fields.append(f"{label}={int(value)}")
                    except Exception:
                        pass
            detail = str(row.get("skip_reason_detail", "")).strip()
            if detail:
                fields.append(f"detail={detail}")
            lines.append(f"- {', '.join(fields)}")
        lines.append("")
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

    lines.extend(_format_reason_summary(df, top_k))

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

    example_lines = _format_reason_examples(df, top_k=top_k)
    if example_lines:
        lines.extend(example_lines)

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
