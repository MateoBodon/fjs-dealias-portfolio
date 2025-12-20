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


def _format_reason_summary(df: pd.DataFrame, top_k: int | None) -> list[str]:
    series = _reason_series(df)
    if series.empty:
        return []
    counts = series.value_counts()
    total = float(series.shape[0])
    lines = ["## Primary Skip Reasons", "", "| reason | count | share |", "| --- | --- | --- |"]
    items = counts.items() if top_k is None else counts.head(top_k).items()
    for reason, count in items:
        share = count / total if total else 0.0
        lines.append(f"| {reason} | {count} | {share:.2%} |")
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


def _format_reason_examples(df: pd.DataFrame, top_k: int, example_k: int = 5) -> list[str]:
    reason_col = "skip_reason_primary" if "skip_reason_primary" in df.columns else "skip_reason"
    if reason_col not in df.columns:
        return []
    reason_values = df[reason_col].fillna("").astype(str)
    series = reason_values[reason_values != ""]
    if series.empty:
        return []
    counts = series.value_counts()
    guard_cols = [col for col in df.columns if col.startswith("guard_")]
    lines: list[str] = ["## Example Windows by Reason"]

    def _fmt_float(val: float, precision: int = 4) -> str:
        if pd.isna(val):
            return "nan"
        return f"{float(val):.{precision}f}"

    for reason, count in counts.head(top_k).items():
        mask = reason_values == reason
        subset = df[mask].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            by=["accepted", "raw_detections", "candidate_pool", "window_index"],
            ascending=[True, False, False, True],
        ).head(example_k)
        share = count / float(series.shape[0]) if series.shape[0] else 0.0
        lines.append(f"### {reason} ({count} | {share:.2%})")
        for _, row in subset.iterrows():
            window_id = int(row.get("window_index", -1))
            fit_start = row.get("fit_start", "")
            fit_end = row.get("fit_end", "")
            hold_start = row.get("hold_start", "")
            hold_end = row.get("hold_end", "")
            label = str(row.get("label", "") or "")
            design = str(row.get("design", "") or "")
            estimator = str(row.get("estimator", "") or "")
            p_val = row.get("p", pd.NA)
            t_val = row.get("t", pd.NA)
            reps_val = row.get("replicates", pd.NA)
            delta_used = row.get("delta_frac_used", float("nan"))
            edge_mode_val = str(row.get("edge_mode", ""))
            edge_used_val = row.get("edge_used", float("nan"))
            lambda_ratio = row.get("lambda_top_over_edge", float("nan"))
            guard_fields = []
            for col in guard_cols:
                try:
                    guard_val = int(row.get(col, 0))
                except Exception:
                    guard_val = 0
                if guard_val:
                    guard_fields.append(f"{col.replace('guard_', '')}={guard_val}")
            guard_fields = guard_fields[:3]
            detail = str(row.get("skip_reason_detail", "")).strip()
            parts = [
                f"w={window_id}",
                f"fit={fit_start}→{fit_end}",
                f"hold={hold_start}→{hold_end}" if hold_start or hold_end else "",
                f"label={label}" if label else "",
                f"{design}/{estimator}" if design or estimator else "",
                f"p={int(p_val) if pd.notna(p_val) else 'na'}",
                f"T={int(t_val) if pd.notna(t_val) else 'na'}",
                f"reps={int(reps_val) if pd.notna(reps_val) else 'na'}",
                f"delta={_fmt_float(delta_used)}",
                f"edge={edge_mode_val}@{_fmt_float(edge_used_val, precision=4)}",
                f"λ/edge={_fmt_float(lambda_ratio, precision=3)}",
                f"guards={'; '.join(guard_fields)}" if guard_fields else "",
                f"accepted={bool(row.get('accepted', False))}",
            ]
            stage = str(row.get("exception_stage", "")).strip()
            exc_short = str(row.get("exception_message_short", "")).strip()
            exc_type = str(row.get("exception_type", "")).strip()
            if detail:
                parts.append(f"detail={detail}")
            if stage or exc_type or exc_short:
                context = " / ".join([val for val in [stage, exc_type, exc_short] if val])
                if context:
                    parts.append(f"exception={context}")
            lines.append(f"- {', '.join([part for part in parts if part])}")
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

    lines.extend(_format_reason_summary(df, top_k=None))

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
