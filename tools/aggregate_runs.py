#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import glob

import pandas as pd


def _resolve_runs(patterns: Sequence[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            path = Path(match)
            if path.is_dir():
                run_dirs.append(path.resolve())
    # Deduplicate while preserving lexical order
    return sorted({run for run in run_dirs})


def _load_run_metadata(run_dir: Path) -> tuple[str, str, str, str]:
    crisis_label = ""
    design = ""
    edge_mode = ""
    gating_mode = ""
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            crisis_label = str(payload.get("crisis_label", ""))
            design = str(payload.get("design", ""))
            edge_mode = str(payload.get("edge_mode", ""))
            gating = payload.get("gating", {})
            if isinstance(gating, dict):
                gating_mode = str(gating.get("mode", ""))
            gating_mode = str(payload.get("gating_mode", gating_mode))
    return crisis_label, design, edge_mode, gating_mode


def aggregate_runs(run_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics_summary.csv"
        if not metrics_path.exists():
            continue
        try:
            df = pd.read_csv(metrics_path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        crisis_label, design, edge_mode, gating_mode = _load_run_metadata(run_dir)
        df.insert(0, "run", run_dir.name)
        df.insert(1, "run_path", str(run_dir))
        df.insert(2, "crisis_label", crisis_label)
        df.insert(3, "design", design)
        if "edge_mode" not in df.columns:
            df.insert(4, "edge_mode", edge_mode)
        else:
            df["edge_mode"] = df["edge_mode"].fillna(edge_mode)
        if "gating_mode" not in df.columns:
            df.insert(5, "gating_mode", gating_mode)
        else:
            df["gating_mode"] = df["gating_mode"].fillna(gating_mode)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics_summary.csv across multiple runs.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Glob pattern(s) for run directories (e.g., experiments/equity_panel/outputs_*).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/aggregate_summary.csv"),
        help="Path to write the aggregated CSV (default: reports/aggregate_summary.csv).",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=None,
        help="Optional path for a LaTeX table representation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _resolve_runs(args.inputs)
    if not run_dirs:
        raise ValueError("No run directories matched the provided patterns.")

    df = aggregate_runs(run_dirs)
    if df.empty:
        raise ValueError("No metrics_summary.csv files found in the matched directories.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if args.tex_out is not None:
        args.tex_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_latex(args.tex_out, index=False, float_format=lambda x: f"{float(x):.4g}")

    print(f"Aggregated {len(run_dirs)} run(s) into {args.out}")
    if args.tex_out is not None:
        print(f"LaTeX table written to {args.tex_out}")


if __name__ == "__main__":
    main()
