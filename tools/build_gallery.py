#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from report.gather import collect_estimator_panel, find_runs, load_run
from report.plots import (
    plot_ablation_heatmap,
    plot_detection_rate,
    plot_dm_pvals,
    plot_edge_margin_hist,
)
from report.tables import table_ablation, table_estimators, table_rejections


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _discover_run_paths(entries: Iterable[dict]) -> list[Path]:
    run_paths: list[Path] = []
    for entry in entries:
        if "path" in entry:
            candidate = Path(entry["path"]).resolve()
            if candidate.is_dir():
                run_paths.append(candidate)
            continue

        root = entry.get("root")
        if not root:
            continue
        pattern = entry.get("pattern")
        root_paths = find_runs(root, pattern=pattern)
        run_paths.extend(root_paths)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in run_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return unique_paths


def _gather_rejections(summary_df: pd.DataFrame, run_tag: str) -> pd.DataFrame:
    records = []
    for column in summary_df.columns:
        if column.startswith("rejection_stats."):
            reason = column.split(".", 1)[1]
            value = summary_df[column].iloc[0]
            records.append({"run": run_tag, "rejection_reason": reason, "count": float(value)})
    return pd.DataFrame.from_records(records)


def _load_ablation(run_path: Path) -> pd.DataFrame:
    ablation_path = run_path / "ablation_summary.csv"
    if not ablation_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(ablation_path)
    if df.empty:
        return df
    df.insert(0, "run", run_path.name)
    return df


def _edge_dataframe(rolling_df: pd.DataFrame, run_tag: str) -> pd.DataFrame:
    if rolling_df.empty:
        return pd.DataFrame()
    column = "edge_margin"
    if column not in rolling_df.columns:
        column = "top_edge_margin"
    if column not in rolling_df.columns:
        return pd.DataFrame()
    data = rolling_df[[column]].dropna().copy()
    if data.empty:
        return pd.DataFrame()
    data.insert(0, "run", run_tag)
    return data


def build_gallery(config_path: Path) -> Path:
    config = _load_config(config_path)
    gallery_cfg = config.get("gallery", {}) or {}
    gallery_name = gallery_cfg.get("name")
    if not gallery_name:
        gallery_name = datetime.utcnow().strftime("gallery_%Y%m%d_%H%M%S")
    gallery_root = Path("figures") / gallery_name

    run_entries = config.get("runs", []) or []
    run_paths = _discover_run_paths(run_entries)
    if not run_paths:
        raise ValueError("No runs found for gallery generation.")

    summary_index: dict[str, list[str]] = defaultdict(list)

    for run_path in run_paths:
        frames = load_run(run_path)
        panel_df = collect_estimator_panel([run_path])
        if panel_df.empty:
            continue

        run_tag = run_path.name
        summary_df = frames["summary"]
        rejection_df = _gather_rejections(summary_df, run_tag) if not summary_df.empty else pd.DataFrame()
        ablation_df = _load_ablation(run_path)
        edge_df = _edge_dataframe(frames["rolling"], run_tag)

        estimators_paths = table_estimators(panel_df, root=gallery_root)
        summary_index[run_tag].append(str(estimators_paths[0]))

        if not rejection_df.empty:
            rej_paths = table_rejections(rejection_df, root=gallery_root)
            summary_index[run_tag].append(str(rej_paths[0]))

        if not ablation_df.empty:
            abl_paths = table_ablation(ablation_df, root=gallery_root)
            summary_index[run_tag].append(str(abl_paths[0]))

        plot_paths = [
            plot_dm_pvals(panel_df, root=gallery_root),
            plot_detection_rate(panel_df, root=gallery_root),
        ]
        summary_index[run_tag].extend(str(path) for path in plot_paths)

        if not edge_df.empty:
            edge_path = plot_edge_margin_hist(edge_df, root=gallery_root)
            summary_index[run_tag].append(str(edge_path))

        if not ablation_df.empty:
            heatmap_path = plot_ablation_heatmap(ablation_df, root=gallery_root)
            summary_index[run_tag].append(str(heatmap_path))

    print(f"Gallery written to {gallery_root.resolve()}")
    for run_tag, artifacts in summary_index.items():
        print(f"- {run_tag}:")
        for artifact in artifacts:
            print(f"  - {artifact}")

    return gallery_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gallery tables and plots from equity runs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/equity_panel/config.gallery.yaml"),
        help="YAML configuration describing runs to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_gallery(args.config.resolve())


if __name__ == "__main__":
    main()
