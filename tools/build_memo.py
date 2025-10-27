#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
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


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _discover_run_paths(entries: Iterable[dict]) -> list[Path]:
    run_paths: list[Path] = []
    for entry in entries:
        if not entry:
            continue
        if "path" in entry:
            candidate = Path(entry["path"]).resolve()
            if candidate.is_dir():
                run_paths.append(candidate)
            continue
        root = entry.get("root")
        if not root:
            continue
        pattern = entry.get("pattern")
        try:
            run_paths.extend(find_runs(root, pattern=pattern))
        except FileNotFoundError:
            print(f"[build_memo] Skipping missing root '{root}'", file=sys.stderr)
            continue
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in run_paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no data)"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, int)) and not pd.isna(val):
                values.append(f"{val:.4g}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def build_memo(config_path: Path) -> Path:
    config = _load_config(config_path)
    gallery_cfg = config.get("gallery", {}) or {}
    gallery_name = gallery_cfg.get("name", "memo")
    run_entries = config.get("runs", []) or []
    run_paths = _discover_run_paths(run_entries)
    if not run_paths:
        raise ValueError("No runs discovered for memo generation.")

    gallery_root = Path("figures") / gallery_name
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[pd.DataFrame] = []
    run_lines: list[str] = []
    artifact_map: dict[str, list[str]] = defaultdict(list)

    for run_path in run_paths:
        frames = load_run(run_path)
        panel_df = collect_estimator_panel([run_path])
        if not panel_df.empty:
            combined_rows.append(panel_df.assign(run=run_path.name))
        summary_df = frames["summary"]
        run_meta_path = run_path / "run_meta.json"
        run_meta = {}
        if run_meta_path.exists():
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

        design = run_meta.get("design") or summary_df.get("design").iloc[0] if not summary_df.empty and "design" in summary_df else "n/a"
        nested = run_meta.get("nested_replicates") or summary_df.get("nested_replicates").iloc[0] if not summary_df.empty and "nested_replicates" in summary_df else "n/a"
        start_date = summary_df.get("start_date").iloc[0] if not summary_df.empty and "start_date" in summary_df else "?"
        end_date = summary_df.get("end_date").iloc[0] if not summary_df.empty and "end_date" in summary_df else "?"
        crisis_label = ""
        if "crisis_label" in panel_df and not panel_df["crisis_label"].dropna().empty:
            crisis_label = str(panel_df["crisis_label"].dropna().iloc[0])
        elif not summary_df.empty and "label" in summary_df:
            crisis_label = str(summary_df["label"].iloc[0])
        estimators = sorted(panel_df["estimator"].unique()) if not panel_df.empty else []

        run_label = crisis_label or run_path.name
        run_lines.append(
            f"- **{run_label}** (design={design}, J={nested}, period={start_date} → {end_date}) — estimators: {', '.join(estimators) if estimators else 'n/a'}"
        )

        artifact_base = gallery_root / run_path.name
        tables_dir = artifact_base / "tables"
        plots_dir = artifact_base / "plots"
        if tables_dir.exists():
            artifact_map[run_path.name].append(f"tables: {tables_dir}")
        if plots_dir.exists():
            artifact_map[run_path.name].append(f"plots: {plots_dir}")
        if run_meta_path.exists():
            artifact_map[run_path.name].append(f"run_meta: {run_meta_path}")
        if run_path.name not in artifact_map:
            artifact_map[run_path.name].append("(no gallery artifacts located)")

    if combined_rows:
        combined_table = pd.concat(combined_rows, ignore_index=True, sort=False)
    else:
        combined_table = pd.DataFrame(columns=["run", "crisis_label", "estimator"])

    # Load table outputs for markdown rendering
    table_rows: list[pd.DataFrame] = []
    for run_path in run_paths:
        table_csv = gallery_root / run_path.name / "tables" / "estimators.csv"
        if table_csv.exists():
            df_csv = pd.read_csv(table_csv)
            df_csv.insert(0, "run", run_path.name)
            table_rows.append(df_csv)
    if table_rows:
        table_concat = pd.concat(table_rows, ignore_index=True)
    else:
        table_concat = combined_table.copy()

    if "crisis_label" not in table_concat.columns:
        table_concat["crisis_label"] = ""
    display_columns = [
        col
        for col in ["run", "crisis_label", "estimator", "detection_rate", "delta_mse_ew", "delta_mse_mv", "dm_p_ew", "dm_p_mv", "n_windows"]
        if col in table_concat.columns
    ]
    key_table_md = _markdown_table(table_concat[display_columns]) if not table_concat.empty else "(no data)"

    unique_detection = (
        table_concat.drop_duplicates(subset=["run", "estimator"])["detection_rate"]
        if "detection_rate" in table_concat
        else pd.Series(dtype=float)
    )
    detection_mean = float(unique_detection.dropna().mean()) if not unique_detection.dropna().empty else float("nan")

    delta_series = table_concat["delta_mse_ew"] if "delta_mse_ew" in table_concat else pd.Series(dtype=float)
    delta_share = float((delta_series.dropna() < 0).mean()) if not delta_series.dropna().empty else float("nan")
    delta_median = float(delta_series.median()) if not delta_series.dropna().empty else float("nan")

    dm_series = []
    if "dm_p_ew" in table_concat:
        dm_series.append(table_concat["dm_p_ew"])
    if "dm_p_mv" in table_concat:
        dm_series.append(table_concat["dm_p_mv"])
    if dm_series:
        dm_concat = pd.concat(dm_series)
        dm_sig = int((dm_concat.dropna() < 0.05).sum())
        dm_total = int(dm_concat.dropna().shape[0])
    else:
        dm_sig = 0
        dm_total = 0

    bullet_detection = (
        f"Average detection coverage across RC runs: {_format_percent(detection_mean)}."
        if not pd.isna(detection_mean)
        else "Detection coverage metrics unavailable."
    )
    bullet_mse = (
        f"ΔMSE(EW) < 0 in {_format_percent(delta_share)} of comparisons (median ΔMSE(EW) = {delta_median:.4g})."
        if not pd.isna(delta_share)
        else "ΔMSE directionality not available."
    )
    bullet_dm = (
        f"{dm_sig} of {dm_total} Diebold–Mariano tests show p < 0.05."
        if dm_total
        else "No Diebold–Mariano statistics reported."
    )

    artifact_lines = [
        f"- {run}: " + "; ".join(paths)
        for run, paths in artifact_map.items()
    ]
    artifact_list = "\n".join(artifact_lines) if artifact_lines else "- (no artifacts located)"

    problem_statement = (
        "This memo summarizes the latest smoke and crisis release-candidate runs, "
        "highlighting detection coverage, variance improvements, and statistical significance."
    )

    template_path = Path("reports/templates/memo.md.j2")
    template = template_path.read_text(encoding="utf-8")
    context = {
        "gallery_name": gallery_name,
        "problem_statement": problem_statement,
        "run_summary": "\n".join(run_lines) if run_lines else "(no runs)",
        "key_table": key_table_md,
        "bullet_detection": bullet_detection,
        "bullet_mse": bullet_mse,
        "bullet_dm": bullet_dm,
        "artifact_list": artifact_list,
    }
    memo_text = template.format(**context)

    memo_path = reports_dir / "memo.md"
    memo_path.write_text(memo_text, encoding="utf-8")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    memo_timestamp_path = reports_dir / f"memo_{timestamp}.md"
    memo_timestamp_path.write_text(memo_text, encoding="utf-8")

    print(f"Memo written to {memo_path} and {memo_timestamp_path}")
    return memo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown memo summarizing RC runs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/equity_panel/config.rc.yaml"),
        help="YAML configuration describing runs to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_memo(args.config.resolve())


if __name__ == "__main__":
    main()
