#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _load_detection_total(path: Path) -> int:
    det_path = path / "detection_summary.csv"
    if not det_path.exists():
        return 0
    try:
        df = pd.read_csv(det_path, usecols=["n_detections"])
    except Exception:
        return 0
    return int(pd.to_numeric(df["n_detections"], errors="coerce").fillna(0).sum())


@dataclass(slots=True)
class RunInfo:
    label: str
    directory: Path
    start_date: str
    end_date: str
    windows: int
    detections: int
    leak_cap: float | None
    energy_min: float | None
    delta_frac: float | None
    signed_a: bool | None


def _extract_run_info(label: str, path: Path) -> RunInfo | None:
    summary = _load_json(path / "summary.json")
    if not summary:
        return None
    run_meta = _load_json(path / "run_meta.json")
    cfg = run_meta.get("config_snapshot", {}) if run_meta else {}
    detections = int(run_meta.get("detections_total", 0)) if run_meta else 0
    if detections == 0:
        detections = _load_detection_total(path)

    def _get_float(mapping: dict[str, Any], key: str) -> float | None:
        if key not in mapping:
            return None
        try:
            value = float(mapping[key])
        except (TypeError, ValueError):
            return None
        return value

    leak_cap = _get_float(cfg, "off_component_leak_cap")
    energy_min = _get_float(cfg, "energy_min_abs")
    delta_frac = _get_float(cfg, "dealias_delta_frac")
    signed_a = cfg.get("signed_a")
    if isinstance(signed_a, str):
        signed_a = signed_a.lower() == "true"
    elif not isinstance(signed_a, bool):
        signed_a = None

    return RunInfo(
        label=label,
        directory=path,
        start_date=str(summary.get("start_date", "?")),
        end_date=str(summary.get("end_date", "?")),
        windows=int(summary.get("rolling_windows_evaluated", 0)),
        detections=int(detections),
        leak_cap=leak_cap,
        energy_min=energy_min,
        delta_frac=delta_frac,
        signed_a=signed_a,
    )


def discover_runs(base_dir: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    primary_dir = base_dir / "outputs"
    info = _extract_run_info("outputs (current)", primary_dir)
    if info:
        runs.append(info)

    archive_dir = base_dir / "archive"
    if archive_dir.exists():
        for child in sorted(archive_dir.iterdir()):
            if child.is_dir():
                child_info = _extract_run_info(child.name, child)
                if child_info:
                    runs.append(child_info)
    return runs


def format_runs(runs: list[RunInfo]) -> str:
    if not runs:
        return "No runs with summary.json found."

    def fmt_float(value: float | None) -> str:
        if value is None:
            return "n/a"
        if abs(value) >= 1 or value == 0.0:
            return f"{value:.2f}"
        return f"{value:.2e}"

    header = [
        "label",
        "start",
        "end",
        "windows",
        "detections",
        "leak_cap",
        "energy_min",
        "delta_frac",
        "signed_a",
        "path",
    ]
    rows = [
        [
            run.label,
            run.start_date,
            run.end_date,
            str(run.windows),
            str(run.detections),
            fmt_float(run.leak_cap),
            fmt_float(run.energy_min),
            fmt_float(run.delta_frac),
            "yes" if run.signed_a else "no" if run.signed_a is not None else "n/a",
            str(run.directory),
        ]
        for run in runs
    ]
    widths = [len(col) for col in header]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    lines = [fmt_line(header), fmt_line(["-" * w for w in widths])]
    lines.extend(fmt_line(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List available equity runs and key guardrail settings."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("experiments/equity_panel"),
        help="Base directory with outputs/ and archive/ subdirectories.",
    )
    args = parser.parse_args()
    runs = discover_runs(args.base)
    print(format_runs(runs))


if __name__ == "__main__":
    main()

