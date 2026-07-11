#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from fjs.real_design_contract import (  # noqa: E402
    RealDesignCellSpec,
    bind_factor_source,
    bind_source_partition,
    derive_real_design_cell,
    file_sha256,
    load_bound_factors,
    load_filtered_sources,
    stable_json_dumps,
    stable_sha256,
    write_real_design_cell,
)
from fjs.real_design_finalizer import (  # noqa: E402
    build_cell_receipt,
    build_final_manifest,
    checkpoint_status,
    expected_cell_id,
    independent_readback,
    load_checkpoint,
    new_checkpoint,
    register_cell,
    required_months,
    write_checkpoint,
    write_final_manifest,
    write_readback,
)

SOURCE_ROOT = Path(
    "/Volumes/Storage/Data/WRDS/raw/crsp/wrds_dsfv2_query/"
    "snapshot=20260707_045553_global_project_priority"
)
RECEIPT_2013_2017 = Path(
    "/Volumes/Storage/Data/wrds/_manifests/"
    "20260707T214900Z_worker8_crsp_dsfv2_month_2017_2010_csvgz/manifest.json"
)
RECEIPT_2018 = Path(
    "/Volumes/Storage/Data/wrds/_manifests/"
    "20260707T204600Z_worker7_crsp_dsfv2_month_recent_csvgz/manifest.json"
)


def _git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _receipt_for_month(month: str) -> Path:
    return RECEIPT_2013_2017 if month < "2018-01" else RECEIPT_2018


def _source_for_month(month: str) -> Path:
    return SOURCE_ROOT / f"month={month}" / "data.csv.gz"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(stable_json_dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_progress(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _month_bounds(month: str) -> tuple[str, str]:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthEnd(1)
    return start.date().isoformat(), pd.Timestamp(end).date().isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run the restart-safe 72-month FJS v4 realistic-design input generation."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument(
        "--factors-csv",
        type=Path,
        default=ROOT / "data/factors/ff5mom_daily.csv",
    )
    parser.add_argument(
        "--factor-registry",
        type=Path,
        default=ROOT / "data/factors/registry.json",
    )
    parser.add_argument("--universe-size", type=int, default=60)
    parser.add_argument("--fit-sessions", type=int, default=10)
    parser.add_argument("--min-factor-observations", type=int, default=8)
    parser.add_argument("--min-window-observations", type=int, default=8)
    parser.add_argument("--min-pairwise-observations", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument("--max-months", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_root / "checkpoint.json"
    progress_path = run_root / "progress.jsonl"
    final_manifest_path = run_root / "full_manifest.json"
    readback_path = run_root / "readback.json"
    cells_root = run_root / "cells"

    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["generation_id"] != args.generation_id:
            raise ValueError("Existing checkpoint belongs to another generation.")
    else:
        checkpoint = new_checkpoint(str(args.generation_id))
        write_checkpoint(checkpoint, checkpoint_path)

    factor_binding = bind_factor_source(args.factors_csv, args.factor_registry)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.monotonic()
    completed_this_process = 0
    timings: list[float] = []
    generator_binding = {
        "path": "tools/run_fjs_m4_real_design_full_v4.py",
        "sha256": file_sha256(Path(__file__).resolve()),
        "size_bytes": Path(__file__).resolve().stat().st_size,
        "git_head": _git_value("rev-parse", "HEAD"),
        "git_tree": _git_value("rev-parse", "HEAD^{tree}"),
    }
    launch = {
        "event": "launch",
        "pid": os.getpid(),
        "generation_id": args.generation_id,
        "run_root": str(run_root),
        "checkpoint": str(checkpoint_path),
        "generator_binding": generator_binding,
        "full_execution_ready": False,
        "aws_execution_authorized": False,
        "outcomes_present": False,
        "holdout_2025_opened": False,
    }
    print(json.dumps(launch, sort_keys=True), flush=True)
    _append_progress(launch, progress_path)

    completed = set(checkpoint_status(checkpoint)["completed_months"])
    pending = [month for month in required_months() if month not in completed]
    if args.max_months is not None:
        pending = pending[: int(args.max_months)]

    for month in pending:
        month_started = time.monotonic()
        month_start, month_end = _month_bounds(month)
        source_binding = bind_source_partition(
            _source_for_month(month), _receipt_for_month(month)
        )
        source_frame, scan = load_filtered_sources(
            [source_binding],
            start=month_start,
            end=month_end,
            chunksize=int(args.chunksize),
        )
        dates = sorted(
            pd.Timestamp(value) for value in source_frame["dlycaldt"].unique()
        )
        fit_sessions = int(args.fit_sessions)
        if len(dates) <= fit_sessions + int(args.min_window_observations) - 1:
            raise ValueError(
                f"Month {month} has only {len(dates)} eligible sessions for the "
                "frozen fit/window split."
            )
        fit_start = dates[0]
        fit_end = dates[fit_sessions - 1]
        window_start = dates[fit_sessions]
        window_end = dates[-1]
        spec = RealDesignCellSpec(
            cell_id=expected_cell_id(month),
            factor_fit_start=fit_start.date().isoformat(),
            factor_fit_end=fit_end.date().isoformat(),
            formation_date=fit_end.date().isoformat(),
            window_start=window_start.date().isoformat(),
            window_end=window_end.date().isoformat(),
            universe_size=int(args.universe_size),
            min_factor_observations=int(args.min_factor_observations),
            min_window_observations=int(args.min_window_observations),
            min_pairwise_observations=int(args.min_pairwise_observations),
            max_cap_staleness_days=5,
            proof_only=False,
        )
        factors = load_bound_factors(
            factor_binding,
            start=spec.factor_fit_start,
            end=spec.window_end,
        )
        cell = derive_real_design_cell(
            source_frame,
            factors,
            spec=spec,
            source_bindings=[source_binding],
            factor_binding=factor_binding,
            scan_receipt=scan,
        )
        cell["generation_metadata"] = {
            "generation_id": args.generation_id,
            "month": month,
            "generator_binding": generator_binding,
            "monthly_input_only": True,
            "detector_outcomes_present": False,
            "holdout_2025_opened": False,
            "aws_execution_authorized": False,
        }
        cell["cell_digest"] = stable_sha256(
            {key: value for key, value in cell.items() if key != "cell_digest"}
        )
        cell_path = cells_root / f"{expected_cell_id(month)}.json"
        artifact = write_real_design_cell(cell, cell_path)
        receipt = build_cell_receipt(
            generation_id=str(args.generation_id), month=month, cell_path=cell_path
        )
        checkpoint = register_cell(checkpoint, receipt)
        write_checkpoint(checkpoint, checkpoint_path)

        elapsed = time.monotonic() - month_started
        timings.append(elapsed)
        completed_this_process += 1
        status = checkpoint_status(checkpoint)
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        remaining = len(status["missing_months"])
        mean_seconds = sum(timings) / len(timings)
        progress = {
            "event": "month_complete",
            "pid": os.getpid(),
            "generation_id": args.generation_id,
            "month": month,
            "cell_id": expected_cell_id(month),
            "cell_digest": artifact["cell_digest"],
            "cell_file_sha256": artifact["sha256"],
            "source_sha256": source_binding.sha256,
            "receipt_manifest_sha256": source_binding.receipt_manifest_sha256,
            "rows_scanned": scan["partitions"][0]["rows_scanned"],
            "rows_after_filters": scan["rows_after_all_filters"],
            "exact_duplicates_collapsed": scan["exact_duplicate_rows_collapsed"],
            "elapsed_seconds": elapsed,
            "process_rss_bytes": rss,
            "process_peak_rss_bytes": peak_rss,
            "completed_count": status["completion_count"],
            "remaining_count": remaining,
            "measured_eta_seconds": mean_seconds * remaining,
            "checkpoint_digest": status["checkpoint_digest"],
            "outcomes_present": False,
            "holdout_2025_opened": False,
            "aws_execution_authorized": False,
        }
        print(json.dumps(progress, sort_keys=True), flush=True)
        _append_progress(progress, progress_path)

    status = checkpoint_status(checkpoint)
    if status["complete"]:
        manifest = build_final_manifest(checkpoint)
        write_final_manifest(manifest, final_manifest_path)
        readback = independent_readback(final_manifest_path)
        write_readback(readback, readback_path)
        finished = {
            "event": "generation_complete",
            "pid": os.getpid(),
            "generation_id": args.generation_id,
            "elapsed_seconds": time.monotonic() - started,
            "completed_this_process": completed_this_process,
            "manifest_path": str(final_manifest_path),
            "manifest_file_sha256": readback["manifest_file_sha256"],
            "manifest_digest": readback["manifest_digest"],
            "source_set_digest": readback["source_set_digest"],
            "cell_set_digest": readback["cell_set_digest"],
            "readback_path": str(readback_path),
            "readback_digest": readback["readback_digest"],
            "process_rss_bytes": process.memory_info().rss,
            "process_peak_rss_bytes": peak_rss,
            "full_execution_ready": False,
            "outcomes_present": False,
            "holdout_2025_opened": False,
            "aws_execution_authorized": False,
        }
        _atomic_json(finished, run_root / "generation_receipt.json")
        print(json.dumps(finished, sort_keys=True), flush=True)
        _append_progress(finished, progress_path)
    else:
        partial = {
            "event": "partial_stop",
            "pid": os.getpid(),
            "generation_id": args.generation_id,
            "completed_count": status["completion_count"],
            "missing_months": status["missing_months"],
            "checkpoint_digest": status["checkpoint_digest"],
            "outcomes_present": False,
            "holdout_2025_opened": False,
            "aws_execution_authorized": False,
        }
        print(json.dumps(partial, sort_keys=True), flush=True)
        _append_progress(partial, progress_path)
    return run_root


if __name__ == "__main__":  # pragma: no cover
    main()
