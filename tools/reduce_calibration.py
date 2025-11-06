#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from experiments.synthetic import calibrate_thresholds as calib_mod


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge calibration cell checkpoints into final artifacts.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Checkpoint run identifier under reports/synthetic/calib/.",
    )
    parser.add_argument(
        "--cells-dir",
        type=str,
        default=None,
        help="Explicit path to the cells directory (overrides --run-id).",
    )
    parser.add_argument("--alpha", type=float, default=0.02, help="Target false positive rate (default: 0.02).")
    parser.add_argument(
        "--replicate-bins",
        type=str,
        nargs="*",
        default=None,
        help="Optional replicate bins formatted as label:min-max.",
    )
    parser.add_argument(
        "--asset-bins",
        type=str,
        nargs="*",
        default=None,
        help="Optional asset bins formatted as label:min-max.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("calibration/edge_delta_thresholds.json"),
        help="Output JSON path (default: calibration/edge_delta_thresholds.json).",
    )
    parser.add_argument(
        "--defaults-out",
        type=Path,
        default=Path("calibration/defaults.json"),
        help="Defaults output path (default: calibration/defaults.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cells_dir:
        cells_dir = Path(args.cells_dir).expanduser()
    else:
        if not args.run_id:
            raise ValueError("Provide either --run-id or --cells-dir.")
        cells_dir = Path("reports/synthetic/calib") / args.run_id / "cells"
    if not cells_dir.exists():
        raise FileNotFoundError(f"Cells directory not found: {cells_dir}")

    cell_payloads = calib_mod._load_cell_payloads(cells_dir)  # type: ignore[attr-defined]
    if not cell_payloads:
        raise RuntimeError(f"No cell payloads found under {cells_dir}")
    entries, grid_records = calib_mod._collect_cell_records(cell_payloads)  # type: ignore[attr-defined]
    if not entries:
        raise RuntimeError("No entries were recovered from the supplied checkpoints.")

    replicate_bins = calib_mod._parse_bins(args.replicate_bins, prefix="replicate")  # type: ignore[attr-defined]
    asset_bins = calib_mod._parse_bins(args.asset_bins, prefix="asset")  # type: ignore[attr-defined]

    thresholds_map, replicate_bins_meta, asset_bins_meta = calib_mod._build_threshold_map(  # type: ignore[attr-defined]
        entries,
        replicate_bins,
        asset_bins,
        float(args.alpha),
    )

    replicate_values = sorted({int(entry["replicates"]) for entry in entries})
    p_grid = sorted({int(entry["p_assets"]) for entry in entries})
    n_grid = sorted({int(entry["n_groups"]) for entry in entries})
    delta_grid = sorted({float(record["delta"]) for record in grid_records})
    delta_frac_grid = sorted({float(entry["delta_frac"]) for entry in entries})
    stability_grid = sorted({float(entry["stability_eta_deg"]) for entry in entries})
    edge_modes = sorted({str(entry["edge_mode"]) for entry in entries})
    trials_null = int(entries[0]["trials_null"])
    trials_alt = int(entries[0]["trials_alt"])

    run_id = args.run_id or cells_dir.parent.name

    payload = {
        "alpha": float(args.alpha),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials_null": trials_null,
        "trials_alt": trials_alt,
        "run_id": run_id,
        "cells_dir": str(cells_dir),
        "replicates": replicate_values,
        "p_grid": p_grid,
        "n_grid": n_grid,
        "delta_grid": delta_grid,
        "delta_frac_grid": delta_frac_grid,
        "stability_grid": stability_grid,
        "edge_modes": edge_modes,
        "workers": None,
        "batch_size": None,
        "thresholds": thresholds_map,
        "entries": entries,
        "grid": grid_records,
        "replicate_bins": replicate_bins_meta,
        "asset_bins": asset_bins_meta,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    calib_mod._maybe_plot(entries, float(args.alpha), args.out)  # type: ignore[attr-defined]

    if args.defaults_out:
        defaults_payload = calib_mod._build_defaults_payload(  # type: ignore[attr-defined]
            thresholds_map,
            alpha=float(args.alpha),
            thresholds_path=args.out,
        )
        args.defaults_out.parent.mkdir(parents=True, exist_ok=True)
        args.defaults_out.write_text(json.dumps(defaults_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
