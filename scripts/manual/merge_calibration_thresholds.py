#!/usr/bin/env python3
"""
Merge individual calibration slices into a single thresholds bundle.

Usage:
    python scripts/manual/merge_calibration_thresholds.py \
        --inputs reports/rc-20251104/calibration/thresholds_p100.json \
                 reports/rc-20251104/calibration/thresholds_p200.json \
        --out reports/rc-20251104/calibration/thresholds.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge calibration threshold slices")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input JSON files to merge.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional ROC plot output path (overwrites if exists).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return payload


def _sorted_unique(values: Iterable) -> list:
    return sorted({val for val in values})


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.inputs]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    payloads = [_load_json(path) for path in inputs]
    alpha_vals = {float(payload["alpha"]) for payload in payloads if "alpha" in payload}
    if len(alpha_vals) != 1:
        raise ValueError(f"Inconsistent alpha values: {alpha_vals}")
    alpha = alpha_vals.pop() if alpha_vals else 0.02

    combined_entries: list[dict] = []
    combined_grid: list[dict] = []
    thresholds: dict[str, dict[str, dict]] = {}

    p_values: set[int] = set()
    n_values: set[int] = set()
    delta_values: set[float] = set()
    delta_frac_values: set[float] = set()
    stability_values: set[float] = set()
    edge_modes: set[str] = set()
    trials_null = 0
    trials_alt = 0
    replicates = 0
    worker_values: set[int] = set()
    worker_none = False

    for payload in payloads:
        combined_entries.extend(payload.get("entries", []))
        combined_grid.extend(payload.get("grid", []))
        thresholds_slice = payload.get("thresholds", {})
        if not isinstance(thresholds_slice, dict):
            continue
        for edge_mode, grid_map in thresholds_slice.items():
            if not isinstance(grid_map, dict):
                continue
            mode_key = str(edge_mode)
            mode_bucket = thresholds.setdefault(mode_key, {})
            for size_key, entry in grid_map.items():
                mode_bucket[size_key] = entry
        p_values.update(int(p) for p in payload.get("p_grid", []))
        n_values.update(int(n) for n in payload.get("n_grid", []))
        delta_values.update(float(d) for d in payload.get("delta_grid", []))
        delta_frac_values.update(float(d) for d in payload.get("delta_frac_grid", []))
        stability_values.update(float(s) for s in payload.get("stability_grid", []))
        edge_modes.update(str(mode) for mode in payload.get("edge_modes", []))
        trials_null = max(trials_null, int(payload.get("trials_null", 0)))
        trials_alt = max(trials_alt, int(payload.get("trials_alt", 0)))
        replicates = max(replicates, int(payload.get("replicates", 0)))
        workers_field = payload.get("workers")
        if isinstance(workers_field, (list, tuple)):
            for item in workers_field:
                if item is None:
                    worker_none = True
                else:
                    worker_values.add(int(item))
        elif workers_field is None:
            worker_none = True
        else:
            worker_values.add(int(workers_field))

    combined_payload = {
        "alpha": alpha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials_null": trials_null,
        "trials_alt": trials_alt,
        "replicates": replicates,
        "p_grid": sorted(p_values),
        "n_grid": sorted(n_values),
        "delta_grid": sorted(delta_values),
        "delta_frac_grid": sorted(delta_frac_values),
        "stability_grid": sorted(stability_values),
        "edge_modes": sorted(edge_modes),
        "workers": sorted(worker_values) + ([None] if worker_none else []),
        "inputs": [str(path) for path in inputs],
        "thresholds": thresholds,
        "entries": combined_entries,
        "grid": combined_grid,
    }

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")

    plot_target = args.plot.resolve() if args.plot else out_path.parent / "roc.png"
    try:
        from experiments.synthetic.calibrate_thresholds import _maybe_plot
    except ImportError:  # pragma: no cover - optional plotting
        _maybe_plot = None  # type: ignore[assignment]
    if _maybe_plot is not None:
        generated = _maybe_plot(combined_entries, alpha, plot_target)
        if generated and generated != plot_target:
            # When helper returns a different default path, copy to the requested location.
            target = plot_target
            target.parent.mkdir(parents=True, exist_ok=True)
            data = Path(generated).read_bytes()
            target.write_bytes(data)


if __name__ == "__main__":
    main()
