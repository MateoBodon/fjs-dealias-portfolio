from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, Tuple

try:  # pragma: no cover - plotting optional
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional
    plt = None  # type: ignore[assignment]

from synthetic.calibration import CalibrationConfig, calibrate_thresholds


def _parse_float_list(values: Sequence[str] | None) -> list[float] | None:
    if not values:
        return None
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except ValueError as exc:  # pragma: no cover - argparse guards
            raise argparse.ArgumentTypeError(str(exc))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic calibration for overlay thresholds")
    parser.add_argument(
        "--p-assets",
        type=int,
        nargs="+",
        default=[100, 200],
        help="Asset dimensions to sweep (default: 100 200).",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        nargs="+",
        default=[252],
        help="Replicate group counts (default: 252).",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        nargs="+",
        default=[3],
        help="Replicates per group (space separated list).",
    )
    parser.add_argument("--alpha", type=float, default=0.02, help="Target false positive rate.")
    parser.add_argument("--trials-null", type=int, default=300, help="Number of null simulations per cell.")
    parser.add_argument("--trials-alt", type=int, default=200, help="Number of power simulations per cell.")
    parser.add_argument(
        "--delta-abs-grid",
        type=float,
        nargs="+",
        default=[0.35, 0.45, 0.55, 0.65],
        help="Absolute MP edge buffers δ to evaluate (default: 0.35 0.45 0.55 0.65).",
    )
    parser.add_argument("--eps", type=float, default=0.02, help="Small eps buffer for MP edge.")
    parser.add_argument(
        "--delta-frac-grid",
        type=str,
        nargs="*",
        default=["0.01", "0.015", "0.02", "0.025", "0.03"],
        help="Optional override for delta_frac grid (space separated).",
    )
    parser.add_argument(
        "--stability-grid",
        type=str,
        nargs="*",
        default=["0.30", "0.40", "0.50", "0.60"],
        help="Optional override for stability eta grid (degrees).",
    )
    parser.add_argument("--spike-strength", type=float, default=4.0, help="Signal strength for power trials.")
    parser.add_argument(
        "--edge-modes",
        type=str,
        nargs="+",
        default=["scm", "tyler"],
        help="Edge modes to calibrate (default: scm tyler).",
    )
    parser.add_argument("--q-max", type=int, default=2, help="Maximum detections to accept per window.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducibility.")
    parser.add_argument("--workers", type=int, default=None, help="Optional worker threads for simulation loop.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Trials simulated per worker batch (default: 100).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional checkpoint identifier (defaults to timestamp).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint directory for the provided run id.",
    )
    parser.add_argument(
        "--replicate-bins",
        type=str,
        nargs="*",
        default=None,
        help="Optional replicate bins formatted as label:min-max or min-max (inclusive).",
    )
    parser.add_argument(
        "--asset-bins",
        type=str,
        nargs="*",
        default=None,
        help="Optional asset bins formatted as label:min-max or min-max (inclusive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned sweep metadata and exit without running calibration.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log progress for each calibration job.",
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
        default=None,
        help="Optional defaults lookup path (filters cells with fpr ≤ alpha).",
    )
    return parser.parse_args(argv)


def _parse_bins(specs: Sequence[str] | None, *, prefix: str) -> list[tuple[str, float, float]]:
    if not specs:
        return []
    bins: list[tuple[str, float, float]] = []
    for raw in specs:
        if ":" in raw:
            label, range_part = raw.split(":", 1)
        else:
            label, range_part = raw, raw
        if "-" not in range_part:
            raise argparse.ArgumentTypeError(
                f"Invalid {prefix} bin '{raw}'; expected 'min-max' or 'label:min-max'."
            )
        start_str, end_str = range_part.split("-", 1)
        try:
            start_val = float(start_str)
            end_val = float(end_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
        low, high = sorted((start_val, end_val))
        bins.append((label.strip(), float(low), float(high)))
    return bins


def _assign_bin(value: float, bins: Sequence[tuple[str, float, float]], *, default_prefix: str) -> tuple[str, Tuple[float, float] | None]:
    for label, low, high in bins:
        if low <= value <= high:
            return label, (low, high)
    return f"{default_prefix}{int(round(value))}", None


def _maybe_plot(entries: list[dict[str, float | int | str | None]], alpha: float, path: Path) -> Path | None:
    if plt is None or not entries:  # pragma: no cover - plotting optional
        return None

    edge_modes = sorted({str(entry["edge_mode"]) for entry in entries})
    deltas = sorted({float(entry["delta"]) for entry in entries})
    max_fpr = max((float(entry["fpr"]) for entry in entries), default=alpha)
    max_power = max((float(entry["power"]) if entry["power"] is not None else 0.0 for entry in entries), default=0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    cmap = plt.cm.get_cmap("tab10")
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for edge_idx, edge_mode in enumerate(edge_modes):
        color = cmap(edge_idx % cmap.N)
        for delta_idx, delta in enumerate(deltas):
            xs: list[float] = []
            ys: list[float] = []
            for entry in entries:
                if str(entry["edge_mode"]) != edge_mode:
                    continue
                if abs(float(entry["delta"]) - delta) > 1e-9:
                    continue
                xs.append(float(entry["fpr"]))
                power_val = entry["power"]
                ys.append(float(power_val) if power_val is not None else 0.0)
            if not xs:
                continue
            marker = markers[delta_idx % len(markers)]
            label = f"{edge_mode} δ={delta:.2f}"
            ax.scatter(xs, ys, label=label, marker=marker, color=color, edgecolor="black", linewidth=0.4, s=48)

    ax.axvline(alpha, color="red", linestyle="--", linewidth=1.0, label=f"α={alpha:.2f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Power")
    ax.set_title("Overlay Threshold Calibration")
    ax.set_xlim(0.0, min(0.1, max_fpr * 1.2 + 0.01))
    ax.set_ylim(0.0, min(1.05, max(0.5, max_power * 1.1 + 0.05)))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    plot_path = path.parent / "roc.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def _cell_identifier(p_assets: int, n_groups: int, replicates: int, delta_abs: float, edge_mode: str) -> str:
    delta_token = int(round(delta_abs * 1_000))
    return f"p{p_assets}_g{n_groups}_r{replicates}_d{delta_token}_{edge_mode}"


def _build_cell_records(
    config: CalibrationConfig,
    result: CalibrationResult,
    edge_mode: str,
) -> tuple[list[dict[str, float | int | str | None]], list[dict[str, float | int | str | None]]]:
    cell_entries: list[dict[str, float | int | str | None]] = []
    cell_grid: list[dict[str, float | int | str | None]] = []
    for stat in result.grid:
        record = stat.to_dict()
        record.update(
            {
                "p_assets": config.p_assets,
                "n_groups": config.n_groups,
                "replicates": config.replicates,
                "edge_mode": edge_mode,
            }
        )
        cell_grid.append(record)
        cell_entries.append(
            {
                "p_assets": config.p_assets,
                "n_groups": config.n_groups,
                "replicates": config.replicates,
                "edge_mode": edge_mode,
                "delta": float(stat.delta_abs),
                "delta_frac": float(stat.delta_frac),
                "stability_eta_deg": float(stat.stability_eta_deg),
                "fpr": float(stat.fpr),
                "power": float(stat.power) if stat.power is not None else None,
                "trials_null": int(config.trials_null),
                "trials_alt": int(config.trials_alt),
                "seed": int(config.seed),
            }
        )
    return cell_entries, cell_grid


def _write_cell_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _load_cell_payloads(cells_dir: Path) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    if not cells_dir.exists():
        return payloads
    for cell_path in sorted(cells_dir.glob("*.json")):
        try:
            payloads.append(json.loads(cell_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return payloads


def _collect_cell_records(
    cell_payloads: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, float | int | str | None]], list[dict[str, float | int | str | None]]]:
    entries: list[dict[str, float | int | str | None]] = []
    grid_records: list[dict[str, float | int | str | None]] = []
    for payload in cell_payloads:
        entries.extend(payload.get("entries", []))  # type: ignore[arg-type]
        grid_records.extend(payload.get("grid", []))  # type: ignore[arg-type]
    return entries, grid_records


def _build_threshold_map(
    entries: Sequence[Mapping[str, object]],
    replicate_bins: Sequence[tuple[str, float, float]],
    asset_bins: Sequence[tuple[str, float, float]],
    alpha: float,
) -> tuple[
    dict[str, dict[str, dict[str, dict[str, dict[str, float | int | None]]]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    thresholds_map: dict[str, dict[str, dict[str, dict[str, dict[str, float | int | None]]]]] = {}
    combo_candidates: dict[tuple[int, int, int, str], list[Mapping[str, object]]] = {}
    for entry in entries:
        key = (
            int(entry["p_assets"]),
            int(entry["n_groups"]),
            int(entry["replicates"]),
            str(entry["edge_mode"]).lower(),
        )
        combo_candidates.setdefault(key, []).append(entry)

    for (p_assets, n_groups, replicates, edge_mode), candidates in combo_candidates.items():
        feasible = [cand for cand in candidates if float(cand["fpr"]) <= float(alpha) + 1e-12]
        if feasible:
            feasible.sort(
                key=lambda item: (
                    float(item["delta_frac"]),
                    float(item["stability_eta_deg"]),
                    -float(item["power"]) if item["power"] is not None else 0.0,
                )
            )
            chosen = feasible[0]
        else:
            chosen = min(candidates, key=lambda item: float(item["fpr"]))

        mode_map = thresholds_map.setdefault(edge_mode, {})
        g_key = f"G{n_groups}"
        g_bucket = mode_map.setdefault(g_key, {})
        r_label, r_bounds = _assign_bin(replicates, replicate_bins, default_prefix="r")
        r_bucket = g_bucket.setdefault(r_label, {})
        p_label, p_bounds = _assign_bin(p_assets, asset_bins, default_prefix="p")
        payload = {
            "delta": float(chosen["delta"]),
            "delta_frac": float(chosen["delta_frac"]),
            "stability_eta_deg": float(chosen["stability_eta_deg"]),
            "fpr": float(chosen["fpr"]),
            "power": float(chosen["power"]) if chosen["power"] is not None else None,
            "trials_null": int(chosen["trials_null"]),
            "trials_alt": int(chosen["trials_alt"]),
            "replicates": int(chosen["replicates"]),
            "replicates_bin": r_label,
            "replicates_bin_bounds": r_bounds,
            "p_assets": int(p_assets),
            "p_bin": p_label,
            "p_bin_bounds": p_bounds,
            "n_groups": int(n_groups),
            "seed": int(chosen["seed"]),
        }
        existing = r_bucket.get(p_label)
        if existing is None or payload["fpr"] < existing.get("fpr", float("inf")) - 1e-6:
            r_bucket[p_label] = payload

    replicate_bins_meta = {
        label: {"min": float(low), "max": float(high)} for label, low, high in replicate_bins
    }
    asset_bins_meta = {label: {"min": float(low), "max": float(high)} for label, low, high in asset_bins}
    return thresholds_map, replicate_bins_meta, asset_bins_meta


def _build_defaults_payload(
    thresholds_map: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]]],
    *,
    alpha: float,
    thresholds_path: Path,
) -> dict[str, object]:
    tol = max(1e-6, alpha * 0.01)
    defaults: dict[str, dict[str, dict[str, dict[str, dict[str, object]]]]] = {}

    for edge_mode, g_buckets in thresholds_map.items():
        if not isinstance(g_buckets, Mapping):
            continue
        g_defaults: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
        for g_key, r_buckets in g_buckets.items():
            if not isinstance(r_buckets, Mapping):
                continue
            r_defaults: dict[str, dict[str, dict[str, object]]] = {}
            for r_label, p_buckets in r_buckets.items():
                if not isinstance(p_buckets, Mapping):
                    continue
                p_defaults: dict[str, dict[str, object]] = {}
                for p_label, payload_raw in p_buckets.items():
                    if not isinstance(payload_raw, Mapping):
                        continue
                    fpr = float(payload_raw.get("fpr", 1.0))
                    if fpr > alpha + tol:
                        continue
                    try:
                        delta_val = float(payload_raw["delta"])
                        delta_frac_val = float(payload_raw["delta_frac"])
                        stability_val = float(payload_raw["stability_eta_deg"])
                    except KeyError:
                        continue
                    entry: dict[str, object] = {
                        "delta": delta_val,
                        "delta_frac": delta_frac_val,
                        "stability_eta_deg": stability_val,
                        "fpr": fpr,
                    }
                    if "replicates" in payload_raw:
                        try:
                            entry["replicates"] = int(payload_raw["replicates"])  # type: ignore[arg-type]
                        except Exception:
                            pass
                    if payload_raw.get("replicates_bin_bounds") is not None:
                        entry["replicates_bin_bounds"] = list(payload_raw["replicates_bin_bounds"])  # type: ignore[list-item]
                    if payload_raw.get("p_bin_bounds") is not None:
                        entry["p_bin_bounds"] = list(payload_raw["p_bin_bounds"])  # type: ignore[list-item]
                    p_defaults[p_label] = entry
                if p_defaults:
                    r_defaults[r_label] = p_defaults
            if r_defaults:
                g_defaults[g_key] = r_defaults
        if g_defaults:
            defaults[edge_mode] = g_defaults

    return {
        "alpha": float(alpha),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_source": str(thresholds_path),
        "defaults": defaults,
        "notes": "Lookup of (edge_mode, n_groups, replicates_bin, p_bin) to (delta, delta_frac, eta) filtered to FPR ≤ alpha.",
    }


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    defaults = CalibrationConfig()
    delta_frac_grid = _parse_float_list(args.delta_frac_grid) or list(defaults.delta_frac_grid)
    stability_grid = _parse_float_list(args.stability_grid) or list(defaults.stability_grid)

    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id.")

    run_id = args.run_id or datetime.utcnow().strftime("calib-%Y%m%dT%H%M%SZ")
    run_root = Path("reports/synthetic/calib") / run_id
    cells_dir = run_root / "cells"
    if args.resume:
        if not cells_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {cells_dir}")
    else:
        if run_root.exists():
            raise FileExistsError(f"Run directory already exists: {run_root}")
        cells_dir.mkdir(parents=True, exist_ok=True)

    replicate_bins = _parse_bins(args.replicate_bins, prefix="replicate")
    asset_bins = _parse_bins(args.asset_bins, prefix="asset")

    replicate_values = [int(val) for val in args.replicates]
    if not replicate_values:
        replicate_values = [int(defaults.replicates)]

    out_path = args.out.expanduser().resolve()
    planned_jobs: list[dict[str, object]] = []

    for p_assets in args.p_assets:
        for n_groups in args.n_groups:
            for replicates in replicate_values:
                for delta_abs in args.delta_abs_grid:
                    planned_jobs.append(
                        {
                            "p_assets": int(p_assets),
                            "n_groups": int(n_groups),
                            "replicates": int(replicates),
                            "delta_abs": float(delta_abs),
                        }
                    )

    if args.dry_run:
        total_cells = len(planned_jobs) * len(args.edge_modes)
        total_trials = total_cells * (int(args.trials_null) + int(args.trials_alt))
        approx_seconds = total_trials * max(replicate_values or [defaults.replicates]) * 0.003
        print("[dry-run] synthetic calibration plan")
        print(f"  output: {out_path}")
        print(f"  jobs: {len(planned_jobs)} sweep cells, {len(args.edge_modes)} edge modes")
        print(f"  total trials (null+alt): {total_trials}")
        print(f"  replicates values: {sorted(set(replicate_values))}")
        print(f"  bins (replicates): {replicate_bins if replicate_bins else 'none'}")
        print(f"  bins (assets): {asset_bins if asset_bins else 'none'}")
        print(f"  est runtime ≈ {approx_seconds/60:.1f} minutes (heuristic)")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_cursor = int(args.seed)
    total_jobs = len(planned_jobs) * len(args.edge_modes)
    progress_start = time.perf_counter()
    job_counter = 0
    for job in planned_jobs:
        p_assets = int(job["p_assets"])
        n_groups = int(job["n_groups"])
        replicates = int(job["replicates"])
        delta_abs = float(job["delta_abs"])
        for edge_mode in args.edge_modes:
            job_counter += 1
            if args.verbose:
                print(
                    f"[{job_counter}/{total_jobs}] edge={edge_mode} p={p_assets} G={n_groups} r={replicates} δ={delta_abs:.3f}",
                    flush=True,
                )
            cell_id = _cell_identifier(p_assets, n_groups, replicates, delta_abs, str(edge_mode))
            cell_path = cells_dir / f"{cell_id}.json"
            config = CalibrationConfig(
                p_assets=int(p_assets),
                n_groups=int(n_groups),
                replicates=int(replicates),
                alpha=float(args.alpha),
                trials_null=int(args.trials_null),
                trials_alt=int(args.trials_alt),
                delta_abs=float(delta_abs),
                eps=float(args.eps),
                delta_frac_grid=tuple(float(val) for val in delta_frac_grid),
                stability_grid=tuple(float(val) for val in stability_grid),
                spike_strength=float(args.spike_strength),
                edge_modes=(str(edge_mode),),
                q_max=int(args.q_max),
                seed=seed_cursor,
                workers=int(args.workers) if args.workers is not None else None,
                batch_size=int(args.batch_size),
            )
            seed_cursor += 1
            already_done = args.resume and cell_path.exists()
            if not already_done:
                result = calibrate_thresholds(config)
                cell_entries, cell_grid = _build_cell_records(config, result, str(edge_mode))
                cell_payload = {
                    "cell_id": cell_id,
                    "config": {
                        "p_assets": p_assets,
                        "n_groups": n_groups,
                        "replicates": replicates,
                        "delta_abs": float(delta_abs),
                        "edge_mode": str(edge_mode),
                        "seed": config.seed,
                    },
                    "entries": cell_entries,
                    "grid": cell_grid,
                }
                _write_cell_payload(cell_path, cell_payload)
            elif args.verbose:
                print(f"[resume] skipping completed cell {cell_id}", flush=True)

            elapsed = time.perf_counter() - progress_start
            completed = job_counter
            remaining_jobs = max(total_jobs - completed, 0)
            avg_per_job = elapsed / max(completed, 1)
            eta_seconds = avg_per_job * remaining_jobs
            progress_event = {
                "event": "calibration_progress",
                "edge_mode": edge_mode,
                "current": completed,
                "total": total_jobs,
                "jobs_completed": completed,
                "jobs_total": total_jobs,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
            }
            print(json.dumps(progress_event), flush=True)

    cell_payloads = _load_cell_payloads(cells_dir)
    if not cell_payloads:
        raise RuntimeError(f"No cell payloads found under {cells_dir}")
    entries, grid_records = _collect_cell_records(cell_payloads)

    thresholds_map, replicate_bins_meta, asset_bins_meta = _build_threshold_map(
        entries,
        replicate_bins,
        asset_bins,
        float(args.alpha),
    )

    payload = {
        "alpha": float(args.alpha),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials_null": int(args.trials_null),
        "trials_alt": int(args.trials_alt),
        "run_id": run_id,
        "cells_dir": str(cells_dir),
        "replicates": replicate_values,
        "p_grid": [int(val) for val in args.p_assets],
        "n_grid": [int(val) for val in args.n_groups],
        "delta_grid": [float(val) for val in args.delta_abs_grid],
        "delta_frac_grid": [float(val) for val in delta_frac_grid],
        "stability_grid": [float(val) for val in stability_grid],
        "edge_modes": [str(mode) for mode in args.edge_modes],
        "workers": int(args.workers) if args.workers is not None else None,
        "batch_size": int(args.batch_size),
        "thresholds": thresholds_map,
        "entries": entries,
        "grid": grid_records,
        "replicate_bins": replicate_bins_meta,
        "asset_bins": asset_bins_meta,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _maybe_plot(entries, float(args.alpha), out_path)

    if args.defaults_out:
        defaults_path = args.defaults_out.expanduser().resolve()
        defaults_payload = _build_defaults_payload(
            thresholds_map,
            alpha=float(args.alpha),
            thresholds_path=out_path,
        )
        defaults_path.parent.mkdir(parents=True, exist_ok=True)
        defaults_path.write_text(json.dumps(defaults_payload, indent=2), encoding="utf-8")

    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
