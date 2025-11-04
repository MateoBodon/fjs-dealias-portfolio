from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

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
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per group.")
    parser.add_argument("--alpha", type=float, default=0.02, help="Target false positive rate.")
    parser.add_argument("--trials-null", type=int, default=60, help="Number of null simulations per cell.")
    parser.add_argument("--trials-alt", type=int, default=60, help="Number of power simulations per cell.")
    parser.add_argument(
        "--delta-abs-grid",
        type=float,
        nargs="+",
        default=[0.35, 0.5, 0.65],
        help="Absolute MP edge buffers δ to evaluate (default: 0.35 0.5 0.65).",
    )
    parser.add_argument("--eps", type=float, default=0.02, help="Small eps buffer for MP edge.")
    parser.add_argument(
        "--delta-frac-grid",
        type=str,
        nargs="*",
        default=None,
        help="Optional override for delta_frac grid (space separated).",
    )
    parser.add_argument(
        "--stability-grid",
        type=str,
        nargs="*",
        default=None,
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
        "--out",
        type=Path,
        default=Path("calibration/thresholds.json"),
        help="Output JSON path (default: calibration/thresholds.json).",
    )
    return parser.parse_args(argv)


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


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    defaults = CalibrationConfig()
    delta_frac_grid = _parse_float_list(args.delta_frac_grid) or list(defaults.delta_frac_grid)
    stability_grid = _parse_float_list(args.stability_grid) or list(defaults.stability_grid)

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, float | int | str | None]] = []
    grid_records: list[dict[str, float | int | str | None]] = []
    combo_candidates: dict[tuple[int, int, str], list[dict[str, float | int | str | None]]] = {}

    seed_cursor = int(args.seed)
    for p_assets in args.p_assets:
        for n_groups in args.n_groups:
            for delta_abs in args.delta_abs_grid:
                config = CalibrationConfig(
                    p_assets=int(p_assets),
                    n_groups=int(n_groups),
                    replicates=int(args.replicates),
                    alpha=float(args.alpha),
                    trials_null=int(args.trials_null),
                    trials_alt=int(args.trials_alt),
                    delta_abs=float(delta_abs),
                    eps=float(args.eps),
                    delta_frac_grid=tuple(float(val) for val in delta_frac_grid),
                    stability_grid=tuple(float(val) for val in stability_grid),
                    spike_strength=float(args.spike_strength),
                    edge_modes=tuple(str(mode) for mode in args.edge_modes),
                    q_max=int(args.q_max),
                    seed=seed_cursor,
                    workers=int(args.workers) if args.workers is not None else None,
                )
                seed_cursor += 1
                result = calibrate_thresholds(config)

                for edge_mode, entry in result.thresholds.items():
                    entries.append(
                        {
                            "p_assets": config.p_assets,
                            "n_groups": config.n_groups,
                            "replicates": config.replicates,
                            "edge_mode": edge_mode,
                            "delta": float(config.delta_abs),
                            "delta_frac": float(entry.delta_frac),
                            "stability_eta_deg": float(entry.stability_eta_deg),
                            "fpr": float(entry.fpr),
                            "power": float(entry.power) if entry.power is not None else None,
                            "trials_null": int(entry.trials_null),
                            "trials_alt": int(entry.trials_alt),
                            "seed": config.seed,
                        }
                    )
                    key = (config.p_assets, config.n_groups, str(edge_mode).lower())
                    combo_candidates.setdefault(key, []).append(
                        {
                            "delta": float(config.delta_abs),
                            "delta_frac": float(entry.delta_frac),
                            "stability_eta_deg": float(entry.stability_eta_deg),
                            "fpr": float(entry.fpr),
                            "power": float(entry.power) if entry.power is not None else None,
                            "trials_null": int(entry.trials_null),
                            "trials_alt": int(entry.trials_alt),
                            "replicates": config.replicates,
                            "seed": config.seed,
                        }
                    )
                for stat in result.grid:
                    record = stat.to_dict()
                    record.update(
                        {
                            "p_assets": config.p_assets,
                            "n_groups": config.n_groups,
                            "replicates": config.replicates,
                        }
                    )
                    grid_records.append(record)

    thresholds_map: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for (p_assets, n_groups, edge_mode), candidates in combo_candidates.items():
        feasible = [cand for cand in candidates if cand["fpr"] <= float(args.alpha) + 1e-12]
        if feasible:
            feasible.sort(
                key=lambda item: (
                    float(item["delta"]),
                    -float(item["power"]) if item["power"] is not None else 0.0,
                )
            )
            chosen = feasible[0]
        else:
            chosen = min(candidates, key=lambda item: float(item["fpr"]))

        mode_map = thresholds_map.setdefault(edge_mode, {})
        size_key = f"{p_assets}x{n_groups}"
        mode_map[size_key] = {
            "delta": float(chosen["delta"]),
            "delta_frac": float(chosen["delta_frac"]),
            "stability_eta_deg": float(chosen["stability_eta_deg"]),
            "fpr": float(chosen["fpr"]),
            "power": float(chosen["power"]) if chosen["power"] is not None else None,
            "trials_null": int(chosen["trials_null"]),
            "trials_alt": int(chosen["trials_alt"]),
            "replicates": int(chosen["replicates"]),
            "seed": int(chosen["seed"]),
        }

    payload = {
        "alpha": float(args.alpha),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trials_null": int(args.trials_null),
        "trials_alt": int(args.trials_alt),
        "replicates": int(args.replicates),
        "p_grid": [int(val) for val in args.p_assets],
        "n_grid": [int(val) for val in args.n_groups],
        "delta_grid": [float(val) for val in args.delta_abs_grid],
        "delta_frac_grid": [float(val) for val in delta_frac_grid],
        "stability_grid": [float(val) for val in stability_grid],
        "edge_modes": [str(mode) for mode in args.edge_modes],
        "workers": int(args.workers) if args.workers is not None else None,
        "thresholds": thresholds_map,
        "entries": entries,
        "grid": grid_records,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _maybe_plot(entries, float(args.alpha), out_path)
    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
