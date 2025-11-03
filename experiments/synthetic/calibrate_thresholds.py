from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from synthetic.calibration import CalibrationConfig, calibrate_thresholds, write_thresholds


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
    parser.add_argument("--p-assets", type=int, default=40, help="Number of assets (p dimension).")
    parser.add_argument("--n-groups", type=int, default=180, help="Number of replicate groups (T dimension).")
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per group.")
    parser.add_argument("--alpha", type=float, default=0.02, help="Target false positive rate.")
    parser.add_argument("--trials-null", type=int, default=400, help="Number of null simulations.")
    parser.add_argument("--trials-alt", type=int, default=200, help="Number of power simulations.")
    parser.add_argument("--delta-abs", type=float, default=0.5, help="Absolute delta buffer for MP edge.")
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
        nargs="*",
        default=["tyler", "huber"],
        help="Edge modes to calibrate (default: tyler huber).",
    )
    parser.add_argument("--q-max", type=int, default=2, help="Maximum detections to accept per window.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("calibration/thresholds.json"),
        help="Output JSON path (default: calibration/thresholds.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    defaults = CalibrationConfig()
    delta_frac_grid = _parse_float_list(args.delta_frac_grid) or defaults.delta_frac_grid
    stability_grid = _parse_float_list(args.stability_grid) or defaults.stability_grid

    config = CalibrationConfig(
        p_assets=args.p_assets,
        n_groups=args.n_groups,
        replicates=args.replicates,
        alpha=args.alpha,
        trials_null=args.trials_null,
        trials_alt=args.trials_alt,
        delta_abs=args.delta_abs,
        eps=args.eps,
        delta_frac_grid=tuple(delta_frac_grid),
        stability_grid=tuple(stability_grid),
        spike_strength=args.spike_strength,
        edge_modes=tuple(args.edge_modes),
        q_max=args.q_max,
        seed=args.seed,
    )

    result = calibrate_thresholds(config)
    output_path = write_thresholds(result, args.out)
    return output_path


if __name__ == "__main__":  # pragma: no cover
    main()
