from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.synthetic.harness_utils import (
    EnergyFloorSelection,
    HarnessConfig,
    roc_table,
    select_energy_floor,
    simulate_scores,
    write_run_metadata,
)

DEFAULT_EDGE_MODES = ("scm", "tyler")
DEFAULT_MU = (4.0, 6.0, 8.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic power harness with ROC and default selection")
    parser.add_argument("--n-assets", type=int, default=40)
    parser.add_argument("--n-groups", type=int, default=60)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--signal-to-noise", type=float, default=0.35)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument(
        "--mu-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_MU),
        help="Spike strengths μ to evaluate (default: 4 6 8).",
    )
    parser.add_argument(
        "--edge-modes",
        type=str,
        nargs="+",
        default=list(DEFAULT_EDGE_MODES),
        help="Edge mode scatter estimators to evaluate (default: scm tyler).",
    )
    parser.add_argument(
        "--null-scores",
        type=Path,
        default=None,
        help="Optional path to pre-computed null score parquet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/synthetic/power_harness"),
        help="Output directory for tabular artifacts.",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--defaults-path",
        type=Path,
        default=Path("calibration_defaults.json"),
        help="Path to write the selected calibration defaults.",
    )
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--delta-frac", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--stability-eta", type=float, default=0.4)
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_null_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Null score file '{path}' not found.")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    expected = {"edge_mode", "score"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Null scores must contain columns {sorted(expected)}.")
    return frame


def _plot_roc(roc: pd.DataFrame, mu_values: Sequence[float], selection: EnergyFloorSelection | None, path: Path) -> None:
    modes = sorted(roc["edge_mode"].unique())
    if not modes:
        return
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        data = roc[roc["edge_mode"] == mode]
        for mu in mu_values:
            subset = data[np.isclose(data["mu"], mu)]
            ax.plot(
                subset["fpr"],
                subset["tpr"],
                label=f"μ={mu:g}",
            )
            if selection is not None and selection.edge_mode == mode:
                point = subset[np.isclose(subset["threshold"], selection.threshold)]
                if not point.empty:
                    ax.scatter(point["fpr"], point["tpr"], color="black", marker="x", s=60)
        ax.set_title(f"{mode.upper()} edge")
        ax.set_xlabel("False positive rate")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(title="Spike strength")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_defaults(
    path: Path,
    *,
    selection: EnergyFloorSelection | None,
    config: HarnessConfig,
    args: argparse.Namespace,
    mu_values: Sequence[float],
) -> None:
    payload = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "mu_values": [float(mu) for mu in mu_values],
        "target_fpr": float(args.target_fpr),
        "parameters": {
            "delta": float(args.delta),
            "delta_frac": float(args.delta_frac),
            "eps": float(args.eps),
            "stability_eta_deg": float(args.stability_eta),
        },
        "score_metric": "lambda_minus_mp_edge",
        "config": config.to_json(),
    }
    if selection is not None:
        payload["selection"] = selection.to_json()
        payload["parameters"]["energy_floor"] = float(selection.threshold)
        payload["parameters"]["edge_mode"] = selection.edge_mode
    else:
        payload["selection"] = None
        payload["parameters"]["energy_floor"] = None
        payload["parameters"]["edge_mode"] = config.edge_modes[0] if config.edge_modes else "scm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    fig_dir = Path(args.figures_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    config = HarnessConfig(
        n_assets=args.n_assets,
        n_groups=args.n_groups,
        replicates=args.replicates,
        noise_variance=args.noise_variance,
        signal_to_noise=args.signal_to_noise,
        edge_modes=tuple(str(mode) for mode in args.edge_modes),
        trials=args.trials,
        seed=args.seed,
    )

    if args.null_scores is not None:
        null_scores = _load_null_scores(args.null_scores)
    else:
        null_scores = simulate_scores(config, mu_values=[0.0], scenario_prefix="null-").scores

    mu_values = tuple(float(mu) for mu in args.mu_values)
    power_frames: dict[float, pd.DataFrame] = {}
    start = time.time()
    for mu in mu_values:
        result = simulate_scores(config, mu_values=[mu], scenario_prefix="mu-")
        power_frames[float(mu)] = result.scores
    elapsed = time.time() - start

    power_df = pd.concat(power_frames.values(), ignore_index=True) if power_frames else pd.DataFrame()
    null_path = out_dir / "null_scores.parquet"
    power_path = out_dir / "power_scores.parquet"
    null_scores.to_parquet(null_path, index=False)
    power_df.to_parquet(power_path, index=False)

    roc = roc_table(null_scores, power_frames)
    roc_path = out_dir / "roc_table.csv"
    roc.to_csv(roc_path, index=False)

    selection = select_energy_floor(null_scores, power_frames, target_fpr=args.target_fpr)

    defaults_path = Path(args.defaults_path)
    _save_defaults(defaults_path, selection=selection, config=config, args=args, mu_values=mu_values)

    figure_path = fig_dir / "roc_power.png"
    _plot_roc(roc, mu_values, selection, figure_path)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "duration_seconds": elapsed,
        "target_fpr": float(args.target_fpr),
        "mu_values": [float(mu) for mu in mu_values],
        "artifacts": {
            "null_scores": str(null_path),
            "power_scores": str(power_path),
            "roc_table": str(roc_path),
            "roc_power_figure": str(figure_path),
            "defaults": str(defaults_path),
        },
        "selection": selection.to_json() if selection is not None else None,
        "parameters": {
            "delta": float(args.delta),
            "delta_frac": float(args.delta_frac),
            "eps": float(args.eps),
            "stability_eta_deg": float(args.stability_eta),
        },
        "config": config.to_json(),
    }
    write_run_metadata(out_dir / "run.json", config=metadata, extra={})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
