from __future__ import annotations

import argparse
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

from experiments.synthetic.harness_utils import HarnessConfig, simulate_scores, write_run_metadata

DEFAULT_EDGE_MODES = ("scm", "tyler")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Synthetic null harness for MP edge calibration")
    parser.add_argument("--n-assets", type=int, default=40)
    parser.add_argument("--n-groups", type=int, default=60)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--signal-to-noise", type=float, default=0.35)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edge-modes",
        type=str,
        nargs="+",
        default=list(DEFAULT_EDGE_MODES),
        help="Edge mode scatter estimators to evaluate (default: scm tyler).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/synthetic/null_harness"),
        help="Output directory for tabular artifacts.",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=Path("reports/figures"),
        help="Directory to store generated figures.",
    )
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_fpr_curve(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    max_score = float(np.nanmax(scores["score"].to_numpy(dtype=np.float64))) if not scores.empty else 0.0
    thresholds = np.linspace(0.0, max(max_score, 0.0), num=101)
    for mode in sorted(scores["edge_mode"].unique()):
        values = scores[scores["edge_mode"] == mode]["score"].to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        for threshold in thresholds:
            fpr = float(np.mean(values >= threshold)) if values.size else 0.0
            rows.append(
                {
                    "edge_mode": mode,
                    "threshold": float(threshold),
                    "fpr": float(fpr),
                }
            )
    return pd.DataFrame(rows)


def _plot_fpr_curve(curve: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, data in curve.groupby("edge_mode"):
        ax.plot(data["threshold"], data["fpr"], label=mode.upper())
    ax.set_xlabel("Energy floor threshold")
    ax.set_ylabel("False positive rate")
    ax.set_title("Null survival curve across edge modes")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Edge mode")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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

    start = time.time()
    scores = simulate_scores(config, mu_values=[0.0], scenario_prefix="null-")
    elapsed = time.time() - start

    scores_path = out_dir / "null_scores.parquet"
    scores.scores.to_parquet(scores_path, index=False)

    curve = _build_fpr_curve(scores.scores)
    curve_path = out_dir / "null_fpr_curve.csv"
    curve.to_csv(curve_path, index=False)

    roc_path = fig_dir / "roc_null.png"
    _plot_fpr_curve(curve, roc_path)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "duration_seconds": elapsed,
        "artifacts": {
            "scores": str(scores_path),
            "fpr_curve": str(curve_path),
            "figure": str(roc_path),
        },
        "config": config.to_json(),
    }
    write_run_metadata(out_dir / "run.json", config=metadata, extra={})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
