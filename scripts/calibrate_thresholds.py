"""Synthetic calibration harness for de-aliasing thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.fjs.overlay import OverlayConfig, detect_spikes


@dataclass(slots=True, frozen=True)
class CalibrationConfig:
    """Configuration options for the calibration sweep."""

    p: int
    n: int
    mu: tuple[float, ...]
    out: Path
    trials: int = 200
    seed: int = 0
    margin_grid: tuple[float, ...] = (0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2)
    isolation: float = 0.05
    fpr_target: float = 0.02

    def __post_init__(self) -> None:
        if self.p <= 0 or self.n <= 1:
            raise ValueError("p and n must be positive (n > 1).")
        if not self.mu:
            raise ValueError("At least one mu value must be supplied.")
        if self.trials <= 0:
            raise ValueError("trials must be positive.")
        if any(m <= 0.0 for m in self.margin_grid):
            raise ValueError("All margin thresholds must be positive.")


@dataclass(slots=True)
class CalibrationResult:
    margin_grid: tuple[float, ...]
    fpr: list[float]
    tpr: dict[str, list[float]]
    recommended_margin: float
    recommended_fpr: float
    recommended_tpr: dict[str, float]

    def to_json(self, config: CalibrationConfig) -> dict[str, object]:
        return {
            "p": config.p,
            "n": config.n,
            "mu": list(map(float, config.mu)),
            "margin_grid": list(self.margin_grid),
            "fpr": self.fpr,
            "tpr": self.tpr,
            "recommended": {
                "min_margin": self.recommended_margin,
                "fpr": self.recommended_fpr,
                "tpr": self.recommended_tpr,
                "fpr_target": config.fpr_target,
            },
        }


def _simulate_panel(rng: np.random.Generator, p: int, n: int, mu: float) -> np.ndarray:
    base = rng.normal(size=(n, p))
    if mu <= 0.0:
        return base
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    factor = rng.normal(size=(n, 1))
    return base + mu * factor @ direction[np.newaxis, :]


def _estimate_detection_probability(
    cfg: OverlayConfig,
    rng: np.random.Generator,
    *,
    p: int,
    n: int,
    mu: float,
    trials: int,
) -> float:
    hits = 0
    for _ in range(trials):
        samples = _simulate_panel(rng, p, n, mu)
        covariance = np.cov(samples, rowvar=False, ddof=1)
        result = detect_spikes(covariance, samples=samples, config=cfg)
        if result.detections:
            hits += 1
    return float(hits) / float(trials)


def run_calibration(config: CalibrationConfig) -> CalibrationResult:
    rng = np.random.default_rng(config.seed)
    fpr: list[float] = []
    tpr: dict[str, list[float]] = {f"{mu:.2f}": [] for mu in config.mu}

    for margin in config.margin_grid:
        overlay_cfg = OverlayConfig(
            min_margin=margin,
            min_isolation=config.isolation,
            max_detections=1,
            shrinkage=0.0,
        )
        # False positive rate (mu = 0)
        fpr_val = _estimate_detection_probability(
            overlay_cfg,
            rng,
            p=config.p,
            n=config.n,
            mu=0.0,
            trials=config.trials,
        )
        fpr.append(fpr_val)

        for mu in config.mu:
            key = f"{mu:.2f}"
            tpr_val = _estimate_detection_probability(
                overlay_cfg,
                rng,
                p=config.p,
                n=config.n,
                mu=mu,
                trials=config.trials,
            )
            tpr[key].append(tpr_val)

    recommended_index = _select_recommended_margin(fpr, tpr, config)
    recommended_margin = config.margin_grid[recommended_index]
    recommended_fpr = fpr[recommended_index]
    recommended_tpr = {
        key: tpr_values[recommended_index] for key, tpr_values in tpr.items()
    }

    return CalibrationResult(
        margin_grid=config.margin_grid,
        fpr=fpr,
        tpr=tpr,
        recommended_margin=recommended_margin,
        recommended_fpr=recommended_fpr,
        recommended_tpr=recommended_tpr,
    )


def _select_recommended_margin(
    fpr: Iterable[float],
    tpr: dict[str, list[float]],
    config: CalibrationConfig,
) -> int:
    best_idx = None
    best_score = -1.0
    for idx, fpr_val in enumerate(fpr):
        if fpr_val > config.fpr_target:
            continue
        avg_tpr = float(np.mean([values[idx] for values in tpr.values()]))
        if avg_tpr > best_score:
            best_idx = idx
            best_score = avg_tpr
    if best_idx is not None:
        return best_idx
    # Fallback: minimise FPR
    return int(np.argmin(np.asarray(fpr)))


def save_results(result: CalibrationResult, config: CalibrationConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    json_path = config.out / "thresholds.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_json(config), handle, indent=2)

    for mu_key, tpr_values in result.tpr.items():
        csv_path = config.out / f"roc_mu{mu_key.replace('.', '_')}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["min_margin", "fpr", "tpr"])
            for margin, fpr_val, tpr_val in zip(result.margin_grid, result.fpr, tpr_values):
                writer.writerow([margin, fpr_val, tpr_val])
        _maybe_plot_roc(config.out, mu_key, result.margin_grid, result.fpr, tpr_values)


def _maybe_plot_roc(
    out_dir: Path,
    mu_key: str,
    margins: Iterable[float],
    fpr: Iterable[float],
    tpr: Iterable[float],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - plotting optional
        return

    fig, ax = plt.subplots()
    ax.plot(list(fpr), list(tpr), marker="o")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (mu={mu_key})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"roc_mu{mu_key.replace('.', '_')}.png"
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate de-alias detection thresholds.")
    parser.add_argument("--p", type=int, default=200, help="Dimensionality of the synthetic panel.")
    parser.add_argument("--n", type=int, default=252, help="Sample size per group.")
    parser.add_argument("--mu", type=str, default="4,6,8", help="Comma-separated signal strengths.")
    parser.add_argument("--trials", type=int, default=200, help="Number of Monte Carlo trials per grid point.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/calibration/",
        help="Output directory for ROC curves and threshold JSON.",
    )
    return parser.parse_args(argv)


def _parse_mu(mu_arg: str) -> tuple[float, ...]:
    values = []
    for chunk in mu_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("--mu must contain at least one numeric value.")
    return tuple(values)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = CalibrationConfig(
        p=args.p,
        n=args.n,
        mu=_parse_mu(args.mu),
        out=Path(args.out),
        trials=args.trials,
        seed=args.seed,
    )
    result = run_calibration(config)
    save_results(result, config)


if __name__ == "__main__":
    main()
