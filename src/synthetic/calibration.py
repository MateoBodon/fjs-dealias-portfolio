from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from fjs.balanced import mean_squares
from synthetic.threshold_eval import evaluate_threshold_grid

__all__ = [
    "CalibrationConfig",
    "CalibrationResult",
    "ThresholdEntry",
    "GridStat",
    "calibrate_thresholds",
    "write_thresholds",
]


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for synthetic calibration of overlay thresholds."""

    p_assets: int = 40
    n_groups: int = 180
    replicates: int = 3
    alpha: float = 0.02
    trials_null: int = 60
    trials_alt: int = 60
    delta_abs: float = 0.5
    eps: float = 0.02
    delta_frac_grid: Sequence[float] = (0.0, 0.015, 0.02, 0.03)
    stability_grid: Sequence[float] = (0.3, 0.4, 0.5)
    spike_strength: float = 4.0
    edge_modes: Sequence[str] = ("scm", "tyler")
    q_max: int = 2
    seed: int = 0
    workers: int | None = None


@dataclass(frozen=True)
class ThresholdEntry:
    delta_frac: float
    stability_eta_deg: float
    fpr: float
    power: float | None
    trials_null: int
    trials_alt: int

    def to_dict(self) -> dict[str, float | int | None]:
        payload: dict[str, float | int | None] = {
            "delta_frac": float(self.delta_frac),
            "stability_eta_deg": float(self.stability_eta_deg),
            "fpr": float(self.fpr),
            "trials_null": int(self.trials_null),
        }
        if self.power is not None:
            payload["power"] = float(self.power)
        payload["trials_alt"] = int(self.trials_alt)
        return payload


@dataclass(frozen=True)
class GridStat:
    edge_mode: str
    delta_abs: float
    delta_frac: float
    stability_eta_deg: float
    fpr: float
    power: float | None

    def to_dict(self) -> dict[str, float | str | None]:
        payload: dict[str, float | str | None] = {
            "edge_mode": str(self.edge_mode),
            "delta": float(self.delta_abs),
            "delta_frac": float(self.delta_frac),
            "stability_eta_deg": float(self.stability_eta_deg),
            "fpr": float(self.fpr),
        }
        if self.power is not None:
            payload["power"] = float(self.power)
        else:
            payload["power"] = None
        return payload


@dataclass(frozen=True)
class CalibrationResult:
    config: CalibrationConfig
    thresholds: Mapping[str, ThresholdEntry]
    alpha: float
    delta_abs: float
    generated_at: datetime
    grid: Sequence[GridStat]

    def to_json(self) -> dict[str, object]:
        return {
            "alpha": float(self.alpha),
            "delta_abs": float(self.delta_abs),
            "generated_at": self.generated_at.isoformat(),
            "p_assets": int(self.config.p_assets),
            "n_groups": int(self.config.n_groups),
            "replicates": int(self.config.replicates),
            "trials_null": int(self.config.trials_null),
            "trials_alt": int(self.config.trials_alt),
            "delta_frac_grid": [float(val) for val in self.config.delta_frac_grid],
            "stability_grid": [float(val) for val in self.config.stability_grid],
            "edge_modes": [str(mode) for mode in self.config.edge_modes],
            "thresholds": {
                str(mode): entry.to_dict() for mode, entry in self.thresholds.items()
            },
            "grid": [stat.to_dict() for stat in self.grid],
        }


def _simulate_panel(
    config: CalibrationConfig,
    rng: np.random.Generator,
    *,
    spike_strength: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    total_obs = config.n_groups * config.replicates
    noise = rng.normal(scale=1.0, size=(total_obs, config.p_assets))
    if spike_strength > 0.0:
        direction = rng.normal(size=config.p_assets)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            direction = np.zeros(config.p_assets, dtype=np.float64)
            direction[0] = 1.0
        else:
            direction = direction / norm
        group_signal = rng.normal(scale=spike_strength, size=config.n_groups)
        signal = np.repeat(group_signal, config.replicates)[:, None] * direction[None, :]
        noise = noise + signal
    groups = np.repeat(np.arange(config.n_groups), config.replicates).astype(np.intp)
    return noise.astype(np.float64), groups


def _select_entry(
    candidates: Sequence[tuple[float, float, float, float | None]],
    alpha: float,
) -> tuple[float, float, float, float | None]:
    feasible = [entry for entry in candidates if entry[2] <= alpha]
    if feasible:
        feasible = sorted(
            feasible,
            key=lambda item: (
                float(item[0]),
                float(item[1]),
                -float(item[3]) if item[3] is not None else 0.0,
            ),
        )
        return feasible[0]
    return min(candidates, key=lambda item: item[2])


def calibrate_thresholds(config: CalibrationConfig) -> CalibrationResult:
    rng = np.random.default_rng(config.seed)
    null_seeds = rng.integers(0, 2**63 - 1, size=config.trials_null, dtype=np.int64)
    alt_seeds = (
        rng.integers(0, 2**63 - 1, size=config.trials_alt, dtype=np.int64)
        if config.trials_alt > 0
        else np.array([], dtype=np.int64)
    )

    edge_modes = [str(mode) for mode in config.edge_modes]
    delta_frac_values = np.asarray(list(config.delta_frac_grid), dtype=np.float64)
    stability_values = np.asarray(list(config.stability_grid), dtype=np.float64)

    null_counts = {
        mode: np.zeros((delta_frac_values.size, stability_values.size), dtype=np.int64)
        for mode in edge_modes
    }
    alt_counts = {
        mode: np.zeros((delta_frac_values.size, stability_values.size), dtype=np.int64)
        for mode in edge_modes
    }

    def _accumulate(seed_list: np.ndarray, spike_strength: float, target: dict[str, np.ndarray]) -> None:
        for seed in seed_list:
            trial_rng = np.random.default_rng(int(seed))
            observations, groups = _simulate_panel(
                config,
                trial_rng,
                spike_strength=spike_strength,
            )
            stats = mean_squares(observations, groups)
            evaluations = evaluate_threshold_grid(
                observations,
                groups,
                delta_abs=float(config.delta_abs),
                eps=float(config.eps),
                edge_modes=edge_modes,
                delta_frac_values=delta_frac_values,
                stability_values=stability_values,
                q_max=int(config.q_max or 1),
                a_grid=120,
                require_isolated=True,
                alignment_min=0.0,
                stats=stats,
            )
            for mode in edge_modes:
                target[mode] += evaluations[mode].astype(np.int64)

    if null_seeds.size:
        _accumulate(null_seeds, 0.0, null_counts)
    if alt_seeds.size:
        _accumulate(alt_seeds, float(config.spike_strength), alt_counts)

    grid_stats: list[GridStat] = []
    thresholds: dict[str, ThresholdEntry] = {}

    for mode in edge_modes:
        candidates: list[tuple[float, float, float, float | None]] = []
        fpr_matrix = null_counts[mode].astype(np.float64) / max(1, int(config.trials_null))
        power_matrix = (
            alt_counts[mode].astype(np.float64) / max(1, int(config.trials_alt))
            if config.trials_alt > 0
            else np.zeros_like(fpr_matrix)
        )
        for stability_idx, stability_eta in enumerate(stability_values):
            for delta_idx, delta_frac in enumerate(delta_frac_values):
                fpr = float(fpr_matrix[delta_idx, stability_idx])
                power = (
                    float(power_matrix[delta_idx, stability_idx])
                    if config.trials_alt > 0
                    else None
                )
                grid_stats.append(
                    GridStat(
                        edge_mode=str(mode),
                        delta_abs=float(config.delta_abs),
                        delta_frac=float(delta_frac),
                        stability_eta_deg=float(stability_eta),
                        fpr=fpr,
                        power=power,
                    )
                )
                candidates.append((float(delta_frac), float(stability_eta), fpr, power))

        best_delta, best_stability, best_fpr, best_power = _select_entry(candidates, config.alpha)
        thresholds[str(mode)] = ThresholdEntry(
            delta_frac=best_delta,
            stability_eta_deg=best_stability,
            fpr=best_fpr,
            power=best_power,
            trials_null=config.trials_null,
            trials_alt=config.trials_alt,
        )

    return CalibrationResult(
        config=config,
        thresholds=thresholds,
        alpha=config.alpha,
        delta_abs=config.delta_abs,
        generated_at=datetime.now(timezone.utc),
        grid=tuple(grid_stats),
    )


def write_thresholds(result: CalibrationResult, path: str | Path) -> Path:
    payload = result.to_json()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
