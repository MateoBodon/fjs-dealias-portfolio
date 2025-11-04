from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from fjs.overlay import OverlayConfig, detect_spikes

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


def _evaluate_detection(
    config: CalibrationConfig,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    edge_mode: str,
    delta_frac: float,
    stability_eta: float,
    seed: int,
) -> bool:
    overlay_cfg = OverlayConfig(
        shrinker="rie",
        delta=config.delta_abs,
        delta_frac=delta_frac,
        eps=config.eps,
        stability_eta_deg=stability_eta,
        q_max=config.q_max,
        max_detections=config.q_max,
        edge_mode=edge_mode,
        seed=seed,
        require_isolated=True,
        a_grid=120,
    )
    try:
        detections = detect_spikes(y, groups, config=overlay_cfg)
    except Exception:
        return True
    return bool(detections)


def _trial_worker(
    args: tuple[int, int, CalibrationConfig, str, float, float, float]
) -> int:
    idx, seed_val, cfg, edge_mode, delta_frac, stability_eta, spike_strength = args
    rng = np.random.default_rng(int(seed_val))
    y, groups = _simulate_panel(cfg, rng, spike_strength=spike_strength)
    fired = _evaluate_detection(
        cfg,
        y,
        groups,
        edge_mode=edge_mode,
        delta_frac=delta_frac,
        stability_eta=stability_eta,
        seed=int(seed_val) + idx,
    )
    return 1 if fired else 0


def _estimate_rate(
    config: CalibrationConfig,
    *,
    edge_mode: str,
    delta_frac: float,
    stability_eta: float,
    seeds: Iterable[int],
    spike_strength: float,
) -> float:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        return 0.0

    trials = [
        (idx, seed_val, config, edge_mode, delta_frac, stability_eta, spike_strength)
        for idx, seed_val in enumerate(seed_list)
    ]

    workers = max(1, int(config.workers)) if config.workers else 1
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            triggered = sum(executor.map(_trial_worker, trials))
    else:
        triggered = sum(_trial_worker(trial) for trial in trials)

    total = float(len(seed_list))
    return float(triggered) / total if total > 0 else 0.0


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

    thresholds: dict[str, ThresholdEntry] = {}
    grid_stats: list[GridStat] = []
    for mode in config.edge_modes:
        candidates: list[tuple[float, float, float, float | None]] = []
        for stability_eta in config.stability_grid:
            for delta_frac in config.delta_frac_grid:
                fpr = _estimate_rate(
                    config,
                    edge_mode=mode,
                    delta_frac=float(delta_frac),
                    stability_eta=float(stability_eta),
                    seeds=null_seeds,
                    spike_strength=0.0,
                )
                power = None
                if config.trials_alt > 0:
                    power = _estimate_rate(
                        config,
                        edge_mode=mode,
                        delta_frac=float(delta_frac),
                        stability_eta=float(stability_eta),
                        seeds=alt_seeds,
                        spike_strength=config.spike_strength,
                    )
                candidates.append((float(delta_frac), float(stability_eta), fpr, power))
                grid_stats.append(
                    GridStat(
                        edge_mode=str(mode),
                        delta_abs=float(config.delta_abs),
                        delta_frac=float(delta_frac),
                        stability_eta_deg=float(stability_eta),
                        fpr=float(fpr),
                        power=float(power) if power is not None else None,
                    )
                )
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
