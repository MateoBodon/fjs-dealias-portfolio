from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from experiments.synthetic_oneway.run import simulate_panel
from fjs.robust import edge_from_scatter, tyler_scatter

__all__ = [
    "HarnessConfig",
    "ScoreResult",
    "SimulatedScores",
    "simulate_scores",
    "roc_table",
    "select_energy_floor",
    "write_run_metadata",
]


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for synthetic null/power harness simulations."""

    n_assets: int = 40
    n_groups: int = 60
    replicates: int = 3
    noise_variance: float = 1.0
    signal_to_noise: float = 0.35
    edge_modes: Sequence[str] = ("scm", "tyler")
    trials: int = 400
    seed: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "n_assets": int(self.n_assets),
            "n_groups": int(self.n_groups),
            "replicates": int(self.replicates),
            "noise_variance": float(self.noise_variance),
            "signal_to_noise": float(self.signal_to_noise),
            "edge_modes": [str(mode) for mode in self.edge_modes],
            "trials": int(self.trials),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class ScoreResult:
    """Container for per-trial spectral scores."""

    scenario: str
    mu: float
    edge_mode: str
    trial: int
    score: float
    lambda_max: float
    mp_edge: float
    condition_number: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "scenario": str(self.scenario),
            "mu": float(self.mu),
            "edge_mode": str(self.edge_mode),
            "trial": int(self.trial),
            "score": float(self.score),
            "lambda_max": float(self.lambda_max),
            "mp_edge": float(self.mp_edge),
            "condition_number": float(self.condition_number),
        }


@dataclass(frozen=True)
class SimulatedScores:
    """Structured return for score simulations."""

    config: HarnessConfig
    scores: pd.DataFrame
    mu_values: Sequence[float]

    def filter_mode(self, edge_mode: str) -> pd.DataFrame:
        mask = self.scores["edge_mode"] == edge_mode
        return self.scores.loc[mask].copy()


def _compute_scatter(observations: NDArray[np.float64], edge_mode: str) -> NDArray[np.float64]:
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("observations must be a 2D array.")
    if edge_mode == "scm":
        scatter = np.cov(y, rowvar=False, ddof=1)
    elif edge_mode == "tyler":
        scatter = tyler_scatter(y)
    else:
        raise ValueError(f"Unsupported edge mode '{edge_mode}'.")
    scatter = np.asarray(scatter, dtype=np.float64)
    scatter = 0.5 * (scatter + scatter.T)
    return scatter


def _score_trial(
    observations: NDArray[np.float64],
    edge_mode: str,
) -> tuple[float, float, float, float]:
    scatter = _compute_scatter(observations, edge_mode)
    eigvals = np.linalg.eigvalsh(scatter)
    if eigvals.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    lambda_max = float(eigvals[-1])
    mp_edge = edge_from_scatter(scatter, scatter.shape[0], observations.shape[0])
    score = max(lambda_max - mp_edge, 0.0)
    cond = float(np.linalg.cond(scatter)) if scatter.size else 0.0
    return score, lambda_max, cond, mp_edge


def _run_single_mu(
    config: HarnessConfig,
    mu: float,
    *,
    scenario_label: str,
) -> list[ScoreResult]:
    label_seed = int.from_bytes(scenario_label.encode("utf-8"), "little", signed=False) & 0xFFFFFFFF
    mu_seed = int(round(mu * 1000.0)) & 0xFFFFFFFF
    base_seed = (int(config.seed) & 0xFFFFFFFF) ^ label_seed ^ mu_seed
    rng = np.random.default_rng(base_seed)
    results: list[ScoreResult] = []
    for trial in range(config.trials):
        local_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
        observations, _ = simulate_panel(
            local_rng,
            n_assets=config.n_assets,
            n_groups=config.n_groups,
            replicates=config.replicates,
            spike_strength=float(mu),
            noise_variance=config.noise_variance,
            signal_to_noise=config.signal_to_noise,
        )
        for edge_mode in config.edge_modes:
            try:
                score, lambda_max, cond, mp_edge = _score_trial(observations, edge_mode)
            except Exception:
                score, lambda_max, cond, mp_edge = float("nan"), float("nan"), float("nan"), float("nan")
            results.append(
                ScoreResult(
                    scenario=scenario_label,
                    mu=float(mu),
                    edge_mode=edge_mode,
                    trial=trial,
                    score=float(score),
                    lambda_max=float(lambda_max),
                    mp_edge=float(mp_edge),
                    condition_number=float(cond),
                )
            )
    return results


def simulate_scores(
    config: HarnessConfig,
    mu_values: Sequence[float],
    *,
    scenario_prefix: str = "",
) -> SimulatedScores:
    """Simulate score distributions for the supplied spike strengths."""

    records: list[dict[str, float | int | str]] = []
    prefix = (scenario_prefix or "").strip()
    for mu in mu_values:
        label = f"{prefix}{mu}" if prefix else str(mu)
        trials = _run_single_mu(config, float(mu), scenario_label=label)
        records.extend(result.to_dict() for result in trials)
    frame = pd.DataFrame.from_records(records)
    return SimulatedScores(config=config, scores=frame, mu_values=tuple(float(mu) for mu in mu_values))


def roc_table(
    null_scores: pd.DataFrame,
    power_scores: Mapping[float, pd.DataFrame],
    *,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return a ROC-style table (FPR vs power) per edge mode and spike."""

    modes = sorted(map(str, null_scores["edge_mode"].unique()))
    if thresholds is None:
        combined = [float(val) for val in null_scores["score"].tolist()]
        for df in power_scores.values():
            combined.extend(float(val) for val in df["score"].tolist())
        thresholds = np.linspace(0.0, max(combined + [0.0]), num=101, dtype=np.float64)

    rows: list[dict[str, float | str]] = []
    threshold_array = np.asarray(list(thresholds), dtype=np.float64)
    for mode in modes:
        null_mode = null_scores[null_scores["edge_mode"] == mode]
        null_values = np.nan_to_num(null_mode["score"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if null_values.size == 0:
            continue
        for threshold in threshold_array:
            fpr = float(np.mean(null_values >= threshold))
            for mu, df in power_scores.items():
                power_mode = df[df["edge_mode"] == mode]
                vals = np.nan_to_num(power_mode["score"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                if vals.size == 0:
                    continue
                tpr = float(np.mean(vals >= threshold))
                rows.append(
                    {
                        "edge_mode": mode,
                        "threshold": float(threshold),
                        "fpr": fpr,
                        "mu": float(mu),
                        "tpr": tpr,
                    }
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EnergyFloorSelection:
    edge_mode: str
    threshold: float
    fpr: float
    tpr_by_mu: Mapping[float, float]
    average_power: float

    def to_json(self) -> dict[str, object]:
        return {
            "edge_mode": self.edge_mode,
            "threshold": float(self.threshold),
            "fpr": float(self.fpr),
            "average_power": float(self.average_power),
            "tpr_by_mu": {f"{mu:.2f}": float(val) for mu, val in self.tpr_by_mu.items()},
        }


def select_energy_floor(
    null_scores: pd.DataFrame,
    power_scores: Mapping[float, pd.DataFrame],
    *,
    target_fpr: float,
) -> EnergyFloorSelection | None:
    """Select an energy floor that satisfies the FPR cap while maximising power."""

    if not power_scores:
        return None

    best: EnergyFloorSelection | None = None
    for mode in sorted(null_scores["edge_mode"].unique()):
        null_mode_scores = null_scores[null_scores["edge_mode"] == mode]["score"].to_numpy(dtype=np.float64)
        if null_mode_scores.size == 0:
            continue
        thresholds = np.unique(np.linspace(0.0, np.nanmax(null_mode_scores), num=128, dtype=np.float64))
        for df in power_scores.values():
            mode_scores = df[df["edge_mode"] == mode]["score"].to_numpy(dtype=np.float64)
            thresholds = np.unique(np.concatenate([thresholds, mode_scores]))

        thresholds = np.asarray(sorted(set(float(x) for x in thresholds if np.isfinite(x))), dtype=np.float64)
        if thresholds.size == 0:
            continue

        for threshold in thresholds:
            fpr = float(np.mean(null_mode_scores >= threshold))
            tpr_map: dict[float, float] = {}
            for mu, df in power_scores.items():
                mode_scores = df[df["edge_mode"] == mode]["score"].to_numpy(dtype=np.float64)
                if mode_scores.size == 0:
                    continue
                tpr_map[float(mu)] = float(np.mean(mode_scores >= threshold))
            if not tpr_map:
                continue
            average_power = float(np.mean(list(tpr_map.values())))
            candidate = EnergyFloorSelection(
                edge_mode=str(mode),
                threshold=float(threshold),
                fpr=float(fpr),
                tpr_by_mu=tpr_map,
                average_power=average_power,
            )
            if fpr <= target_fpr:
                if (
                    best is None
                    or best.fpr > target_fpr
                    or candidate.average_power > best.average_power
                    or (
                        np.isclose(candidate.average_power, best.average_power)
                        and candidate.threshold < best.threshold
                    )
                ):
                    best = candidate
            elif best is None or best.fpr > target_fpr:
                best = candidate
    return best


def write_run_metadata(
    path: Path,
    *,
    config: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> None:
    payload = dict(config)
    payload.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
