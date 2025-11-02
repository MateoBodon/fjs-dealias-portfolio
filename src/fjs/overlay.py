from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from finance.ledoit import lw_cov
from finance.shrinkage import oas_covariance
from fjs.dealias import Detection, dealias_search

__all__ = [
    "OverlayConfig",
    "detect_spikes",
    "apply_overlay",
]


@dataclass(slots=True)
class OverlayConfig:
    shrinker: str = "rie"
    sample_count: int | None = None
    max_detections: int | None = None
    q_max: int | None = 1
    delta: float = 0.5
    eps: float = 0.02
    a_grid: int = 120
    require_isolated: bool = True
    off_component_cap: float | None = 0.3
    min_edge_margin: float = 0.0
    edge_mode: str = "tyler"
    seed: int = 0


def _rie_covariance(
    sample_covariance: NDArray[np.float64],
    *,
    sample_count: int | None,
) -> NDArray[np.float64]:
    sigma = np.asarray(sample_covariance, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    p = eigvals.size
    if p == 0:
        return sigma.copy()
    if sample_count is None or sample_count <= 0:
        shrinkage = 0.5
    else:
        shrinkage = min(0.99, max(0.0, float(p) / float(sample_count)))
    bulk_mean = float(np.mean(eigvals))
    shrunk = (1.0 - shrinkage) * eigvals + shrinkage * bulk_mean
    adjusted = eigvecs @ np.diag(shrunk) @ eigvecs.T
    return np.asarray(0.5 * (adjusted + adjusted.T), dtype=np.float64)


def _baseline_covariance(
    sample_covariance: NDArray[np.float64],
    *,
    observations: NDArray[np.float64] | None,
    config: OverlayConfig,
) -> NDArray[np.float64]:
    shrinker = (config.shrinker or "rie").strip().lower()
    if shrinker == "lw":
        if observations is None:
            raise ValueError("observations required for Ledoit–Wolf shrinkage.")
        return np.asarray(lw_cov(observations), dtype=np.float64)
    if shrinker == "oas":
        if observations is None:
            raise ValueError("observations required for OAS shrinkage.")
        return np.asarray(oas_covariance(observations), dtype=np.float64)
    if shrinker == "sample":
        sigma = np.asarray(sample_covariance, dtype=np.float64)
        return np.asarray(0.5 * (sigma + sigma.T), dtype=np.float64)
    # Default RIE-style shrinkage
    return _rie_covariance(
        np.asarray(sample_covariance, dtype=np.float64),
        sample_count=config.sample_count,
    )


def detect_spikes(
    observations: NDArray[np.float64],
    groups: Sequence[int],
    *,
    config: OverlayConfig | None = None,
    stats: dict | None = None,
) -> list[Detection]:
    cfg = config or OverlayConfig()
    _ = np.random.default_rng(cfg.seed)  # ensure deterministic rng initialisation
    detections = dealias_search(
        np.asarray(observations, dtype=np.float64),
        np.asarray(groups, dtype=np.intp),
        target_r=0,
        delta=float(cfg.delta),
        eps=float(cfg.eps),
        a_grid=int(cfg.a_grid),
        use_tvector=bool(cfg.require_isolated),
        off_component_leak_cap=cfg.off_component_cap,
        edge_mode=str(cfg.edge_mode),
        stats=stats,
    )
    filtered: list[Detection] = []
    for det in detections:
        margin = det.get("edge_margin")
        if margin is None:
            continue
        if float(margin) < float(cfg.min_edge_margin):
            continue
        filtered.append(det)

    filtered.sort(key=lambda det: float(det.get("edge_margin") or det["mu_hat"]), reverse=True)
    limit_q = cfg.q_max if cfg.q_max is not None else len(filtered)
    limit_m = cfg.max_detections if cfg.max_detections is not None else len(filtered)
    cap = min(limit_q, limit_m)
    if filtered and cap < len(filtered):
        filtered = filtered[: int(cap)]
    return filtered


def apply_overlay(
    sample_covariance: NDArray[np.float64],
    detections: Iterable[Detection],
    *,
    observations: NDArray[np.float64] | None = None,
    config: OverlayConfig | None = None,
    baseline_covariance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    cfg = config or OverlayConfig()
    if baseline_covariance is None:
        base = _baseline_covariance(sample_covariance, observations=observations, config=cfg)
    else:
        base = np.asarray(baseline_covariance, dtype=np.float64)
    overlay = np.asarray(base, dtype=np.float64)
    overlay = 0.5 * (overlay + overlay.T)

    max_use = cfg.max_detections if cfg.max_detections is not None else cfg.q_max
    applied = 0
    for det in detections:
        if max_use is not None and applied >= int(max_use):
            break
        vec = np.asarray(det["eigvec"], dtype=np.float64).reshape(-1, 1)
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            continue
        vec /= norm
        mu = float(det["mu_hat"])
        if not np.isfinite(mu) or mu <= 0.0:
            continue
        current = float((vec.T @ overlay @ vec)[0, 0])
        overlay = overlay + (mu - current) * (vec @ vec.T)
        applied += 1

    overlay = np.asarray(0.5 * (overlay + overlay.T), dtype=np.float64)
    try:
        eigvals = np.linalg.eigvalsh(overlay)
    except np.linalg.LinAlgError:
        ridge = 1e-6 * np.eye(overlay.shape[0], dtype=np.float64)
        overlay = np.asarray(0.5 * (overlay + overlay.T), dtype=np.float64) + ridge
        try:
            eigvals = np.linalg.eigvalsh(overlay)
        except np.linalg.LinAlgError:
            return np.asarray(base, dtype=np.float64)
    min_eig = float(eigvals.min(initial=0.0)) if eigvals.size else 0.0
    if min_eig < 0.0:
        overlay = overlay + (-min_eig + 1e-8) * np.eye(overlay.shape[0], dtype=np.float64)
    return overlay
