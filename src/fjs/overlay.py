"""
De-aliasing overlay estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .edge import EdgeConfig, EdgeEstimate, EdgeMode, compute_edge

__all__ = [
    "Detection",
    "Spectrum",
    "DetectionResult",
    "OverlayConfig",
    "detect_spikes",
    "apply_overlay",
]


@dataclass(slots=True, frozen=True)
class Detection:
    """Diagnostic bundle describing an accepted spike."""

    index: int
    eigenvalue: float
    margin: float
    isolation: float
    edge: float
    replacement: float
    score: float
    direction: NDArray[np.float64] = field(repr=False, compare=False)


@dataclass(slots=True, frozen=True)
class Spectrum:
    """Eigen decomposition of the covariance matrix (descending order)."""

    eigenvalues: NDArray[np.float64] = field(repr=False, compare=False)
    eigenvectors: NDArray[np.float64] = field(repr=False, compare=False)


@dataclass(slots=True, frozen=True)
class DetectionResult:
    """Container for detections and supporting diagnostics."""

    detections: tuple[Detection, ...]
    spectrum: Spectrum
    edge: EdgeEstimate


@dataclass(slots=True, frozen=True)
class OverlayConfig:
    """Configuration parameters for the overlay pipeline."""

    edge: EdgeConfig = field(default_factory=EdgeConfig)
    max_detections: int = 5
    min_margin: float = 0.05
    min_isolation: float = 0.05
    shrinkage: float = 0.05
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive.")
        if self.min_margin < 0.0:
            raise ValueError("min_margin must be non-negative.")
        if self.min_isolation < 0.0:
            raise ValueError("min_isolation must be non-negative.")
        if not (0.0 <= self.shrinkage <= 1.0):
            raise ValueError("shrinkage must lie in [0, 1].")


def _eigendecompose(covariance: NDArray[np.float64]) -> Spectrum:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    sorted_vals = eigvals[order]
    sorted_vecs = eigvecs[:, order]
    return Spectrum(eigenvalues=sorted_vals, eigenvectors=sorted_vecs)


def _edge_from_covariance(
    covariance: NDArray[np.float64],
    *,
    config: OverlayConfig,
) -> EdgeEstimate:
    diag = np.diag(covariance)
    finite = diag[np.isfinite(diag)]
    if finite.size == 0:
        noise = float(np.mean(np.linalg.eigvalsh(covariance)))
    else:
        noise = float(np.median(finite))
    if config.sample_count is None or config.sample_count <= 0:
        raise ValueError("sample_count must be provided when samples are omitted.")
    ratio = covariance.shape[0] / float(config.sample_count)
    raw_edge = noise * (1.0 + np.sqrt(max(ratio, 0.0))) ** 2
    buffered = raw_edge * (1.0 + config.edge.buffer_frac) + config.edge.buffer
    return EdgeEstimate(edge=float(buffered), raw_edge=float(raw_edge), noise_scale=float(noise), mode=EdgeMode.SCM)


def detect_spikes(
    covariance: NDArray[np.float64],
    *,
    samples: Iterable[Iterable[float]] | NDArray[np.float64] | None = None,
    config: OverlayConfig | None = None,
) -> DetectionResult:
    """
    Identify eigen-directions that warrant eigenvalue replacement.
    """

    cfg = config or OverlayConfig()
    spectrum = _eigendecompose(covariance)

    if samples is not None:
        edge = compute_edge(samples, config=cfg.edge)
    else:
        edge = _edge_from_covariance(covariance, config=cfg)

    detections: list[Detection] = []
    edge_value = max(edge.edge, 0.0)
    if edge_value == 0.0:
        return DetectionResult(detections=tuple(), spectrum=spectrum, edge=edge)

    eigenvalues = spectrum.eigenvalues
    for idx, eigenvalue in enumerate(eigenvalues):
        margin = (float(eigenvalue) - edge_value) / edge_value
        if margin < cfg.min_margin:
            break
        if idx + 1 < eigenvalues.shape[0]:
            next_val = float(eigenvalues[idx + 1])
            isolation = (float(eigenvalue) - next_val) / max(next_val, 1e-12)
        else:
            isolation = float("inf")
        if isolation < cfg.min_isolation:
            continue
        direction = spectrum.eigenvectors[:, idx]
        detection = Detection(
            index=idx,
            eigenvalue=float(eigenvalue),
            margin=float(margin),
            isolation=float(isolation),
            edge=edge_value,
            replacement=edge_value,
            score=float(margin * min(isolation, 10.0)),
            direction=direction,
        )
        detections.append(detection)
        if len(detections) >= cfg.max_detections:
            break

    return DetectionResult(detections=tuple(detections), spectrum=spectrum, edge=edge)


def _apply_shrinkage(
    eigenvalues: NDArray[np.float64],
    *,
    detections: Sequence[Detection],
    shrinkage: float,
    target: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    if shrinkage <= 0.0:
        return eigenvalues

    updated = eigenvalues.copy()
    mask = np.ones_like(updated, dtype=bool)
    for detection in detections:
        mask[detection.index] = False

    if np.isscalar(target):
        updated[mask] = (1.0 - shrinkage) * updated[mask] + shrinkage * float(target)
    else:
        target_arr = np.asarray(target, dtype=np.float64)
        if target_arr.shape != updated.shape:
            raise ValueError("shrinkage_target shape must match eigenvalues.")
        updated[mask] = (1.0 - shrinkage) * updated[mask] + shrinkage * target_arr[mask]
    return updated


def apply_overlay(
    covariance: NDArray[np.float64],
    result: DetectionResult,
    *,
    config: OverlayConfig | None = None,
    shrinkage_target: float | NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Apply eigenvalue substitutions for accepted detections while shrinking others.
    """

    cfg = config or OverlayConfig()
    eigenvalues = result.spectrum.eigenvalues.copy()
    eigenvectors = result.spectrum.eigenvectors

    for detection in result.detections:
        eigenvalues[detection.index] = detection.replacement

    target = shrinkage_target
    if target is None:
        target = result.edge.edge
    eigenvalues = _apply_shrinkage(eigenvalues, detections=result.detections, shrinkage=cfg.shrinkage, target=target)

    rebuilt = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    rebuilt = 0.5 * (rebuilt + rebuilt.T)
    return rebuilt
