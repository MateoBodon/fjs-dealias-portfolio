"""
Robust Marčenko–Pastur edge estimation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .robust import edge_from_scatter, huber_scatter, tyler_scatter

__all__ = [
    "EdgeMode",
    "EdgeConfig",
    "EdgeEstimate",
    "compute_edge",
]


class EdgeMode(str, Enum):
    """Supported edge estimation back-ends."""

    SCM = "scm"
    TYLER = "tyler"
    HUBER = "huber"


@dataclass(slots=True, frozen=True)
class EdgeConfig:
    """Configuration parameters for edge estimation."""

    mode: EdgeMode = EdgeMode.TYLER
    huber_c: float = 2.5
    buffer: float = 0.0
    buffer_frac: float = 0.05

    def __post_init__(self) -> None:
        if self.mode is EdgeMode.HUBER and self.huber_c <= 0.0:
            raise ValueError("Huber threshold must be positive.")
        if self.buffer_frac < 0.0:
            raise ValueError("buffer_frac must be non-negative.")
        if self.buffer < 0.0:
            raise ValueError("buffer must be non-negative.")


@dataclass(slots=True, frozen=True)
class EdgeEstimate:
    """Container for an estimated edge and accompanying diagnostics."""

    edge: float
    raw_edge: float
    noise_scale: float
    mode: EdgeMode = field(compare=False)


def _ensure_2d(observations: Iterable[Iterable[float]] | NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(observations, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("observations must be a two-dimensional array.")
    if array.shape[0] <= 1 or array.shape[1] == 0:
        raise ValueError("observations must contain at least two samples and one feature.")
    return array


def _scatter_from_mode(
    data: NDArray[np.float64],
    *,
    config: EdgeConfig,
) -> NDArray[np.float64]:
    if config.mode is EdgeMode.TYLER:
        return tyler_scatter(data)
    if config.mode is EdgeMode.HUBER:
        return huber_scatter(data, config.huber_c)
    # SCM fallback
    demeaned = data - np.mean(data, axis=0, keepdims=True)
    return np.cov(demeaned, rowvar=False, ddof=1)


def _noise_scale(scatter: NDArray[np.float64]) -> float:
    diag = np.diag(scatter)
    finite = diag[np.isfinite(diag)]
    if finite.size == 0:
        eigvals = np.linalg.eigvalsh(scatter)
        return float(np.mean(eigvals))
    return float(np.median(finite))


def compute_edge(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    *,
    config: EdgeConfig | None = None,
) -> EdgeEstimate:
    """
    Estimate the (buffered) upper Marčenko–Pastur edge for the supplied samples.
    """

    cfg = config or EdgeConfig()
    data = _ensure_2d(observations)
    n_samples, n_features = data.shape
    scatter = _scatter_from_mode(data, config=cfg)
    raw_edge = edge_from_scatter(scatter, n_features, n_samples)
    noise_scale = _noise_scale(scatter)
    buffered = raw_edge * (1.0 + cfg.buffer_frac) + cfg.buffer
    return EdgeEstimate(edge=float(buffered), raw_edge=float(raw_edge), noise_scale=float(noise_scale), mode=cfg.mode)
