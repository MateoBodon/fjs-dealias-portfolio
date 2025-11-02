"""
De-aliasing overlay scaffolding.

Sprint 1 focuses on wiring the detection pipeline and eigenvalue substitution.
This stub exposes the objects and function signatures required for downstream
development and testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .edge import EdgeEstimate, EdgeConfig

__all__ = [
    "Detection",
    "OverlayConfig",
    "detect_spikes",
    "apply_overlay",
]


@dataclass(frozen=True)
class Detection:
    """Diagnostic bundle describing an accepted spike."""

    index: int
    eigenvalue: float
    margin: float
    score: float
    direction: NDArray[np.float64]


@dataclass(frozen=True)
class OverlayConfig:
    """Configuration parameters for the overlay pipeline."""

    edge: EdgeConfig | None = None
    max_detections: int = 10
    shrinkage: float = 0.05


def detect_spikes(
    covariance: NDArray[np.float64],
    *,
    samples: Iterable[Iterable[float]] | NDArray[np.float64] | None = None,
    config: OverlayConfig | None = None,
) -> list[Detection]:
    """
    Identify eigen-directions that warrant eigenvalue replacement.
    """

    raise NotImplementedError("Spike detection will be implemented in Sprint 1.")


def apply_overlay(
    covariance: NDArray[np.float64],
    eigvecs: NDArray[np.float64],
    detections: Sequence[Detection],
    *,
    shrinkage_target: float | NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Apply eigenvalue substitutions for accepted detections while shrinking others.
    """

    raise NotImplementedError("Overlay application will be implemented in Sprint 1.")
