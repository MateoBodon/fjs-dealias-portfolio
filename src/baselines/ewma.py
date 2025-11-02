"""
Exponentially weighted covariance estimates (EWMA) scaffolding.

The final implementation will provide tuned lambda schedules and diagnostics.
This stub keeps the function signatures stable for early testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["EWMAConfig", "ewma_covariance"]


@dataclass(frozen=True)
class EWMAConfig:
    """Configuration for an exponentially weighted covariance estimate."""

    lambda_: float = 0.94
    debias: bool = True


def ewma_covariance(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    *,
    config: EWMAConfig | None = None,
) -> NDArray[np.float64]:
    """
    Placeholder entry point for an EWMA covariance matrix.

    Parameters
    ----------
    observations
        Sample matrix shaped ``(n_samples, n_features)``.
    config
        Optional EWMA configuration; defaults align with industry convention.

    Returns
    -------
    ndarray
        The covariance matrix produced by the EWMA smoother (to be implemented).
    """

    raise NotImplementedError("EWMA covariance will be implemented in Sprint 2.")
