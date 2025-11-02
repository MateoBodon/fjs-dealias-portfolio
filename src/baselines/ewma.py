"""
Exponentially weighted covariance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["EWMAConfig", "ewma_covariance"]


@dataclass(slots=True, frozen=True)
class EWMAConfig:
    """Configuration for an exponentially weighted covariance estimate."""

    lambda_: float = 0.94
    debias: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.lambda_ < 1.0):
            raise ValueError("lambda_ must lie strictly between 0 and 1.")


def _ensure_matrix(observations: Iterable[Iterable[float]] | NDArray[np.float64]) -> NDArray[np.float64]:
    data = np.asarray(observations, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("observations must be a two-dimensional array.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("observations must contain at least one sample and one feature.")
    return data


def _effective_weights(n_samples: int, lambda_: float, debias: bool) -> NDArray[np.float64]:
    powers = np.arange(n_samples - 1, -1, -1, dtype=np.float64)
    weights = (1.0 - lambda_) * np.power(lambda_, powers)
    normaliser = weights.sum()
    if debias:
        debias_norm = 1.0 - lambda_ ** n_samples
        if debias_norm > 0.0:
            normaliser = debias_norm
    if normaliser <= 0.0:
        normaliser = 1.0
    weights /= normaliser
    return weights


def ewma_covariance(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    *,
    config: EWMAConfig | None = None,
) -> NDArray[np.float64]:
    """
    Compute an exponentially weighted covariance matrix with optional debiasing.
    """

    cfg = config or EWMAConfig()
    data = _ensure_matrix(observations)
    n_samples, _ = data.shape

    weights = _effective_weights(n_samples, cfg.lambda_, cfg.debias)
    weights = weights.reshape(-1, 1)
    weighted_mean = np.sum(weights * data, axis=0, keepdims=True)
    demeaned = data - weighted_mean

    cov = (demeaned.T * weights.ravel()) @ demeaned
    cov = 0.5 * (cov + cov.T)
    return cov
