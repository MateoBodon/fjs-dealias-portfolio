"""
Random matrix (RIE / QuEST-inspired) covariance cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["RIEConfig", "rie_covariance"]


@dataclass(slots=True, frozen=True)
class RIEConfig:
    """Configuration for the random matrix cleaner."""

    min_eigenvalue: float = 1e-6

    def __post_init__(self) -> None:
        if self.min_eigenvalue <= 0.0:
            raise ValueError("min_eigenvalue must be positive.")


def _ensure_matrix(observations: Iterable[Iterable[float]] | NDArray[np.float64]) -> NDArray[np.float64]:
    data = np.asarray(observations, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("observations must be a two-dimensional array.")
    if data.shape[0] <= 1 or data.shape[1] == 0:
        raise ValueError("observations must contain at least two samples and one feature.")
    return data


def _mp_upper_edge(eigenvalues: NDArray[np.float64], *, n_samples: int, n_features: int) -> float:
    sigma2 = float(np.mean(eigenvalues))
    sigma2 = max(sigma2, 0.0)
    if sigma2 == 0.0:
        return 0.0
    gamma = float(n_features) / float(n_samples)
    gamma = max(gamma, 0.0)
    return sigma2 * (1.0 + np.sqrt(gamma)) ** 2


def rie_covariance(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    *,
    config: RIEConfig | None = None,
) -> NDArray[np.float64]:
    """
    Clean a sample covariance matrix by clipping bulk eigenvalues to their mean.
    """

    cfg = config or RIEConfig()
    data = _ensure_matrix(observations)
    n_samples, n_features = data.shape
    demeaned = data - np.mean(data, axis=0, keepdims=True)
    sample_cov = np.cov(demeaned, rowvar=False, ddof=1)

    eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
    lambda_plus = _mp_upper_edge(eigenvalues, n_samples=n_samples, n_features=n_features)

    cleaned = eigenvalues.copy()
    if lambda_plus > 0.0:
        bulk_mask = cleaned <= lambda_plus
        if np.any(bulk_mask):
            bulk_mean = float(np.mean(cleaned[bulk_mask]))
            cleaned[bulk_mask] = bulk_mean
    cleaned = np.clip(cleaned, cfg.min_eigenvalue, None)

    rebuilt = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    rebuilt = 0.5 * (rebuilt + rebuilt.T)
    return rebuilt
