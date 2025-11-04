from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from finance.ledoit import lw_cov
from finance.shrinkage import cc_covariance as _cc_covariance
from finance.shrinkage import oas_covariance as _oas_covariance

__all__ = [
    "sample_covariance",
    "lw_covariance",
    "oas_covariance",
    "cc_covariance",
    "ewma_covariance",
    "quest_covariance",
    "rie_covariance",
]


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def sample_covariance(observations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sample covariance with Bessel correction."""

    data = np.asarray(observations, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("observations must be two-dimensional (samples × assets).")
    if data.shape[0] <= 1:
        raise ValueError("At least two observations required for sample covariance.")
    centred = data - data.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / float(data.shape[0] - 1)
    return _symmetrize(cov)


def lw_covariance(observations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ledoit–Wolf shrinkage covariance estimator."""

    return _symmetrize(np.asarray(lw_cov(np.asarray(observations, dtype=np.float64)), dtype=np.float64))


def oas_covariance(observations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Oracle Approximating Shrinkage (OAS) estimator."""

    return _symmetrize(_oas_covariance(np.asarray(observations, dtype=np.float64)))


def cc_covariance(observations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ledoit–Wolf constant-correlation shrinkage estimator."""

    return _symmetrize(_cc_covariance(np.asarray(observations, dtype=np.float64)))


def rie_covariance(
    sample_covariance: NDArray[np.float64],
    *,
    sample_count: int | None = None,
) -> NDArray[np.float64]:
    """Rotationally-invariant estimator (RIE) shrinkage towards the spectrum mean."""

    sigma = np.asarray(sample_covariance, dtype=np.float64)
    sigma = _symmetrize(sigma)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    p = eigvals.size
    if p == 0:
        return sigma.copy()
    if sample_count is None or sample_count <= 0:
        shrinkage = 0.5
    else:
        shrinkage = float(p) / float(sample_count)
        shrinkage = min(0.99, max(0.0, shrinkage))
    bulk_mean = float(np.mean(eigvals))
    shrunk = (1.0 - shrinkage) * eigvals + shrinkage * bulk_mean
    adjusted = eigvecs @ np.diag(shrunk) @ eigvecs.T
    return _symmetrize(adjusted)


def quest_covariance(
    sample_covariance: NDArray[np.float64],
    *,
    sample_count: int,
) -> NDArray[np.float64]:
    """QuEST-style spectral clipping based on Marchenko–Pastur support."""

    sigma = np.asarray(sample_covariance, dtype=np.float64)
    sigma = _symmetrize(sigma)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    p = eigvals.size
    if p == 0:
        return sigma.copy()
    n = max(sample_count, 1)
    q = float(p) / float(n)
    mean_eig = float(np.mean(eigvals))
    if mean_eig <= 0.0:
        return sigma.copy()
    sqrt_q = float(np.sqrt(max(q, 1e-6)))
    lam_min = mean_eig * max((1.0 - sqrt_q) ** 2, 0.0)
    lam_max = mean_eig * (1.0 + sqrt_q) ** 2
    shrunk = np.clip(eigvals, lam_min, lam_max)
    total_original = float(np.sum(eigvals))
    total_shrunk = float(np.sum(shrunk))
    if total_shrunk > 0.0 and total_original > 0.0:
        shrunk *= total_original / total_shrunk
    adjusted = eigvecs @ np.diag(shrunk) @ eigvecs.T
    return _symmetrize(adjusted)


def ewma_covariance(
    observations: NDArray[np.float64],
    *,
    halflife: float = 30.0,
    centre: bool = True,
) -> NDArray[np.float64]:
    """Exponentially weighted moving-average covariance estimate."""

    data = np.asarray(observations, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("observations must be two-dimensional (samples × assets).")
    n_samples, n_assets = data.shape
    if n_samples <= 1:
        raise ValueError("At least two observations required for EWMA covariance.")
    if halflife <= 0:
        raise ValueError("halflife must be positive.")

    decay = 0.5 ** (1.0 / float(halflife))
    weights = decay ** np.arange(n_samples - 1, -1, -1, dtype=np.float64)
    weights /= float(np.sum(weights))
    if centre:
        mean = np.average(data, axis=0, weights=weights)
    else:
        mean = np.zeros(n_assets, dtype=np.float64)
    centred = data - mean
    cov = np.zeros((n_assets, n_assets), dtype=np.float64)
    for obs, weight in zip(centred, weights, strict=False):
        cov += weight * np.outer(obs, obs)
    return _symmetrize(cov)
