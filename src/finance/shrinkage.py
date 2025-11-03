from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import OAS

_PSD_TOL = 1e-10
_LOGGER = logging.getLogger(__name__)


def _validate_input(R: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(R, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Input array must be two-dimensional.")
    if arr.shape[0] <= 1:
        raise ValueError("At least two observations are required.")
    return arr


def _sample_covariance(X: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = X - X.mean(axis=0, keepdims=True)
    n_samples = centered.shape[0]
    return (centered.T @ centered) / (n_samples - 1)


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _warn_and_fill_nonfinite(name: str, data: NDArray[np.float64]) -> NDArray[np.float64]:
    mask = ~np.isfinite(data)
    if not np.any(mask):
        return data

    total = int(mask.sum())
    row_counts = {int(idx): int(count) for idx, count in enumerate(mask.sum(axis=1)) if count}
    col_counts = {int(idx): int(count) for idx, count in enumerate(mask.sum(axis=0)) if count}
    _LOGGER.warning(
        "%s received %d non-finite values; zero-filled residuals (rows=%s, cols=%s)",
        name,
        total,
        row_counts,
        col_counts,
    )
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _assert_psd_and_symmetric(name: str, matrix: NDArray[np.float64]) -> None:
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError(f"{name} covariance is not symmetric within tolerance.")
    eigvals = np.linalg.eigvalsh(matrix)
    if eigvals.size and float(eigvals.min()) < -_PSD_TOL:
        raise ValueError(f"{name} covariance is not positive semi-definite.")


def oas_covariance(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Oracle Approximating Shrinkage covariance targeting the identity matrix."""

    data = _validate_input(R)
    data = _warn_and_fill_nonfinite("oas_covariance", data)
    estimator = OAS(assume_centered=False)
    estimator.fit(data)
    sigma = _symmetrize(np.asarray(estimator.covariance_, dtype=np.float64))
    _assert_psd_and_symmetric("oas_covariance", sigma)
    return sigma


def cc_covariance(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ledoit–Wolf constant-correlation shrinkage covariance estimator."""

    data = _validate_input(R)
    n_samples, n_assets = data.shape
    if n_assets == 1:
        return np.array([[np.var(data, ddof=1)]], dtype=np.float64)

    centered = data - data.mean(axis=0, keepdims=True)
    sample_cov = _sample_covariance(data)
    variances = np.diag(sample_cov)
    if np.any(variances <= 0):
        raise ValueError("Sample covariance has non-positive variances.")
    std_dev = np.sqrt(variances)

    corr = sample_cov / np.outer(std_dev, std_dev)
    np.fill_diagonal(corr, 0.0)
    avg_corr = float(np.sum(corr) / (n_assets * (n_assets - 1)))
    target = avg_corr * np.outer(std_dev, std_dev)
    np.fill_diagonal(target, variances)

    xc = centered
    xc2 = xc**2
    phi_mat = (xc2.T @ xc2) / n_samples
    phi_mat -= 2 * (xc.T @ xc) * sample_cov / n_samples
    phi_mat += sample_cov**2
    phi = float(phi_mat.sum())

    gamma = float(np.sum((sample_cov - target) ** 2))
    if gamma <= 0:
        shrinkage = 0.0
    else:
        kappa = phi / gamma
        shrinkage = max(0.0, min(1.0, kappa / n_samples))

    sigma = shrinkage * target + (1.0 - shrinkage) * sample_cov
    sigma = _symmetrize(sigma)
    _assert_psd_and_symmetric("cc_covariance", sigma)

    return sigma
