from __future__ import annotations

"""
Robust scatter estimators and Marčenko–Pastur edge adjustments.

The helpers here provide light-weight alternatives to the standard sample
covariance (SCM) edge that can be used to scale detection margins without
modifying the core FJS machinery.
"""

from math import isfinite, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["tyler_scatter", "huber_scatter", "edge_from_scatter"]


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Input matrix must be two-dimensional.")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("Input matrix must have positive shape.")
    return x


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _initial_scatter(x: np.ndarray) -> NDArray[np.float64]:
    n, p = x.shape
    if n <= 1:
        return np.eye(p, dtype=np.float64)
    cov = np.cov(x, rowvar=False, ddof=1)
    if not np.all(np.isfinite(cov)):
        cov = np.cov(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), rowvar=False, ddof=1)
    cov = _symmetrize(np.asarray(cov, dtype=np.float64))
    trace = float(np.trace(cov))
    if not isfinite(trace) or trace <= 0.0:
        cov = np.eye(p, dtype=np.float64)
        trace = float(p)
    return cov / (trace / float(p))


def tyler_scatter(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    ridge: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Return the Tyler fixed-point scatter estimate with optional ridge regularisation.

    Parameters
    ----------
    observations
        Array-like input shaped ``(n_samples, n_features)``.
    max_iter
        Maximum number of fixed-point iterations.
    tol
        Convergence tolerance on the Frobenius norm between successive iterates.
    ridge
        Non-negative ridge added to the diagonal at the end to guarantee positive
        definiteness.
    """

    x = np.asarray(observations, dtype=np.float64)
    x = _ensure_2d(x)
    n, p = x.shape
    if n < p:
        # Tyler requires n >= p for strict convergence; fall back to a mild ridge.
        ridge = max(ridge, (float(p - n) + 1.0) * 1e-3)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    scatter = _initial_scatter(x)

    for _ in range(max_iter):
        try:
            inv = np.linalg.pinv(scatter, rcond=1e-12)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(scatter + 1e-6 * np.eye(p), rcond=1e-12)

        quad = np.sum((x @ inv) * x, axis=1)
        quad = np.maximum(quad, 1e-12)
        weights = (p / quad).reshape(-1, 1)
        scatter_next = (x * weights).T @ x
        scatter_next /= float(n)
        scatter_next = _symmetrize(scatter_next)

        trace = float(np.trace(scatter_next))
        if not isfinite(trace) or trace <= 0.0:
            scatter_next = scatter
            break
        scatter_next /= trace / float(p)

        diff = np.linalg.norm(scatter_next - scatter, ord="fro")
        scatter = scatter_next
        if diff < tol:
            break

    scatter += float(max(ridge, 0.0)) * np.eye(p, dtype=np.float64)
    return _symmetrize(scatter)


def huber_scatter(
    observations: Iterable[Iterable[float]] | NDArray[np.float64],
    c: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    ridge: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Compute a Huber-type reweighted scatter estimator.

    Parameters
    ----------
    observations
        Array-like input shaped ``(n_samples, n_features)``.
    c
        Positive threshold parameter controlling the Huber influence function.
    max_iter
        Maximum number of fixed-point iterations.
    tol
        Convergence tolerance on the Frobenius norm between successive iterates.
    ridge
        Non-negative ridge added to the diagonal at the end for stability.
    """

    if c <= 0.0 or not isfinite(c):
        raise ValueError("Huber threshold c must be positive and finite.")

    x = np.asarray(observations, dtype=np.float64)
    x = _ensure_2d(x)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n, p = x.shape

    scatter = _initial_scatter(x)

    for _ in range(max_iter):
        try:
            inv = np.linalg.pinv(scatter, rcond=1e-12)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(scatter + 1e-6 * np.eye(p), rcond=1e-12)

        quad = np.sum((x @ inv) * x, axis=1)
        quad = np.maximum(quad, 1e-12)
        distances = np.sqrt(quad)
        weights = np.ones_like(distances)
        mask = distances > c
        weights[mask] = c / distances[mask]
        weights = weights.reshape(-1, 1)

        scatter_next = (x * weights).T @ x
        scatter_next /= float(np.sum(weights**2))
        scatter_next = _symmetrize(scatter_next)

        trace = float(np.trace(scatter_next))
        if not isfinite(trace) or trace <= 0.0:
            scatter_next = scatter
            break
        scatter_next /= trace / float(p)

        diff = np.linalg.norm(scatter_next - scatter, ord="fro")
        scatter = scatter_next
        if diff < tol:
            break

    scatter += float(max(ridge, 0.0)) * np.eye(p, dtype=np.float64)
    return _symmetrize(scatter)


def edge_from_scatter(
    scatter: NDArray[np.float64],
    n_features: int,
    n_samples: int,
) -> float:
    """
    Estimate the upper Marčenko–Pastur edge from a scatter matrix.

    A median-diagonal noise proxy is combined with the classical MP formula
    ``σ² (1 + √γ)²`` where ``γ = p / n``.
    """

    sigma = np.asarray(scatter, dtype=np.float64)
    sigma = _symmetrize(sigma)
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Scatter matrix must be square.")
    if n_features <= 0 or n_samples <= 0:
        raise ValueError("n_features and n_samples must be positive.")
    if sigma.shape[0] != n_features:
        raise ValueError("Scatter matrix dimension must match n_features.")

    diag = np.diag(sigma)
    diag = diag[np.isfinite(diag)]
    if diag.size == 0:
        noise = float(np.mean(np.linalg.eigvalsh(sigma)))
    else:
        noise = float(np.median(diag))
    noise = float(max(noise, 0.0))
    if noise == 0.0:
        return 0.0

    ratio = float(n_features) / float(n_samples)
    ratio = max(ratio, 0.0)
    upper = noise * (1.0 + sqrt(ratio)) ** 2
    return float(max(upper, 0.0))
