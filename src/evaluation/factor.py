from __future__ import annotations

"""Factor-based covariance baselines used in evaluation and runner pipelines."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["observed_factor_covariance", "poet_lite_covariance", "POETResult"]


def observed_factor_covariance(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """
    Estimate Σ = B Σ_f Bᵀ + Σ_ε from observed factor returns via cross-sectional OLS.

    This thin wrapper normalises inputs before delegating to the finance.factors
    implementation used elsewhere in the project.
    """

    from finance.factors import factor_covariance

    if returns.empty:
        raise ValueError("returns DataFrame must contain observations.")
    if factors.empty:
        raise ValueError("factors DataFrame must contain observations.")
    returns = returns.copy()
    factors = factors.copy()
    returns.index = pd.to_datetime(returns.index)
    factors.index = pd.to_datetime(factors.index)
    returns = returns.sort_index()
    factors = factors.sort_index()
    return factor_covariance(returns, factors, add_intercept=add_intercept)


@dataclass(frozen=True)
class POETResult:
    covariance: np.ndarray
    n_factors: int


def _poet_ic(residual_var: np.ndarray, k: int, p: int, n: int) -> float:
    sigma2 = float(np.mean(np.clip(residual_var, 1e-12, None)))
    penalty = k * np.log(max(p, n)) / max(p, n)
    return float(np.log(sigma2) + penalty)


def poet_lite_covariance(
    returns: pd.DataFrame,
    *,
    max_factors: int = 10,
    shrink: str = "diag",
) -> POETResult:
    """
    Estimate a POET-lite covariance using PCA loadings with simple residual shrinkage.

    Parameters
    ----------
    returns
        Asset return matrix indexed by time, columns are assets.
    max_factors
        Maximum number of latent factors to consider when selecting k via IC.
    shrink
        Residual shrinkage scheme: 'diag' (default) or 'const' for constant correlation.
    """

    if returns.empty:
        raise ValueError("returns DataFrame must contain observations.")
    data = returns.astype(np.float64)
    data = data.dropna(axis=0, how="any")
    if data.empty:
        raise ValueError("returns contain only NaNs after cleaning.")
    x = data.to_numpy(dtype=np.float64, copy=True)
    x -= np.mean(x, axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False, ddof=1)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    p = cov.shape[0]
    n = x.shape[0]
    max_k = int(min(max_factors, max(p - 1, 0), max(n - 1, 0)))

    best_ic = float("inf")
    best_k = 0
    residual_cache: dict[int, np.ndarray] = {}

    for k in range(0, max_k + 1):
        if k == 0:
            residual = cov.copy()
        else:
            loadings = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0.0))
            factor_cov = loadings @ loadings.T
            residual = cov - factor_cov
        residual = 0.5 * (residual + residual.T)
        residual_cache[k] = residual
        ic_val = _poet_ic(np.diag(residual), k, p, n)
        if ic_val < best_ic:
            best_ic = ic_val
            best_k = k

    residual = residual_cache[best_k]
    if best_k > 0:
        loadings = eigvecs[:, :best_k] * np.sqrt(np.maximum(eigvals[:best_k], 0.0))
        factor_cov = loadings @ loadings.T
    else:
        loadings = np.zeros((p, 0), dtype=np.float64)
        factor_cov = np.zeros((p, p), dtype=np.float64)

    diag_variance = np.clip(np.diag(residual), 0.0, None)
    if shrink.lower() == "const" and p > 1:
        std = np.sqrt(np.clip(diag_variance, 1e-12, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = residual / np.outer(std, std)
        mask = ~np.eye(p, dtype=bool)
        avg_corr = float(np.mean(corr[mask][np.isfinite(corr[mask])])) if mask.any() else 0.0
        avg_corr = max(min(avg_corr, 0.99), -0.99)
        resid_cov = np.outer(std, std) * avg_corr
        np.fill_diagonal(resid_cov, diag_variance)
    else:
        resid_cov = np.diag(diag_variance)

    covariance = factor_cov + resid_cov
    covariance = np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)
    return POETResult(covariance=covariance, n_factors=best_k)
