from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(returns_df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Clip each column of ``returns_df`` to its [q, 1-q] empirical quantiles."""

    if not 0.0 < q < 0.5:
        raise ValueError("winsorize quantile must be between 0 and 0.5.")
    if returns_df.empty:
        return returns_df.copy()
    lower = returns_df.quantile(q, interpolation="linear")
    upper = returns_df.quantile(1.0 - q, interpolation="linear")
    clipped = returns_df.clip(lower=lower, upper=upper, axis="columns")
    return clipped


def huberize(returns_df: pd.DataFrame, c: float) -> pd.DataFrame:
    """Apply column-wise Huber clipping using median and MAD scale."""

    if c <= 0.0:
        raise ValueError("Huber threshold must be positive.")
    if returns_df.empty:
        return returns_df.copy()
    med = returns_df.median(axis=0, skipna=True)
    abs_dev = (returns_df - med).abs()
    mad = abs_dev.median(axis=0, skipna=True)
    # Convert MAD to an approximation of the std dev; fall back to sample std when degenerate
    scale = 1.4826 * mad
    fallback = returns_df.std(axis=0, ddof=1).replace(0.0, np.nan)
    scale = scale.where(scale > 0.0, fallback)
    scale = scale.fillna(1.0)
    lower = med - c * scale
    upper = med + c * scale
    clipped = returns_df.clip(lower=lower, upper=upper, axis="columns")
    return clipped


def tyler_shrink_covariance(
    observations: np.ndarray | pd.DataFrame,
    *,
    ridge: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    """Return a Tyler M-estimator with ridge regularisation for positive definiteness."""

    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("observations must be a 2-D array.")
    n, p = x.shape
    if n == 0 or p == 0:
        raise ValueError("observations must have positive shape.")
    x = x - np.nanmean(x, axis=0, keepdims=True)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    cov = np.cov(x, rowvar=False, bias=True)
    if not np.isfinite(cov).all() or np.linalg.norm(cov) == 0.0:
        cov = np.eye(p, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    trace = np.trace(cov)
    if trace <= 0.0:
        cov = np.eye(p, dtype=np.float64)
        trace = p
    cov /= trace / float(p)

    for _ in range(max_iter):
        try:
            inv_cov = np.linalg.pinv(cov, rcond=1e-12)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov + 1e-6 * np.eye(p), rcond=1e-12)
        quad = np.sum((x @ inv_cov) * x, axis=1)
        quad = np.maximum(quad, 1e-12)
        weights = p / quad
        cov_next = (x.T * weights) @ x
        cov_next /= weights.sum()
        cov_next = 0.5 * (cov_next + cov_next.T)
        trace_next = np.trace(cov_next)
        if not np.isfinite(trace_next) or trace_next <= 0.0:
            cov_next = cov
            break
        cov_next /= trace_next / float(p)
        diff = np.linalg.norm(cov_next - cov, ord="fro")
        cov = cov_next
        if diff < tol:
            break

    cov += float(max(ridge, 0.0)) * np.eye(p, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return cov
