from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_PSD_TOL = 1e-10


def _align_frames(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    industry: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Align inputs on their shared date index and drop factor-side NaNs."""

    index = returns.index.intersection(factors.index)
    if industry is not None:
        index = index.intersection(industry.index)
    if index.empty:
        raise ValueError("No overlapping observations between returns and factors.")

    factors_aligned = factors.loc[index].copy()
    if factors_aligned.empty:
        raise ValueError("Factor matrix is empty after alignment.")
    factors_aligned = factors_aligned.astype(np.float64)
    factors_aligned = factors_aligned.dropna(axis=0, how="any")
    if factors_aligned.empty:
        raise ValueError("Factor matrix contains only NaNs after alignment.")

    returns_aligned = returns.loc[factors_aligned.index].copy()
    returns_aligned = returns_aligned.astype(np.float64)

    if industry is not None:
        industry_aligned = industry.loc[factors_aligned.index].copy()
        industry_aligned = industry_aligned.astype(np.float64)
    else:
        industry_aligned = None

    return returns_aligned, factors_aligned, industry_aligned


def _prepare_design(
    factors: pd.DataFrame,
    industry: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine factor and industry data into a single numeric design matrix."""

    pieces: list[pd.DataFrame] = [factors]
    if industry is not None:
        pieces.append(industry)

    design = pd.concat(pieces, axis=1)
    if design.empty:
        raise ValueError("No factors available for covariance estimation.")

    design = design.astype(np.float64)
    design = design.dropna(axis=0, how="any")
    if design.empty:
        raise ValueError("Design matrix contains only NaN rows.")
    return design


def factor_covariance(
    R_df: pd.DataFrame,
    F_df: pd.DataFrame,
    *,
    add_intercept: bool = True,
    industry_df: pd.DataFrame | None = None,
) -> NDArray[np.float64]:
    """Estimate an observed-factor covariance matrix via cross-sectional OLS.

    Parameters
    ----------
    R_df
        Asset return matrix shaped ``(n_periods, n_assets)``.
    F_df
        Factor return matrix shaped ``(n_periods, n_factors)``.
    add_intercept
        Whether to include an intercept when fitting factor loadings.
    industry_df
        Optional industry factor returns aligned by date.

    Returns
    -------
    numpy.ndarray
        Estimated covariance matrix shaped ``(n_assets, n_assets)``.
    """

    if R_df.empty:
        raise ValueError("R_df must contain observations.")
    if F_df.empty:
        raise ValueError("F_df must contain observations.")

    returns_aligned, factors_aligned, industry_aligned = _align_frames(
        R_df, F_df, industry_df
    )
    design = _prepare_design(factors_aligned, industry_aligned)

    factor_matrix = design.to_numpy(dtype=np.float64, copy=True)
    n_obs, n_factors = factor_matrix.shape
    if n_obs <= 1:
        raise ValueError("At least two observations required for covariance estimation.")
    if n_factors == 0:
        raise ValueError("At least one factor required for covariance estimation.")

    factor_cov = np.atleast_2d(np.cov(factor_matrix, rowvar=False, ddof=1))
    if not np.isfinite(factor_cov).all():
        raise ValueError("Factor covariance contains non-finite entries.")

    n_assets = returns_aligned.shape[1]
    betas = np.zeros((n_assets, n_factors), dtype=np.float64)
    resid_vars = np.zeros(n_assets, dtype=np.float64)

    for idx, column in enumerate(returns_aligned.columns):
        y = returns_aligned[column].to_numpy(dtype=np.float64, copy=True)
        valid = np.isfinite(y)
        if valid.sum() <= n_factors + int(add_intercept):
            raise ValueError(
                f"Not enough valid observations to fit factor loadings for {column}."
            )

        X = factor_matrix[valid, :]
        y_valid = y[valid]

        if add_intercept:
            X_aug = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])
        else:
            X_aug = X

        beta, _, _, _ = np.linalg.lstsq(X_aug, y_valid, rcond=None)
        betas[idx, :] = beta[1:] if add_intercept else beta

        residuals = y_valid - X_aug @ beta
        if residuals.size > 1:
            resid_vars[idx] = max(float(np.var(residuals, ddof=1)), 0.0)
        else:
            resid_vars[idx] = 0.0

    sigma = betas @ factor_cov @ betas.T + np.diag(resid_vars)
    sigma = np.asarray(0.5 * (sigma + sigma.T), dtype=np.float64)

    eigvals = np.linalg.eigvalsh(sigma)
    if eigvals.size and float(eigvals.min()) < -_PSD_TOL:
        raise ValueError("Estimated covariance is not positive semi-definite.")

    return sigma
