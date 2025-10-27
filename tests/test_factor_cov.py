from __future__ import annotations

import numpy as np
import pandas as pd

from finance.factors import factor_covariance


def test_factor_covariance_matches_theoretical_when_noise_zero() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-05", periods=7, freq="W-FRI")

    factors_full = rng.normal(scale=0.02, size=(7, 2))
    factors_full -= factors_full.mean(axis=0, keepdims=True)
    betas = np.array([[0.6, -0.2], [1.1, 0.3], [-0.4, 0.9]])

    returns_full = factors_full @ betas.T

    R_df = pd.DataFrame(returns_full, index=dates, columns=list("ABC"))
    F_df = pd.DataFrame(factors_full[1:], index=dates[1:], columns=["MKT", "SMB"])

    sigma_hat = factor_covariance(R_df, F_df)

    factors_used = factors_full[1:]
    expected = betas @ np.cov(factors_used, rowvar=False, ddof=1) @ betas.T

    np.testing.assert_allclose(sigma_hat, expected, atol=1e-10)
    eigvals = np.linalg.eigvalsh(sigma_hat)
    assert eigvals.min() >= -1e-10


def test_factor_covariance_with_industry_and_missing_returns() -> None:
    rng = np.random.default_rng(2024)
    dates = pd.date_range("2024-02-02", periods=10, freq="W-FRI")

    base_factors = rng.normal(scale=0.015, size=(10, 2))
    base_factors -= base_factors.mean(axis=0, keepdims=True)
    industry_factor = rng.normal(scale=0.01, size=(10, 1))
    industry_factor -= industry_factor.mean(axis=0, keepdims=True)

    F_df = pd.DataFrame(base_factors, index=dates, columns=["MKT", "SMB"])
    industry_df = pd.DataFrame(industry_factor, index=dates, columns=["IND"])

    combined = np.hstack([base_factors, industry_factor])
    betas = np.array(
        [
            [0.8, -0.1, 0.3],
            [0.5, 0.9, -0.2],
            [-0.6, 0.7, 0.4],
            [0.3, -0.5, 0.6],
        ]
    )
    noise = rng.normal(scale=0.002, size=(10, 4))

    returns = combined @ betas.T + noise
    R_df = pd.DataFrame(returns, index=dates, columns=list("WXYZ"))

    R_df.iloc[0, 1] = np.nan
    R_df.iloc[5, 3] = np.nan
    noise_with_nan = noise.copy()
    noise_with_nan[0, 1] = np.nan
    noise_with_nan[5, 3] = np.nan

    sigma_hat = factor_covariance(
        R_df, F_df, add_intercept=False, industry_df=industry_df
    )

    factor_cov = np.cov(combined, rowvar=False, ddof=1)
    resid_var = np.nanvar(noise_with_nan, axis=0, ddof=1)
    expected = betas @ factor_cov @ betas.T + np.diag(resid_var)

    np.testing.assert_allclose(sigma_hat, expected, atol=1e-3)
    eigvals = np.linalg.eigvalsh(sigma_hat)
    assert eigvals.min() >= -1e-10
