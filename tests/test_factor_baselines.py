from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.factor import observed_factor_covariance, poet_lite_covariance


def test_observed_factor_covariance_matches_population() -> None:
    rng = np.random.default_rng(2025)
    n_obs, p, k = 400, 6, 3
    factors = rng.normal(size=(n_obs, k))
    betas = rng.normal(size=(p, k))
    epsilon = rng.normal(scale=0.1, size=(n_obs, p))
    returns = factors @ betas.T + epsilon
    returns -= returns.mean(axis=0, keepdims=True)
    factors -= factors.mean(axis=0, keepdims=True)

    dates = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    tickers = [f"A{i}" for i in range(p)]
    factor_cols = [f"F{i}" for i in range(k)]

    returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
    factors_df = pd.DataFrame(factors, index=dates, columns=factor_cols)

    est = observed_factor_covariance(returns_df, factors_df, add_intercept=False)

    factor_cov = np.cov(factors, rowvar=False, ddof=1)
    resid_var = np.var(epsilon, axis=0, ddof=1)
    true_cov = betas @ factor_cov @ betas.T + np.diag(resid_var)
    np.testing.assert_allclose(est, true_cov, rtol=5e-2, atol=2e-2)


def test_poet_lite_covariance_returns_valid_matrix() -> None:
    rng = np.random.default_rng(77)
    n_obs, p, k = 250, 8, 2
    factors = rng.normal(size=(n_obs, k))
    loadings = rng.normal(size=(p, k))
    eps = rng.normal(scale=0.15, size=(n_obs, p))
    returns = factors @ loadings.T + eps
    dates = pd.date_range("2021-01-01", periods=n_obs, freq="B")
    returns_df = pd.DataFrame(returns, index=dates, columns=[f"A{i}" for i in range(p)])

    poet_result = poet_lite_covariance(returns_df, max_factors=5)
    cov = poet_result.covariance
    eigvals = np.linalg.eigvalsh(cov)
    assert cov.shape == (p, p)
    assert np.all(np.isfinite(cov))
    assert eigvals.min() > -1e-6
    assert 0 <= poet_result.n_factors <= 5
