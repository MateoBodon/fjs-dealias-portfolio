from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from baselines.factors import PrewhitenResult, prewhiten_returns
from fjs.dealias import dealias_search


def _simulated_returns(
    rng: np.random.Generator,
    *,
    n_obs: int,
    n_assets: int,
    n_factors: int,
    factor_scale: float,
    noise_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=n_obs, freq="B")
    factors = rng.normal(scale=factor_scale, size=(n_obs, n_factors))
    betas = rng.normal(scale=0.8, size=(n_assets, n_factors))
    noise = rng.normal(scale=noise_scale, size=(n_obs, n_assets))
    returns = factors @ betas.T + noise
    returns_df = pd.DataFrame(
        returns,
        index=dates,
        columns=[f"A{i}" for i in range(n_assets)],
    )
    factor_cols = [f"F{i}" for i in range(n_factors)]
    factors_df = pd.DataFrame(factors, index=dates, columns=factor_cols)
    return returns_df, factors_df


def test_prewhiten_reduces_spike_strength() -> None:
    rng = np.random.default_rng(2025)
    returns_df, factors_df = _simulated_returns(
        rng,
        n_obs=400,
        n_assets=12,
        n_factors=1,
        factor_scale=1.5,
        noise_scale=0.1,
    )

    result = prewhiten_returns(returns_df, factors_df)
    assert isinstance(result, PrewhitenResult)
    cov_before = np.cov(returns_df.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
    cov_after = np.cov(
        result.residuals.to_numpy(dtype=np.float64),
        rowvar=False,
        ddof=1,
    )
    eig_before = np.linalg.eigvalsh(cov_before)
    eig_after = np.linalg.eigvalsh(cov_after)
    assert eig_after.max() < 0.5 * eig_before.max()
    assert result.r_squared.max() > 0.8
    assert result.betas.shape == (12, 1)


@pytest.mark.slow
def test_prewhiten_residuals_preserve_null_fpr() -> None:
    rng = np.random.default_rng(77)
    groups = 30
    replicates = 3
    assets = 18
    trials = 50
    detections = 0

    for _ in range(trials):
        n_obs = groups * replicates
        returns_df, factors_df = _simulated_returns(
            rng,
            n_obs=n_obs,
            n_assets=assets,
            n_factors=1,
            factor_scale=1.0,
            noise_scale=0.5,
        )
        residuals = prewhiten_returns(returns_df, factors_df).residuals
        y_matrix = residuals.to_numpy(dtype=np.float64).reshape(groups, replicates, assets)
        y_flat = y_matrix.reshape(groups * replicates, assets)
        group_labels = np.repeat(np.arange(groups, dtype=np.intp), replicates)
        found = dealias_search(
            y_flat,
            group_labels,
            target_r=0,
            delta=0.5,
            eps=0.02,
            a_grid=60,
        )
        detections += int(bool(found))

    assert detections <= math.ceil(0.02 * trials)


def test_prewhiten_result_exposes_betas_and_intercepts() -> None:
    rng = np.random.default_rng(11)
    returns_df, factors_df = _simulated_returns(
        rng,
        n_obs=180,
        n_assets=6,
        n_factors=3,
        factor_scale=1.2,
        noise_scale=0.2,
    )
    result = prewhiten_returns(returns_df, factors_df)
    assert list(result.betas.columns) == list(factors_df.columns)
    assert not result.intercept.empty
    assert result.intercept.index.tolist() == returns_df.columns.tolist()
    assert not result.r_squared.empty
    assert result.r_squared.index.tolist() == returns_df.columns.tolist()
    assert np.all(result.r_squared.to_numpy(dtype=np.float64) >= 0.0)
