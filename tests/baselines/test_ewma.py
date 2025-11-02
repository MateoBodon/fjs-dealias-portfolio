from __future__ import annotations

import numpy as np
import pytest

from src.baselines.ewma import EWMAConfig, ewma_covariance


def test_ewma_covariance_matches_weighted_variance() -> None:
    data = np.array([[1.0], [2.0], [5.0]])
    config = EWMAConfig(lambda_=0.5, debias=True)
    cov = ewma_covariance(data, config=config)

    weights = np.array([(1 - config.lambda_) * config.lambda_ ** k for k in range(data.shape[0] - 1, -1, -1)])
    weights /= (1 - config.lambda_ ** data.shape[0])
    mean = np.sum(weights * data.ravel())
    expected_var = np.sum(weights * (data.ravel() - mean) ** 2)

    assert cov.shape == (1, 1)
    assert cov[0, 0] == pytest.approx(expected_var, rel=1e-6)


def test_ewma_covariance_positive_semidefinite() -> None:
    rng = np.random.default_rng(99)
    samples = rng.normal(size=(400, 3))
    cov = ewma_covariance(samples, config=EWMAConfig(lambda_=0.97, debias=False))
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10)
    assert np.allclose(cov, cov.T, atol=1e-10)
