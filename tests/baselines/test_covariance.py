from __future__ import annotations

import numpy as np

from baselines.covariance import ewma_covariance, quest_covariance, rie_covariance


def _sample_data(seed: int, n_obs: int, n_assets: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.5, size=(n_obs, n_assets))


def test_rie_covariance_psd() -> None:
    data = _sample_data(1, 120, 6)
    sample_cov = np.cov(data, rowvar=False, ddof=1)
    sigma = rie_covariance(sample_cov, sample_count=data.shape[0])
    eigvals = np.linalg.eigvalsh(sigma)
    assert sigma.shape == (6, 6)
    assert eigvals.min() >= -1e-8


def test_quest_covariance_clips_spectrum() -> None:
    data = _sample_data(2, 90, 8)
    sample_cov = np.cov(data, rowvar=False, ddof=1)
    sigma = quest_covariance(sample_cov, sample_count=data.shape[0])
    eigvals = np.linalg.eigvalsh(sigma)
    assert sigma.shape == (8, 8)
    assert eigvals.min() >= -1e-8
    assert eigvals.max() <= max(np.linalg.eigvalsh(sample_cov)) * 1.2


def test_ewma_covariance_matches_sample_in_limit() -> None:
    data = _sample_data(3, 64, 5)
    ewma = ewma_covariance(data, halflife=1_000.0)
    sample_cov = np.cov(data, rowvar=False, ddof=1)
    eigvals = np.linalg.eigvalsh(ewma)
    assert ewma.shape == (5, 5)
    assert eigvals.min() >= -1e-8
    assert np.allclose(ewma, sample_cov, atol=1e-2)
