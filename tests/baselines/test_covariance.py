from __future__ import annotations

import numpy as np
import pytest

from baselines.covariance import (
    cc_covariance,
    ewma_covariance,
    lw_covariance,
    oas_covariance,
    quest_covariance,
    rie_covariance,
    sample_covariance,
)


def _make_observations(samples: int = 64, assets: int = 6, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(samples, assets))
    return data


def _assert_psd(matrix: np.ndarray, atol: float = 1e-10) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.size:
        assert float(eigenvalues.min()) >= -atol


@pytest.mark.parametrize("factory", [sample_covariance, lw_covariance, oas_covariance, cc_covariance])
def test_statistical_covariances_are_psd(factory) -> None:  # type: ignore[no-untyped-def]
    data = _make_observations()
    sigma = factory(data)
    assert sigma.shape == (data.shape[1], data.shape[1])
    assert np.allclose(sigma, sigma.T, atol=1e-12)
    _assert_psd(sigma)


def test_ewma_covariance_matches_manual_loop() -> None:
    data = _make_observations(samples=32, assets=4)
    sigma = ewma_covariance(data, halflife=12.0)
    # Compare against brute-force weighted sum
    decay = 0.5 ** (1.0 / 12.0)
    weights = decay ** np.arange(data.shape[0] - 1, -1, -1, dtype=np.float64)
    weights /= float(np.sum(weights))
    mean = np.average(data, axis=0, weights=weights)
    centred = data - mean
    brute = sum(w * np.outer(row, row) for w, row in zip(weights, centred, strict=False))
    brute = 0.5 * (brute + brute.T)
    assert np.allclose(sigma, brute, atol=1e-10)


def test_rie_covariance_shrinks_towards_mean_spectrum() -> None:
    rng = np.random.default_rng(42)
    sigma = rng.normal(size=(5, 5))
    sigma = sigma @ sigma.T  # PSD sample covariance
    sigma = 0.5 * (sigma + sigma.T)
    shrunk = rie_covariance(sigma, sample_count=200)
    eigvals = np.linalg.eigvalsh(sigma)
    shrunk_eigvals = np.linalg.eigvalsh(shrunk)
    assert shrunk.shape == sigma.shape
    # Shrunk eigenvalues should lie between min and max of original spectrum
    assert shrunk_eigvals.min() >= eigvals.min() - 1e-10
    assert shrunk_eigvals.max() <= eigvals.max() + 1e-10


def test_quest_covariance_clips_to_mp_support() -> None:
    data = _make_observations(samples=200, assets=8)
    sample = sample_covariance(data)
    quest = quest_covariance(sample, sample_count=data.shape[0])
    eigvals_sample = np.linalg.eigvalsh(sample)
    eigvals_quest = np.linalg.eigvalsh(quest)
    assert eigvals_quest.min() >= eigvals_sample.min() - 1e-8
    assert eigvals_quest.max() <= eigvals_sample.max() + 1e-8
