from __future__ import annotations

import numpy as np

from src.baselines.rie import RIEConfig, rie_covariance


def _spiked_samples(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, p = 600, 8
    noise = rng.normal(size=(n, p))
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    factor = rng.normal(size=(n, 1))
    return noise + 5.0 * factor @ direction[np.newaxis, :]


def test_rie_covariance_flattens_bulk_eigenvalues() -> None:
    samples = _spiked_samples()
    sample_cov = np.cov(samples, rowvar=False, ddof=1)
    cleaned = rie_covariance(samples)

    sample_eigs = np.linalg.eigvalsh(sample_cov)
    cleaned_eigs = np.linalg.eigvalsh(cleaned)

    n, p = samples.shape
    sigma2 = float(np.mean(sample_eigs))
    lambda_plus = sigma2 * (1.0 + np.sqrt(p / n)) ** 2
    bulk_mask = sample_eigs <= lambda_plus + 1e-10
    if np.any(bulk_mask):
        assert np.std(cleaned_eigs[bulk_mask]) < 1e-8
    assert cleaned.shape == sample_cov.shape
    assert np.allclose(cleaned, cleaned.T, atol=1e-10)


def test_rie_covariance_respects_min_eigenvalue() -> None:
    samples = _spiked_samples(seed=11)
    config = RIEConfig(min_eigenvalue=0.05)
    cleaned = rie_covariance(samples, config=config)
    eigvals = np.linalg.eigvalsh(cleaned)
    assert eigvals.min() >= config.min_eigenvalue - 1e-8
