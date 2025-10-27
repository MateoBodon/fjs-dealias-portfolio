from __future__ import annotations

import numpy as np

from finance.shrinkage import cc_covariance, oas_covariance


def test_oas_covariance_returns_psd() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(scale=0.02, size=(256, 5))

    sigma = oas_covariance(data)

    assert sigma.shape == (5, 5)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(sigma)
    assert eigvals.min() >= -1e-10


def test_constant_correlation_shrinkage_reduces_off_diagonal_weight() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(scale=0.03, size=(320, 4))

    sample_cov = np.cov(data, rowvar=False, ddof=1)
    cc_cov = cc_covariance(data)

    off_diag_sample = np.mean(np.abs(sample_cov - np.diag(np.diag(sample_cov))))
    off_diag_cc = np.mean(np.abs(cc_cov - np.diag(np.diag(cc_cov))))
    assert off_diag_cc <= off_diag_sample

    eigvals = np.linalg.eigvalsh(cc_cov)
    assert eigvals.min() >= -1e-10

    std = np.sqrt(np.diag(cc_cov))
    corr = cc_cov / np.outer(std, std)
    np.testing.assert_allclose(np.diag(corr), np.ones(4), atol=1e-12)
