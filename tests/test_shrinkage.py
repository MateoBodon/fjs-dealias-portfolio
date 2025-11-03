from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance.ledoit import lw_cov
from finance.robust import huberize, tyler_shrink_covariance, winsorize
from finance.shrinkage import cc_covariance, oas_covariance

pytestmark = pytest.mark.unit


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


def test_shrinkers_warn_on_nonfinite_and_remain_psd(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(7)
    base = rng.standard_normal((128, 6))
    base[0, 0] = np.nan
    base[5, 3] = np.inf

    with caplog.at_level("WARNING"):
        sigma_oas = oas_covariance(base.copy())
    assert any("oas_covariance received" in record.message for record in caplog.records)
    np.testing.assert_allclose(sigma_oas, sigma_oas.T, atol=1e-12)
    assert np.linalg.eigvalsh(sigma_oas).min() >= -1e-10

    caplog.clear()
    with caplog.at_level("WARNING"):
        sigma_lw = lw_cov(base.copy())
    assert any("lw_cov received" in record.message for record in caplog.records)
    np.testing.assert_allclose(sigma_lw, sigma_lw.T, atol=1e-12)
    assert np.linalg.eigvalsh(sigma_lw).min() >= -1e-10


def test_winsorize_clips_extremes() -> None:
    frame = pd.DataFrame({"a": [-10.0, -1.0, 0.0, 1.0, 10.0]})
    clipped = winsorize(frame, 0.2)
    lower = frame.quantile(0.2)["a"]
    upper = frame.quantile(0.8)["a"]
    assert clipped["a"].min() >= lower - 1e-9
    assert clipped["a"].max() <= upper + 1e-9


def test_huberize_limits_outliers() -> None:
    frame = pd.DataFrame({"b": [-100.0, -0.5, 0.0, 0.5, 100.0]})
    clipped = huberize(frame, 1.5)
    assert clipped["b"].iloc[0] > -10.0
    assert clipped["b"].iloc[-1] < 10.0


def test_tyler_shrink_covariance_is_positive_definite() -> None:
    rng = np.random.default_rng(123)
    data = rng.standard_normal((80, 4))
    sigma = tyler_shrink_covariance(data, ridge=1e-3)
    assert sigma.shape == (4, 4)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(sigma)
    assert np.all(eigvals > 0.0)
