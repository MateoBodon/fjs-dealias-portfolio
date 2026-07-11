from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fjs.mp import (
    admissible_m_from_lambda,
    m_edge,
    mp_edge,
    z0,
    z0_double_prime,
    z0_prime,
)

pytestmark = pytest.mark.unit


def _balanced_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Two-strata balanced design; selected to yield concave stationary edge
    a = np.array([0.2, 0.0], dtype=np.float64)
    bulk_scales = np.array([1.0, 1.0], dtype=np.float64)
    d = np.array([10.0, 20.0], dtype=np.float64)
    n_bulk = 20.0
    return a, bulk_scales, d, n_bulk


def test_derivatives_match_finite_differences() -> None:
    a, bulk_scales, d, n_bulk = _balanced_params()
    # Choose an m away from singularities: ensure 1 + k_s m not too small
    k = (n_bulk / d) * a * bulk_scales
    m = -0.2
    # Ensure we are well away from singularities
    assert np.all(1.0 + k * m > 1e-3)

    h = 1e-6
    zc = z0(m, a, bulk_scales, d, n_bulk)
    zp = z0(m + h, a, bulk_scales, d, n_bulk)
    zm = z0(m - h, a, bulk_scales, d, n_bulk)

    num_first = (zp - zm) / (2.0 * h)
    num_second = (zp - 2.0 * zc + zm) / (h * h)

    ana_first = z0_prime(m, a, bulk_scales, d, n_bulk)
    ana_second = z0_double_prime(m, a, bulk_scales, d, n_bulk)

    assert_allclose(ana_first, num_first, rtol=1e-6, atol=1e-7)
    assert_allclose(ana_second, num_second, rtol=1e-5, atol=5e-6)


def test_round_trip_lambda_to_m_to_lambda() -> None:
    a, bulk_scales, d, n_bulk = _balanced_params()
    # Upper edge of the bulk
    z_plus = mp_edge(a, bulk_scales, d, n_bulk)
    # Probe the outlier branch strictly above the upper edge.
    eps = 1e-3
    lam_grid = np.linspace(z_plus + eps, 2.0 * z_plus, 12)

    for lam in lam_grid:
        m = admissible_m_from_lambda(lam, a, bulk_scales, d, n_bulk)
        lam_back = z0(m, a, bulk_scales, d, n_bulk)
        assert_allclose(lam_back, lam, rtol=0.0, atol=1e-6)


def test_convexity_at_upper_edge() -> None:
    a, bulk_scales, d, n_bulk = _balanced_params()
    m_plus = m_edge(a, bulk_scales, d, n_bulk)
    curv = z0_double_prime(m_plus, a, bulk_scales, d, n_bulk)
    assert curv > 0.0
    # Cross-check: z(m_plus) equals mp_edge(a, C, d, N)
    assert_allclose(
        z0(m_plus, a, bulk_scales, d, n_bulk),
        mp_edge(a, bulk_scales, d, n_bulk),
        atol=1e-10,
    )
