from __future__ import annotations

import numpy as np
import pytest

from src.fjs.edge import EdgeConfig, EdgeEstimate, EdgeMode, compute_edge


def test_compute_edge_tyler_buffers_raw_edge() -> None:
    rng = np.random.default_rng(42)
    data = rng.standard_t(df=5, size=(800, 6))
    estimate = compute_edge(data)
    assert isinstance(estimate, EdgeEstimate)
    assert estimate.mode is EdgeMode.TYLER
    assert estimate.edge > estimate.raw_edge > 0.0
    assert estimate.noise_scale > 0.0


def test_compute_edge_scm_matches_mp_formula() -> None:
    rng = np.random.default_rng(123)
    n, p = 1200, 12
    data = rng.normal(size=(n, p))
    cfg = EdgeConfig(mode=EdgeMode.SCM, buffer=0.0, buffer_frac=0.0)
    estimate = compute_edge(data, config=cfg)
    gamma = p / n
    expected = estimate.noise_scale * (1.0 + np.sqrt(gamma)) ** 2
    assert estimate.raw_edge == pytest.approx(expected, rel=0.05)
    assert estimate.edge == pytest.approx(estimate.raw_edge, rel=1e-6)
