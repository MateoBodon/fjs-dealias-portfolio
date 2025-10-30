from __future__ import annotations

import numpy as np

from fjs.robust import edge_from_scatter, tyler_scatter


def test_tyler_scatter_returns_psd_matrix() -> None:
    rng = np.random.default_rng(123)
    data = rng.standard_t(df=5, size=(600, 6))
    scatter = tyler_scatter(data, max_iter=100, tol=1e-5)
    assert scatter.shape == (6, 6)
    eigvals = np.linalg.eigvalsh(scatter)
    assert np.all(np.isfinite(eigvals))
    assert float(eigvals.min()) > 0.0
    trace = float(np.trace(scatter))
    assert np.isclose(trace / 6.0, 1.0, atol=1e-2)


def test_edge_from_scatter_monotone_in_scale() -> None:
    rng = np.random.default_rng(321)
    observations = rng.normal(size=(500, 5))
    cov = np.cov(observations, rowvar=False, ddof=1)
    edge_base = edge_from_scatter(cov, 5, 500)
    edge_scaled = edge_from_scatter(1.5 * cov, 5, 500)
    assert np.isfinite(edge_base) and np.isfinite(edge_scaled)
    assert edge_scaled > edge_base
