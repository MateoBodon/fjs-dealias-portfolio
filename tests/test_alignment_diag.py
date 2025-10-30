from __future__ import annotations

import numpy as np

from evaluation.evaluate import alignment_diagnostics


def test_alignment_angle_boundaries() -> None:
    cov = np.diag([4.0, 1.5, 0.2])
    direction = np.array([1.0, 0.2, 0.0], dtype=float)
    angle_deg, energy_mu = alignment_diagnostics(cov, direction, top_p=2)
    assert np.isfinite(angle_deg)
    assert 0.0 <= angle_deg <= 90.0
    assert np.isfinite(energy_mu)


def test_alignment_angle_handles_full_rank() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + np.eye(5)
    direction = rng.normal(size=5)
    angle_deg, energy_mu = alignment_diagnostics(cov, direction, top_p=3)
    assert np.isfinite(angle_deg)
    assert 0.0 <= angle_deg <= 90.0
    assert np.isfinite(energy_mu)
