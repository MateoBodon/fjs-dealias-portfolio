from __future__ import annotations

import math

import numpy as np
import pytest

from fjs.balanced import mean_squares
from fjs.dealias import dealias_search
from fjs.theta_solver import ThetaSolverParams, solve_theta_for_t2_zero


def _synthetic_oneway_dataset(
    *,
    seed: int = 0,
    groups: int = 4,
    replicates: int = 6,
    p: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    labels: list[int] = []
    for g in range(groups):
        base = np.array(
            [g * 3.0, (-1) ** g * 0.8, g * 0.4],
            dtype=np.float64,
        )
        for _ in range(replicates):
            noise = rng.normal(scale=0.15, size=p)
            observations.append(base + noise)
            labels.append(g)
    return np.asarray(observations, dtype=np.float64), np.asarray(labels, dtype=np.int64)


@pytest.mark.unit
def test_theta_solver_brackets_root(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_t_vec(
        lambda_hat: float,
        a: list[float],
        C: list[float],
        d: list[float],
        N: float,
        c: list[float],
        order: list[list[int]],
        Cs: list[float] | None = None,
    ) -> np.ndarray:
        theta = math.atan2(a[1], a[0])
        return np.array([1.0, math.cos(theta) - 0.25], dtype=np.float64)

    monkeypatch.setattr("fjs.theta_solver.t_vec", fake_t_vec)

    params = ThetaSolverParams(
        C=np.ones(2, dtype=np.float64),
        d=np.array([12.0, 18.0], dtype=np.float64),
        N=5.0,
        c=np.array([4.0, 1.0], dtype=np.float64),
        order=[[1, 2], [2]],
        Cs=None,
        eps=1e-3,
        delta=1e-4,
        grid_size=360,
        tol=1e-6,
        max_iter=80,
    )

    theta = solve_theta_for_t2_zero(2.0, params)
    assert theta is not None
    t_vals = fake_t_vec(
        2.0,
        [math.cos(theta), math.sin(theta)],
        [],
        [],
        0.0,
        [],
        [],
    )
    assert abs(t_vals[1]) <= 1e-6
    expected = math.acos(0.25)
    # Compare modulo 2π
    diff = abs(((theta - expected) + math.pi) % (2.0 * math.pi) - math.pi)
    assert diff <= 1e-3


@pytest.mark.integration
def test_theta_solver_fallback_to_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    y, groups = _synthetic_oneway_dataset(seed=7)
    stats = mean_squares(y, groups)

    monkeypatch.setattr(
        "fjs.theta_solver.solve_theta_for_t2_zero",
        lambda lambda_hat, params: None,
    )

    detections = dealias_search(
        y,
        groups,
        target_r=0,
        delta=0.0,
        delta_frac=None,
        eps=1e-6,
        stability_eta_deg=0.0,
        use_tvector=True,
        nonnegative_a=False,
        a_grid=90,
        cs_drop_top_frac=0.0,
        cs_sensitivity_frac=0.0,
        scan_basis="ms",
        diagnostics={},
        stats=stats,
        oneway_a_solver="rootfind",
    )
    assert detections
    assert all(det.get("solver_used") == "grid" for det in detections)


@pytest.mark.integration
def test_theta_solver_logs_solver_flag() -> None:
    y, groups = _synthetic_oneway_dataset(seed=13)
    stats = mean_squares(y, groups)

    detections = dealias_search(
        y,
        groups,
        target_r=0,
        delta=0.0,
        delta_frac=None,
        eps=1e-4,
        stability_eta_deg=0.2,
        use_tvector=True,
        nonnegative_a=False,
        a_grid=72,
        cs_drop_top_frac=0.0,
        cs_sensitivity_frac=0.0,
        scan_basis="ms",
        diagnostics={},
        stats=stats,
        oneway_a_solver="auto",
    )
    assert detections
    assert all(det.get("solver_used") in {"grid", "rootfind"} for det in detections)
