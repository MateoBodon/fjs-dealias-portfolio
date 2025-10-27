from __future__ import annotations

import numpy as np

from finance.portfolio import apply_turnover_cost, minvar_ridge_box, turnover


def test_minvar_ridge_box_respects_box_and_sum_constraints() -> None:
    sigma = np.array(
        [
            [0.05, 0.02, 0.015],
            [0.02, 0.04, 0.018],
            [0.015, 0.018, 0.03],
        ],
        dtype=np.float64,
    )

    weights, info = minvar_ridge_box(sigma, box=(0.0, 0.6), ridge=1e-3)

    assert info["converged"] is True
    assert np.isfinite(info["objective"])
    assert abs(weights.sum() - 1.0) <= 1e-6
    assert np.all(weights >= -1e-9)
    assert np.all(weights <= 0.6 + 1e-9)


def test_minvar_ridge_box_matches_objective_and_improves_conditioning() -> None:
    sigma = np.array(
        [
            [1.0, 0.99, 0.99],
            [0.99, 1.0, 0.99],
            [0.99, 0.99, 1.0],
        ],
        dtype=np.float64,
    )
    ridge = 1e-2

    weights, info = minvar_ridge_box(sigma, box=(0.0, 1.0), ridge=ridge)
    penalised = sigma + ridge * np.eye(3)

    manual_objective = float(weights @ penalised @ weights)
    assert abs(manual_objective - info["objective"]) <= 1e-9

    cond_original = np.linalg.cond(sigma)
    cond_penalised = np.linalg.cond(penalised)
    assert cond_penalised < cond_original


def test_turnover_and_turnover_cost_application() -> None:
    prev = np.array([0.3, 0.4, 0.3])
    new = np.array([0.25, 0.5, 0.25])

    one_way = turnover(prev, new)
    assert abs(one_way - 0.1) <= 1e-12

    var_series = np.array([0.02, 0.021, 0.019])
    weights = [prev, new, np.array([0.2, 0.55, 0.25])]

    adjusted, costs = apply_turnover_cost(var_series, weights, bps=25)
    assert adjusted.shape == var_series.shape
    assert costs[0] == 0.0
    expected_cost = turnover(new, weights[2]) * (25 / 10000.0)
    assert abs(costs[2] - expected_cost) <= 1e-12
    assert np.all(adjusted <= var_series + 1e-12)
