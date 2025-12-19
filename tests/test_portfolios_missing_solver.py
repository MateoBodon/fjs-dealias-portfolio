from __future__ import annotations

import numpy as np
import pytest

import finance.portfolios as portfolios


def _raise_missing(*_args, **_kwargs):
    raise portfolios.MissingSolverError("cvxpy missing for test")


def test_minimum_variance_raises_when_solver_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    covariance = np.eye(2, dtype=np.float64)
    monkeypatch.setattr(portfolios, "_get_cvxpy", _raise_missing)

    with pytest.raises(portfolios.MissingSolverError):
        portfolios.minimum_variance(covariance)


def test_optimize_portfolio_skip_flag_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    covariance = np.eye(3, dtype=np.float64)
    monkeypatch.setattr(portfolios, "_get_cvxpy", _raise_missing)

    result = portfolios.optimize_portfolio(covariance, skip_on_missing_solver=True)

    assert result.skipped is True
    assert result.solver_status == "missing_dependency"
    assert result.converged is False
    assert result.weights.size == 0  # no silent EW fallback
    assert np.isnan(result.objective)
