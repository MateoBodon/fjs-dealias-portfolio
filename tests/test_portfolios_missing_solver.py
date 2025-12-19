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


def test_optimize_portfolio_raises_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    covariance = np.eye(2, dtype=np.float64)
    monkeypatch.setattr(portfolios, "_get_cvxpy", _raise_missing)

    with pytest.raises(portfolios.MissingSolverError):
        portfolios.optimize_portfolio(covariance)


def test_optimize_portfolio_skip_flag_marks_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    covariance = np.eye(3, dtype=np.float64)
    monkeypatch.setattr(portfolios, "_get_cvxpy", _raise_missing)

    result = portfolios.optimize_portfolio(covariance, skip_on_missing_solver=True)

    assert result.skipped is True
    assert result.skip_reason == "missing_solver"
    assert result.solver_status == "missing_solver"
    assert result.converged is False
    assert result.weights.size == 0  # no silent EW fallback
    assert np.isnan(result.objective)


def test_force_missing_env_triggers_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    covariance = np.eye(2, dtype=np.float64)
    monkeypatch.setenv("FJS_FORCE_MISSING_CVXPY", "1")

    result = portfolios.optimize_portfolio(
        covariance, skip_on_missing_solver=True, box=(0.0, 1.0)
    )

    assert result.skipped is True
    assert result.skip_reason == "missing_solver"
    assert result.solver_status == "missing_solver"
    assert result.weights.size == 0
    assert np.isnan(result.objective)
