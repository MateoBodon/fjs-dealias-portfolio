from __future__ import annotations

import numpy as np
import pytest

import finance.portfolios as portfolios
from experiments.eval import run as eval_run


def _raise_missing(*_args, **_kwargs):
    raise portfolios.MissingSolverError("cvxpy missing for eval test")


def test_min_variance_weights_raises_when_cvxpy_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portfolios, "_get_cvxpy", _raise_missing)

    with pytest.raises(portfolios.MissingSolverError):
        eval_run._min_variance_weights(np.eye(2), solver="cvxpy")


def test_min_variance_weights_skip_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portfolios, "_get_cvxpy", _raise_missing)

    weights, info = eval_run._min_variance_weights(
        np.eye(3), solver="cvxpy", skip_on_missing_solver=True, box=(0.0, 1.0)
    )

    assert weights.size == 0
    assert info["skipped"] is True
    assert info["skip_reason"] == "missing_solver"
    assert info["solver_status"] == "missing_solver"
    assert np.isnan(float(info.get("objective", np.nan)))
