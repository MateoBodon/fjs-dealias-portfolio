from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class MissingSolverError(RuntimeError):
    """Raised when a required optimisation solver dependency is unavailable."""


def _get_cvxpy():  # pragma: no cover - optional dependency
    try:
        import cvxpy as cp  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise MissingSolverError(
            "cvxpy is required for the minimum-variance optimiser. Install via `pip install cvxpy` "
            "or rerun with an explicit skip flag if you intend to drop MV results."
        ) from exc
    return cp


@dataclass
class OptimizationResult:
    """Result container for portfolio optimisation routines."""

    weights: NDArray[np.float64]
    objective: float
    converged: bool
    solver_status: str | None = None
    skipped: bool = False


def equal_weight(p: int) -> NDArray[np.float64]:
    """Return the equal-weight vector for ``p`` assets.

    Parameters
    ----------
    p
        Number of assets in the universe.

    Returns
    -------
    numpy.ndarray
        Weight vector summing to one.
    """

    if p <= 0:
        raise ValueError("Number of assets must be positive.")
    return np.full(p, 1.0 / p, dtype=np.float64)


def minimum_variance(
    covariance: NDArray[np.float64],
    *,
    allow_short: bool = False,
    solver: str | None = None,
) -> OptimizationResult:
    """Solve the minimum-variance problem using cvxpy (if available).

    Parameters
    ----------
    covariance
        Sample covariance matrix shaped ``(p, p)``.
    allow_short
        If ``False`` (default), impose non-negativity on weights.
    solver
        Optional cvxpy solver name.

    Returns
    -------
    OptimizationResult
        Portfolio weights and objective value.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    cp = _get_cvxpy()

    n = covariance.shape[0]
    cov = (covariance + covariance.T) / 2.0
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, warm_start=True)

    solver_status = str(problem.status)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        weights = equal_weight(n)
        return OptimizationResult(
            weights=weights,
            objective=float("nan"),
            converged=False,
            solver_status=solver_status,
        )

    weights = np.asarray(w.value, dtype=np.float64).flatten()
    objective_value = float(problem.value)
    return OptimizationResult(
        weights=weights,
        objective=objective_value,
        converged=True,
        solver_status=solver_status,
    )


def min_variance_box(
    covariance: NDArray[np.float64],
    lb: float = -0.02,
    ub: float = 0.02,
    *,
    solver: str | None = None,
) -> OptimizationResult:
    """
    Solve the minimum-variance problem with box constraints.

    Parameters
    ----------
    covariance
        Sample covariance matrix shaped ``(p, p)``.
    lb, ub
        Lower/upper bounds for each weight. Defaults to +/-2%.
    solver
        Optional cvxpy solver name.

    Returns
    -------
    OptimizationResult
        Portfolio weights and objective value.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if lb >= ub:
        raise ValueError("Lower bound must be strictly less than upper bound.")

    cp = _get_cvxpy()

    n = covariance.shape[0]
    cov = (covariance + covariance.T) / 2.0
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [
        cp.sum(w) == 1,
        w >= lb,
        w <= ub,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, warm_start=True)

    solver_status = str(problem.status)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        weights = equal_weight(n)
        return OptimizationResult(
            weights=weights,
            objective=float("nan"),
            converged=False,
            solver_status=solver_status,
        )

    weights = np.asarray(w.value, dtype=np.float64).flatten()
    objective_value = float(problem.value)
    return OptimizationResult(
        weights=weights,
        objective=objective_value,
        converged=True,
        solver_status=solver_status,
    )


def optimize_portfolio(
    covariance: NDArray[np.float64],
    target_return: float | None = None,
    *,
    allow_short: bool = False,
    skip_on_missing_solver: bool = False,
) -> OptimizationResult:
    """Return the minimum-variance portfolio; fail loud if solver is missing by default.

    Parameters
    ----------
    covariance
        Sample covariance matrix shaped ``(p, p)``.
    target_return
        Unused placeholder for future extensions.
    allow_short
        If ``False`` (default), impose non-negativity.
    skip_on_missing_solver
        If ``True``, mark the optimisation as skipped when cvxpy is absent instead of
        raising an error.

    Returns
    -------
    OptimizationResult
        Candidate solution with convergence flag.
    """

    try:
        return minimum_variance(covariance, allow_short=allow_short)
    except MissingSolverError:
        if skip_on_missing_solver:
            weights = np.array([], dtype=np.float64)
            return OptimizationResult(
                weights=weights,
                objective=float("nan"),
                converged=False,
                solver_status="missing_dependency",
                skipped=True,
            )
        raise
