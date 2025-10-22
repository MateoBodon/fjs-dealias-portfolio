from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:  # pragma: no cover - optional dependency
    import cvxpy as cp

    HAS_CVXPY = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HAS_CVXPY = False


@dataclass
class OptimizationResult:
    """Result container for portfolio optimisation routines."""

    weights: NDArray[np.float64]
    objective: float
    converged: bool


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
    if not HAS_CVXPY:
        raise ImportError("cvxpy is required for the minimum-variance optimiser.")

    n = covariance.shape[0]
    cov = (covariance + covariance.T) / 2.0
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, warm_start=True)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        weights = equal_weight(n)
        return OptimizationResult(
            weights=weights,
            objective=float("nan"),
            converged=False,
        )

    weights = np.asarray(w.value, dtype=np.float64).flatten()
    objective_value = float(problem.value)
    return OptimizationResult(
        weights=weights,
        objective=objective_value,
        converged=True,
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

    if not HAS_CVXPY:
        raise ImportError(
            "cvxpy is required for the box-constrained minimum-variance optimiser."
        )

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

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        weights = equal_weight(n)
        return OptimizationResult(
            weights=weights,
            objective=float("nan"),
            converged=False,
        )

    weights = np.asarray(w.value, dtype=np.float64).flatten()
    objective_value = float(problem.value)
    return OptimizationResult(
        weights=weights,
        objective=objective_value,
        converged=True,
    )


def optimize_portfolio(
    covariance: NDArray[np.float64],
    target_return: float | None = None,
    *,
    allow_short: bool = False,
) -> OptimizationResult:
    """Return the minimum-variance portfolio when possible, else equal-weight.

    Parameters
    ----------
    covariance
        Sample covariance matrix shaped ``(p, p)``.
    target_return
        Unused placeholder for future extensions.
    allow_short
        If ``False`` (default), impose non-negativity.

    Returns
    -------
    OptimizationResult
        Candidate solution with convergence flag.
    """

    try:
        return minimum_variance(covariance, allow_short=allow_short)
    except ImportError:
        weights = equal_weight(covariance.shape[0])
        objective = float(weights @ covariance @ weights)
        return OptimizationResult(
            weights=weights,
            objective=objective,
            converged=False,
        )
