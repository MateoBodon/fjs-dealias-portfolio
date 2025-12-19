from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from numpy.typing import NDArray


class MissingSolverError(RuntimeError):
    """Raised when a required optimisation solver dependency is unavailable."""


def _get_cvxpy():  # pragma: no cover - optional dependency
    force_missing = os.environ.get("FJS_FORCE_MISSING_CVXPY")
    if force_missing and force_missing.strip().lower() not in {"0", "false", "off"}:
        raise MissingSolverError(
            "cvxpy import forced missing via FJS_FORCE_MISSING_CVXPY=1."
        )
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
    skip_reason: str | None = None
    solver_used: str | None = None


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


def _solve_min_variance_cvxpy(
    covariance: NDArray[np.float64],
    *,
    allow_short: bool = False,
    box: tuple[float, float] | None = None,
    ridge: float = 0.0,
    solver: str | None = None,
) -> OptimizationResult:
    """Solve the minimum-variance problem using cvxpy.

    This helper is shared by the public wrappers and encapsulates dependency
    handling and failure reporting so callers can choose to raise or skip.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if box is not None and box[0] >= box[1]:
        raise ValueError("Lower bound must be strictly less than upper bound.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    cp = _get_cvxpy()

    n = covariance.shape[0]
    cov = (covariance + covariance.T) / 2.0
    if ridge > 0.0:
        cov = cov + float(ridge) * np.eye(n, dtype=np.float64)

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]

    if box is not None:
        lb, ub = float(box[0]), float(box[1])
        constraints.append(w >= lb)
        constraints.append(w <= ub)
        allow_short = allow_short or lb < 0.0
    if not allow_short:
        constraints.append(w >= 0)

    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
    problem.solve(solver=solver, warm_start=True)

    solver_status = str(problem.status)
    converged = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if not converged or w.value is None:
        return OptimizationResult(
            weights=np.array([], dtype=np.float64),
            objective=float("nan"),
            converged=False,
            solver_status=solver_status,
            skipped=True,
            skip_reason="solver_failed",
            solver_used=solver or "cvxpy",
        )

    weights = np.asarray(w.value, dtype=np.float64).flatten()
    objective_value = float(problem.value)
    return OptimizationResult(
        weights=weights,
        objective=objective_value,
        converged=True,
        solver_status=solver_status,
        skipped=False,
        skip_reason=None,
        solver_used=solver or "cvxpy",
    )


def minimum_variance(
    covariance: NDArray[np.float64],
    *,
    allow_short: bool = False,
    solver: str | None = None,
    ridge: float = 0.0,
) -> OptimizationResult:
    """Solve the minimum-variance problem using cvxpy (if available)."""

    return _solve_min_variance_cvxpy(
        covariance,
        allow_short=allow_short,
        solver=solver,
        ridge=ridge,
        box=None,
    )


def min_variance_box(
    covariance: NDArray[np.float64],
    lb: float = -0.02,
    ub: float = 0.02,
    *,
    solver: str | None = None,
    ridge: float = 0.0,
) -> OptimizationResult:
    """
    Solve the minimum-variance problem with box constraints.
    """

    return _solve_min_variance_cvxpy(
        covariance,
        allow_short=True,
        box=(lb, ub),
        ridge=ridge,
        solver=solver,
    )


def optimize_portfolio(
    covariance: NDArray[np.float64],
    target_return: float | None = None,
    *,
    allow_short: bool = False,
    skip_on_missing_solver: bool = False,
    box: tuple[float, float] | None = None,
    ridge: float = 0.0,
    solver: str | None = None,
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
    box
        Optional pair of lower/upper bounds for each weight.
    ridge
        Optional ridge penalty added to the covariance prior to optimisation.
    solver
        Optional cvxpy solver name.

    Returns
    -------
    OptimizationResult
        Candidate solution with convergence flag.
    """

    try:
        return _solve_min_variance_cvxpy(
            covariance,
            allow_short=allow_short,
            box=box,
            ridge=ridge,
            solver=solver,
        )
    except MissingSolverError:
        if skip_on_missing_solver:
            weights = np.array([], dtype=np.float64)
            return OptimizationResult(
                weights=weights,
                objective=float("nan"),
                converged=False,
                solver_status="missing_solver",
                skipped=True,
                skip_reason="missing_solver",
                solver_used=solver or "cvxpy",
            )
        raise
