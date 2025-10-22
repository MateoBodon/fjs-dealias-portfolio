from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class OptimizationResult:
    """Result container for portfolio optimisation routines."""

    weights: NDArray[np.float64]
    objective: float
    converged: bool


def optimize_portfolio(
    covariance: NDArray[np.float64],
    target_return: float,
) -> OptimizationResult:
    """
    Solve a risk-aware portfolio optimisation problem.

    Parameters
    ----------
    covariance:
        Covariance matrix used for risk evaluation.
    target_return:
        Required expected return for the optimal portfolio.

    Returns
    -------
    OptimizationResult
        Candidate solution and metadata about solver convergence.
    """
    raise NotImplementedError("Portfolio optimisation is not implemented yet.")
