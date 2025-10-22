from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BalancedConfig:
    """Configuration for the balanced risk contribution solver."""

    regularization: float
    max_iter: int
    tolerance: float = 1e-6
    mode: Literal["long-only", "long-short"] = "long-only"

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if self.mode not in {"long-only", "long-short"}:
            raise ValueError("mode must be either 'long-only' or 'long-short'.")


def _validate_balanced_inputs(
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.int_], np.ndarray]:
    """Return validated observations and grouping assignments for a balanced design."""
    observations = np.asarray(y, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("Y must be a two-dimensional array shaped (n, p).")

    group_assignments = np.asarray(groups)
    if group_assignments.ndim != 1:
        raise ValueError("groups must be a one-dimensional array.")
    if observations.shape[0] != group_assignments.shape[0]:
        raise ValueError("Y and groups must contain the same number of rows.")

    unique_groups, inverse, counts = np.unique(
        group_assignments, return_inverse=True, return_counts=True
    )
    if unique_groups.size == 0:
        raise ValueError("At least one group is required.")
    if not np.all(counts == counts[0]):
        raise ValueError("Design must be balanced; unequal group sizes detected.")

    return observations, inverse, counts, unique_groups


def _compute_group_means(
    observations: NDArray[np.float64],
    inverse: NDArray[np.intp],
    counts: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Accumulate per-group means using the grouping inverse index."""
    n_groups = counts.size
    n_features = observations.shape[1]
    totals = np.zeros((n_groups, n_features), dtype=np.float64)
    np.add.at(totals, inverse, observations)
    return totals / counts[:, None]


def group_means(
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute per-group and overall means for a balanced one-way MANOVA design.

    Parameters
    ----------
    Y:
        Observation matrix shaped `(n, p)`, where `p` is the feature dimension.
    groups:
        Group membership labels of length `n`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the matrix of group means shaped `(I, p)` and the
        overall mean vector shaped `(p,)`.
    """
    observations, inverse, counts, _ = _validate_balanced_inputs(y, groups)
    group_mean_matrix = _compute_group_means(observations, inverse, counts)
    overall_mean = observations.mean(axis=0)
    return group_mean_matrix, overall_mean


def mean_squares(y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """
    Estimate balanced one-way MANOVA mean squares and covariance components.

    Parameters
    ----------
    Y:
        Observation matrix shaped `(n, p)`, with `p` features.
    groups:
        Group membership labels of length `n`.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the between-group and within-group mean squares,
        covariance component estimates, and design metadata.
    """
    observations, inverse, counts, unique_groups = _validate_balanced_inputs(y, groups)

    n_groups = unique_groups.size
    if n_groups < 2:
        raise ValueError("At least two groups are required for MANOVA mean squares.")

    replicates = int(counts[0])
    if replicates < 2:
        raise ValueError("Each group must contain at least two replicates.")

    n, p = observations.shape
    if n <= n_groups:
        raise ValueError("The design requires more observations than groups.")

    group_mean_matrix = _compute_group_means(observations, inverse, counts)
    overall_mean = observations.mean(axis=0)
    centered_means = group_mean_matrix - overall_mean

    ms_between = (replicates / float(n_groups - 1)) * (
        centered_means.T @ centered_means
    )

    residuals = observations - group_mean_matrix[inverse]
    ms_within = (residuals.T @ residuals) / float(n - n_groups)

    sigma2_hat = ms_within
    sigma1_hat = (ms_between - ms_within) / float(replicates)

    return {
        "MS1": ms_between,
        "MS2": ms_within,
        "Sigma1_hat": sigma1_hat,
        "Sigma2_hat": sigma2_hat,
        "I": int(n_groups),
        "J": int(replicates),
        "n": int(n),
        "p": int(p),
    }


def compute_balanced_weights(
    returns: NDArray[np.float64],
    config: BalancedConfig,
) -> NDArray[np.float64]:
    """
    Compute portfolio weights that balance the contribution of estimated MANOVA spikes.

    Parameters
    ----------
    returns:
        Matrix of asset returns shaped `(n_samples, n_assets)`.
    config:
        Hyper-parameters controlling convergence and regularisation.

    Returns
    -------
    numpy.ndarray
        Array of weights shaped `(n_assets,)`.

    Raises
    ------
    NotImplementedError
        Always raised until the solver is implemented in a subsequent milestone.
    """
    raise NotImplementedError("Balanced weight computation is not implemented yet.")
