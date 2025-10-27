from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NestedDesignMetadata:
    """Balanced nested Year⊃Week design metadata."""

    d: NDArray[np.float64]
    c: NDArray[np.float64]
    N: float
    I: int
    J: int
    replicates: int
    n: int
    p: int


def _validate_labels(
    labels: Sequence[object],
    name: str,
    expected_length: int,
) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.shape[0] != expected_length:
        raise ValueError(f"{name} must contain {expected_length} entries.")
    return arr


def mean_squares_nested(
    y: np.ndarray,
    year_labels: Sequence[object],
    week_of_year_labels: Sequence[object],
    replicates: int,
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], NestedDesignMetadata]:
    """
    Compute balanced nested Year⊃Week MANOVA mean squares.

    Parameters
    ----------
    y:
        Observation matrix shaped ``(n, p)``.
    year_labels:
        Year identifiers for each observation (balanced).
    week_of_year_labels:
        Week identifiers nested within each year.
    replicates:
        Number of replicates per (year, week) cell.

    Returns
    -------
    tuple
        ``(MS_between_year, MS_week_within_year, MS_residual)`` along with
        :class:`NestedDesignMetadata`.
    """

    observations = np.asarray(y, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("y must be a two-dimensional array shaped (n, p).")

    n_samples, n_features = observations.shape
    if n_samples == 0:
        raise ValueError("At least one observation is required.")

    reps = int(replicates)
    if reps <= 1:
        raise ValueError("replicates must be an integer greater than 1.")

    years = _validate_labels(year_labels, "year_labels", n_samples)
    weeks = _validate_labels(week_of_year_labels, "week_of_year_labels", n_samples)

    # Build structured pairs to count unique (year, week) combinations.
    dtype = np.dtype([("year", years.dtype), ("week", weeks.dtype)])
    combined = np.empty(n_samples, dtype=dtype)
    combined["year"] = years
    combined["week"] = weeks
    pair_values, pair_inverse, pair_counts = np.unique(
    combined, return_inverse=True, return_counts=True
    )

    if pair_values.size == 0:
        raise ValueError("Balanced nested design requires at least one (year, week) cell.")

    if not np.all(pair_counts == reps):
        raise ValueError("Observed replicate counts do not match the supplied replicates.")

    unique_years, year_inverse = np.unique(years, return_inverse=True)
    I = int(unique_years.size)
    if I < 2:
        raise ValueError("At least two years are required for the nested design.")

    # Map each (year, week) pair to its year index.
    year_lookup = {val: idx for idx, val in enumerate(unique_years)}
    pair_year_index = np.array(
        [year_lookup[val] for val in pair_values["year"]], dtype=np.intp
    )
    counts_per_year = np.bincount(pair_year_index, minlength=I)
    if not np.all(counts_per_year == counts_per_year[0]):
        raise ValueError("Each year must contain the same number of weeks.")
    J = int(counts_per_year[0])
    if J < 1:
        raise ValueError("At least one week per year is required.")

    expected_samples = I * J * reps
    if n_samples != expected_samples:
        raise ValueError("Observation count does not match I * J * replicates.")

    # Compute per-year totals and means.
    year_totals = np.zeros((I, n_features), dtype=np.float64)
    np.add.at(year_totals, year_inverse, observations)
    per_year_denominator = J * reps
    year_means = year_totals / per_year_denominator
    grand_mean = observations.mean(axis=0)

    # Compute per (year, week) means.
    n_cells = pair_values.shape[0]
    cell_totals = np.zeros((n_cells, n_features), dtype=np.float64)
    np.add.at(cell_totals, pair_inverse, observations)
    cell_means = cell_totals / reps

    # Between-year mean square (stratum 1).
    centered_year_means = year_means - grand_mean
    ss_year = J * reps * (centered_year_means.T @ centered_year_means)
    ms_year = ss_year / float(I - 1)

    # Week-within-year mean square (stratum 2).
    cell_year_means = year_means[pair_year_index]
    diff_week_within = cell_means - cell_year_means
    ss_week = reps * (diff_week_within.T @ diff_week_within)
    ms_week = ss_week / float(I * (J - 1))

    # Residual mean square (stratum 3).
    residuals = observations - cell_means[pair_inverse]
    ss_residual = residuals.T @ residuals
    ms_residual = ss_residual / float(I * J * (reps - 1))

    d = np.array(
        [float(I - 1), float(I * (J - 1)), float(expected_samples - I * J)],
        dtype=np.float64,
    )
    c = np.array(
        [float(J * reps), float(reps), 1.0],
        dtype=np.float64,
    )

    metadata = NestedDesignMetadata(
        d=d,
        c=c,
        N=float(reps),
        I=I,
        J=J,
        replicates=reps,
        n=expected_samples,
        p=n_features,
    )

    return (ms_year, ms_week, ms_residual), metadata
