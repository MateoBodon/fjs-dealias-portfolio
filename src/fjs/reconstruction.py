from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def orthonormalize_candidate_directions(
    directions: Sequence[NDArray[np.float64]],
    *,
    dimension: int,
    rank_tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    """Return the closest ordered orthonormal basis for a candidate span.

    Symmetric (Löwdin) orthonormalisation is equivariant to candidate
    permutation and direction-sign changes.  It therefore provides a unique
    subspace treatment without the order dependence of sequential rank-one
    updates.  Rank-deficient candidate sets fail loudly because distinct
    eigenvalue targets cannot be assigned to duplicate directions.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if not directions:
        return np.empty((dimension, 0), dtype=np.float64)
    if len(directions) > dimension:
        raise ValueError("Candidate count cannot exceed the covariance dimension.")

    columns: list[NDArray[np.float64]] = []
    for direction in directions:
        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        if vector.size != dimension:
            raise ValueError("Candidate direction dimension does not match covariance.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Candidate directions must contain only finite values.")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("Candidate directions must have positive norm.")
        columns.append(np.asarray(vector / norm, dtype=np.float64))

    candidate_matrix = np.column_stack(columns)
    gram = np.asarray(candidate_matrix.T @ candidate_matrix, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    largest = float(np.max(eigenvalues, initial=0.0))
    threshold = float(rank_tolerance) * max(largest, 1.0)
    if np.any(eigenvalues <= threshold):
        raise ValueError(
            "Candidate directions are rank-deficient or numerically collinear."
        )

    inverse_root = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    basis = np.asarray(candidate_matrix @ inverse_root, dtype=np.float64)
    if not np.allclose(
        basis.T @ basis,
        np.eye(basis.shape[1], dtype=np.float64),
        rtol=1e-9,
        atol=1e-10,
    ):
        raise RuntimeError("Symmetric candidate orthonormalisation was unstable.")
    return basis


def replace_spectral_subspace(
    baseline: NDArray[np.float64],
    directions: Sequence[NDArray[np.float64]],
    target_eigenvalues: Sequence[float],
) -> NDArray[np.float64]:
    """Install candidate eigenpairs while preserving the orthogonal block.

    For the symmetric-orthonormalised candidate basis ``Q`` and
    ``P=I-QQ'``, the replacement is ``P B P + Q diag(mu) Q'``.  This makes
    every column of ``Q`` an exact eigenvector, preserves ``B`` on the
    orthogonal complement, and is invariant to candidate permutation and sign.
    """

    matrix = np.asarray(baseline, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("baseline must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("baseline must contain only finite values.")
    matrix = np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)

    targets = np.asarray(target_eigenvalues, dtype=np.float64).reshape(-1)
    if len(directions) != targets.size:
        raise ValueError("Each candidate direction requires one target eigenvalue.")
    if not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("Target eigenvalues must be positive and finite.")
    if targets.size == 0:
        return matrix.copy()

    basis = orthonormalize_candidate_directions(
        directions,
        dimension=matrix.shape[0],
    )
    projector = np.eye(matrix.shape[0], dtype=np.float64) - basis @ basis.T
    replaced = projector @ matrix @ projector + basis @ np.diag(targets) @ basis.T
    return np.asarray(0.5 * (replaced + replaced.T), dtype=np.float64)
