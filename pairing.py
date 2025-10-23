from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def _normalize_rows(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array of vectors (n, p).")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero; leave zero vectors unchanged
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def align_spikes(
    true_vecs: NDArray[np.float64],
    est_vecs: NDArray[np.float64],
) -> NDArray[np.intp]:
    """
    Compute an assignment that pairs estimated spike directions to true ones.

    The assignment maximizes total absolute cosine similarity using the
    Hungarian algorithm. Returned is a permutation `perm` of length `k` where
    `perm[i]` is the index (column) in `est_vecs` matched to `true_vecs[i]`.
    If there are fewer estimated vectors than true vectors, unmatched entries
    are set to -1.
    """
    T = _normalize_rows(np.asarray(true_vecs, dtype=np.float64))
    E = _normalize_rows(np.asarray(est_vecs, dtype=np.float64))
    k = int(T.shape[0])
    m = int(E.shape[0])
    if T.shape[1] != E.shape[1]:
        raise ValueError("true_vecs and est_vecs must have the same width.")
    if k == 0:
        return np.zeros(0, dtype=np.intp)
    if m == 0:
        return -np.ones(k, dtype=np.intp)

    # Similarity matrix: absolute cosine similarity
    sim = np.abs(T @ E.T)
    # Convert to a minimization cost for Hungarian algorithm
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    perm = -np.ones(k, dtype=np.intp)
    for r, c in zip(row_ind, col_ind, strict=False):
        if r < k:
            perm[int(r)] = int(c)
    return perm

