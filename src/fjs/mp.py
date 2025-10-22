from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ruff: noqa: N803

ArrayLike = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True)
class MarchenkoPasturModel:
    """Summary statistics for a Marchenko–Pastur limiting law."""

    n_samples: int
    n_features: int

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")

    @property
    def aspect_ratio(self) -> float:
        """Return the sample-to-feature aspect ratio."""
        return self.n_features / self.n_samples


def _prepare_inputs(  # noqa: N803
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a_arr = np.asarray(a, dtype=np.float64)
    c_arr = np.asarray(C, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)
    if a_arr.ndim != 1 or c_arr.ndim != 1 or d_arr.ndim != 1:
        raise ValueError("Parameters a, C, and d must be one-dimensional.")
    if not (a_arr.shape == c_arr.shape == d_arr.shape):
        raise ValueError("Parameters a, C, and d must have identical shapes.")
    if np.any(d_arr <= 0):
        raise ValueError("All entries in d must be positive.")
    n_float = float(N)
    if not np.isfinite(n_float) or n_float <= 0:
        raise ValueError("N must be a positive finite scalar.")
    return a_arr, c_arr, d_arr, n_float


def _k_values(  # noqa: N803
    a: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    N: float,
) -> np.ndarray:
    return np.asarray((N / d) * a * C, dtype=np.float64)


def z_of_m(  # noqa: N803
    m: float,
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
) -> float:
    """
    Evaluate the closed-form Marčenko--Pastur z(m) transform.

    Parameters
    ----------
    m:
        Candidate value of the Stieltjes transform.
    a, C, d, N:
        Model parameters as described in Proposition 5.4.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    m_val = float(m)
    if not np.isfinite(m_val) or m_val == 0.0:
        raise ValueError("m must be a non-zero finite scalar.")
    k_vals = _k_values(a_arr, c_arr, d_arr, n_float)
    denom = 1.0 + k_vals * m_val
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        raise ValueError("Denominator becomes singular for the supplied m.")
    contributions = c_arr * a_arr / denom
    return float(-1.0 / m_val + np.sum(contributions))


def _dz_dm(  # noqa: N803
    m: float,
    a: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    N: float,
    k_vals: np.ndarray,
) -> float:
    denom = 1.0 + k_vals * m
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        return float("nan")
    term1 = 1.0 / (m * m)
    term2 = float(np.sum((N / d) * (a**2) * (C**2) / (denom**2)))
    return term1 - term2


def _d2z_dm2(  # noqa: N803
    m: float,
    a: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    N: float,
    k_vals: np.ndarray,
) -> float:
    denom = 1.0 + k_vals * m
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        return float("nan")
    term1 = -2.0 / (m**3)
    term2 = float(np.sum(2.0 * (N / d) * (a**2) * (C**2) * k_vals / (denom**3)))
    return term1 + term2


def _logspace_grid() -> np.ndarray:
    grid = np.logspace(-12, 6, num=200, dtype=np.float64)
    return -grid


def _augment_with_singularities(grid: np.ndarray, k_vals: np.ndarray) -> np.ndarray:
    points = list(grid.tolist())
    for value in k_vals:
        if value <= 0:
            continue
        singularity = -1.0 / value
        eps = max(1e-8, abs(singularity) * 1e-6)
        points.append(singularity - eps)
        shifted = singularity + eps
        if shifted < -1e-12:
            points.append(shifted)
    points.append(-1e-12)
    points = sorted(set(pt for pt in points if pt < -1e-12))
    return np.array(points, dtype=np.float64)


def _bisect(
    func: Callable[[float], float],
    left: float,
    right: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> float:
    f_left = func(left)
    f_right = func(right)
    if not (np.isfinite(f_left) and np.isfinite(f_right)):
        raise RuntimeError("Non-finite function evaluation encountered in bisection.")
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0:
        raise RuntimeError("Bisection endpoints do not bracket a root.")
    a, b = left, right
    fa = f_left
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        for attempt in range(8):
            try:
                fm = func(mid)
            except ValueError:
                direction = a if attempt % 2 == 0 else b
                shifted = np.nextafter(mid, direction)
                if shifted == mid:
                    continue
                mid = shifted
                continue
            if not np.isfinite(fm):
                direction = a if attempt % 2 == 0 else b
                shifted = np.nextafter(mid, direction)
                if shifted == mid:
                    continue
                mid = shifted
                continue
            break
        else:
            raise RuntimeError("Unable to evaluate function during bisection.")
        if abs(fm) < tol or abs(b - a) < tol:
            return mid
        if fa * fm > 0:
            a, fa = mid, fm
        else:
            b = mid
    return 0.5 * (a + b)


def _root_brackets(
    func: Callable[[float], float],
    points: np.ndarray,
) -> list[tuple[float, float]]:
    brackets: list[tuple[float, float]] = []
    prev_x: float | None = None
    prev_val: float | None = None
    for x in points:
        val = func(x)
        if not np.isfinite(val):
            prev_x = None
            prev_val = None
            continue
        if prev_x is not None and prev_val is not None:
            if val == 0.0:
                brackets.append((x, x))
            elif prev_val == 0.0:
                brackets.append((prev_x, prev_x))
            elif val * prev_val < 0:
                brackets.append((prev_x, x))
        prev_x = x
        prev_val = val
    return brackets


def mp_edge(  # noqa: N803
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
) -> float:
    """
    Locate the upper bulk edge of the Marčenko--Pastur distribution.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    k_vals = _k_values(a_arr, c_arr, d_arr, n_float)

    grid = _augment_with_singularities(_logspace_grid(), k_vals)

    def derivative(m_val: float) -> float:
        return _dz_dm(m_val, a_arr, c_arr, d_arr, n_float, k_vals)

    def curvature(m_val: float) -> float:
        return _d2z_dm2(m_val, a_arr, c_arr, d_arr, n_float, k_vals)

    brackets = _root_brackets(derivative, grid)

    roots: list[float] = []
    for left, right in brackets:
        if left == right:
            root = left
        else:
            try:
                root = _bisect(derivative, left, right)
            except RuntimeError:
                continue
        roots.append(root)

    if not roots:
        # fall back to the point with minimal absolute derivative
        values = [
            (abs(derivative(point)), point)
            for point in grid
            if np.isfinite(derivative(point))
        ]
        if not values:
            raise RuntimeError("Unable to locate a stationary point for z(m).")
        _, candidate = min(values, key=lambda pair: pair[0])
        roots.append(candidate)

    # Prefer roots with negative second derivative.
    roots_sorted = sorted(roots, reverse=True)
    for root in roots_sorted:
        curvature_val = curvature(root)
        if np.isfinite(curvature_val) and curvature_val < 0:
            return float(z_of_m(root, a_arr, c_arr, d_arr, n_float))

    # If no root satisfies curvature < 0, return the best available candidate.
    best_root = min(roots_sorted, key=lambda r: abs(derivative(r)))
    return float(z_of_m(best_root, a_arr, c_arr, d_arr, n_float))


def admissible_m_from_lambda(  # noqa: N803
    lam: float,
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
) -> float:
    """
    Recover the admissible real root of z(m) = λ with positive slope.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    lam_val = float(lam)
    if not np.isfinite(lam_val):
        raise ValueError("λ must be a finite scalar.")
    k_vals = _k_values(a_arr, c_arr, d_arr, n_float)

    def equation(m_val: float) -> float:
        return z_of_m(m_val, a_arr, c_arr, d_arr, n_float) - lam_val

    grid = _augment_with_singularities(_logspace_grid(), k_vals)
    brackets = _root_brackets(equation, grid)
    roots: list[float] = []

    for left, right in brackets:
        if left == right:
            root = left
        else:
            try:
                root = _bisect(equation, left, right)
            except RuntimeError:
                continue
        if np.isfinite(root):
            roots.append(root)

    if not roots:
        raise RuntimeError("No real root found for the supplied λ.")

    def derivative(m_val: float) -> float:
        return _dz_dm(m_val, a_arr, c_arr, d_arr, n_float, k_vals)

    admissible_roots = [
        root for root in roots if derivative(root) > 0 and root < -1e-12
    ]
    if not admissible_roots:
        raise RuntimeError("No admissible root with positive slope found.")

    # Choose the root closest to the real axis (largest m, i.e., least negative).
    best_root = max(admissible_roots)
    return float(best_root)


def _normalise_order(
    order: Sequence[Sequence[int]],
    n_strata: int,
) -> list[list[int]]:
    if not order:
        raise ValueError("Order specification must be non-empty.")
    flat_indices = [idx for subset in order for idx in subset]
    if not flat_indices:
        raise ValueError("Each order subset must contain at least one index.")
    min_idx = min(flat_indices)
    if min_idx < 0:
        raise ValueError("Order indices must be non-negative or one-based.")
    offset = 1 if min_idx >= 1 else 0
    normalised: list[list[int]] = []
    for subset in order:
        converted: list[int] = []
        for idx in subset:
            adjusted = idx - offset
            if adjusted < 0 or adjusted >= n_strata:
                raise ValueError("Order index out of bounds.")
            converted.append(int(adjusted))
        normalised.append(converted)
    return normalised


def t_vec(  # noqa: N803
    lam: float,
    a: Sequence[float],
    C: Sequence[float],
    d: Sequence[float],
    N: int | float,
    c: Sequence[float],
    order: Sequence[Sequence[int]],
) -> np.ndarray:
    """
    Evaluate the t-vector associated with λ using the admissible root m(λ).
    """
    a_arr, c_weights, d_arr, n_float = _prepare_inputs(a, C, d, N)
    c_vec = np.asarray(c, dtype=np.float64)
    if c_vec.ndim != 1:
        raise ValueError("c must be a one-dimensional array-like.")
    if len(order) != c_vec.shape[0]:
        raise ValueError("Order length must match the length of c.")

    m_root = admissible_m_from_lambda(lam, a_arr, c_weights, d_arr, n_float)
    k_vals = _k_values(a_arr, c_weights, d_arr, n_float)
    denom = 1.0 + k_vals * m_root
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        raise ValueError("Encountered singularity while computing the t-vector.")
    base_terms = a_arr / denom

    normalised_order = _normalise_order(order, len(a_arr))
    t_values = np.zeros_like(c_vec, dtype=np.float64)
    for idx_r, indices in enumerate(normalised_order):
        if indices:
            t_values[idx_r] = c_vec[idx_r] * np.sum(base_terms[indices])
    return t_values


def marchenko_pastur_edges(model: MarchenkoPasturModel) -> tuple[float, float]:
    """
    Compute the theoretical support edges for a Marchenko–Pastur distribution.

    Parameters
    ----------
    model:
        MarchenkoPasturModel describing the observation regime.

    Returns
    -------
    tuple[float, float]
        Lower and upper spectral edges.
    """
    raise NotImplementedError("Marchenko–Pastur edge computation is not implemented.")


def marchenko_pastur_pdf(
    model: MarchenkoPasturModel,
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Evaluate the Marchenko–Pastur density over a grid.

    Parameters
    ----------
    model:
        MarchenkoPasturModel describing the observation regime.
    grid:
        Points at which to evaluate the density.

    Returns
    -------
    numpy.ndarray
        Density values matching the grid shape.
    """
    raise NotImplementedError("Marchenko–Pastur PDF evaluation is not implemented.")
