from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import NDArray

# ruff: noqa: N803

ArrayLike = Sequence[float] | NDArray[np.float64]

_MP_CACHE_MAX = 512
_MP_CACHE_DIR_ENV = os.environ.get("MP_EDGE_CACHE_DIR")
_MP_CACHE_DIR = Path(_MP_CACHE_DIR_ENV).expanduser() if _MP_CACHE_DIR_ENV else None
if _MP_CACHE_DIR:
    _MP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MP_CACHE: OrderedDict[str, float] = OrderedDict()


def _cache_get(key: str) -> float | None:
    if key in _MP_CACHE:
        value = _MP_CACHE.pop(key)
        _MP_CACHE[key] = value
        return value
    if _MP_CACHE_DIR:
        cache_file = _MP_CACHE_DIR / f"{key}.npz"
        if cache_file.exists():
            try:
                payload = np.load(cache_file)
                value = float(payload["value"])
                _cache_set(key, value)
                return value
            except Exception:
                try:
                    cache_file.unlink()
                except OSError:
                    pass
    return None


def _cache_set(key: str, value: float) -> None:
    if key in _MP_CACHE:
        _MP_CACHE.pop(key)
    _MP_CACHE[key] = float(value)
    if len(_MP_CACHE) > _MP_CACHE_MAX:
        _MP_CACHE.popitem(last=False)
    if _MP_CACHE_DIR:
        cache_file = _MP_CACHE_DIR / f"{key}.npz"
        try:
            np.savez_compressed(cache_file, value=float(value))
        except Exception:
            pass


def _hash_arrays(*arrays: ArrayLike | None) -> str:
    hasher = hashlib.sha1()
    for arr in arrays:
        if arr is None:
            hasher.update(b"none")
            continue
        arr_np = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        hasher.update(str(arr_np.shape).encode("utf-8"))
        hasher.update(arr_np.tobytes())
    return hasher.hexdigest()


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


def _prepare_cs(
    Cs: ArrayLike | None,
    template: np.ndarray,
) -> np.ndarray:
    if Cs is None:
        return np.zeros_like(template, dtype=np.float64)
    cs_arr = np.asarray(Cs, dtype=np.float64)
    if cs_arr.ndim != 1:
        raise ValueError("Cs must be one-dimensional.")
    if cs_arr.shape != template.shape:
        raise ValueError("Cs must match the shape of a.")
    return cs_arr


def estimate_Cs_from_MS(  # noqa: N802
    MS_list: Sequence[ArrayLike],
    d_list: Sequence[float],
    c_list: Sequence[float],
    drop_top: int = 1,
) -> np.ndarray:
    """
    Estimate trace-based noise plug-ins C_s from the supplied mean squares.

    Parameters
    ----------
    MS_list
        Sequence of mean square matrices (one per stratum).
    d_list
        Degrees of freedom associated with each stratum (used for validation).
    c_list
        Design coefficients associated with each stratum.
    drop_top
        Number of leading eigenvalues to discard before averaging.
    """

    if drop_top < 0:
        raise ValueError("drop_top must be non-negative.")
    ms_arrays = [np.asarray(ms, dtype=np.float64) for ms in MS_list]
    if not ms_arrays:
        raise ValueError("MS_list must contain at least one matrix.")

    first_shape = ms_arrays[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError("Mean squares must be square matrices.")
    for ms in ms_arrays[1:]:
        if ms.shape != first_shape:
            raise ValueError("All mean squares must share the same shape.")

    d_arr = np.asarray(d_list, dtype=np.float64).reshape(-1)
    c_arr = np.asarray(c_list, dtype=np.float64).reshape(-1)
    if d_arr.size != len(ms_arrays) or c_arr.size != len(ms_arrays):
        raise ValueError("d_list and c_list must match the number of mean squares.")
    if np.any(d_arr <= 0):
        raise ValueError("Degrees of freedom must be positive.")

    p = first_shape[0]
    drop_count = min(int(drop_top), max(p - 1, 0))

    sigma_estimates = np.zeros(len(ms_arrays), dtype=np.float64)
    for idx, ms in enumerate(ms_arrays):
        sym = 0.5 * (ms + ms.T)
        eigenvalues = np.linalg.eigvalsh(sym)
        if eigenvalues.size == 0:
            raise ValueError("Encountered empty eigenvalue spectrum.")
        discard = min(drop_count, eigenvalues.size - 1)
        if discard > 0:
            trimmed = eigenvalues[:-discard]
        else:
            trimmed = eigenvalues
        sigma_sq = float(np.mean(trimmed))
        sigma_estimates[idx] = max(sigma_sq, 0.0)

    cs_values = np.zeros(len(ms_arrays), dtype=np.float64)
    for idx in range(len(ms_arrays) - 1, -1, -1):
        cs_values[idx] = float(np.dot(c_arr[idx:], sigma_estimates[idx:]))
    return cs_values


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
    Cs: ArrayLike | None = None,
) -> float:
    """
    Evaluate the closed-form Marčenko–Pastur z(m) transform.

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
    cs_arr = _prepare_cs(Cs, a_arr)
    # Denominator uses Cs when provided (non-zero); otherwise fall back to design c
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_arr
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    denom = 1.0 + k_vals * m_val
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        raise ValueError("Denominator becomes singular for the supplied m.")
    numerators = c_arr * a_arr + cs_arr
    contributions = numerators / denom
    return float(-1.0 / m_val + np.sum(contributions))


def z0(  # noqa: N802
    m: float,
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
    Cs: ArrayLike | None = None,
) -> float:
    """Balanced one-way z0(m) in closed form.

    This is a thin wrapper over :func:`z_of_m` specialised for the
    balanced MANOVA setting used throughout the codebase.

    Parameters
    ----------
    m
        Real argument for the Stieltjes transform branch (negative real line).
    a, C, d, N, Cs
        Design parameters matching :func:`z_of_m`.
    """
    return z_of_m(m, a, C, d, N, Cs)


def _dz_dm(  # noqa: N803
    m: float,
    k_vals: np.ndarray,
    numerators: np.ndarray,
) -> float:
    denom = 1.0 + k_vals * m
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        return float("nan")
    term1 = 1.0 / (m * m)
    term2 = float(np.sum(numerators * k_vals / (denom**2)))
    return term1 - term2


def _d2z_dm2(  # noqa: N803
    m: float,
    k_vals: np.ndarray,
    numerators: np.ndarray,
) -> float:
    denom = 1.0 + k_vals * m
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        return float("nan")
    term1 = -2.0 / (m**3)
    term2 = float(np.sum(2.0 * numerators * (k_vals**2) / (denom**3)))
    return term1 + term2


def z0_prime(  # noqa: N802
    m: float,
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
    Cs: ArrayLike | None = None,
) -> float:
    """Closed-form first derivative z0'(m) for balanced one-way design.

    Uses the identity
    z'(m) = 1/m^2 - sum_s ( (c_s a_s + Cs_s) k_s / (1 + k_s m)^2 ),
    where k_s = (N/d_s) a_s C_s.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    cs_arr = _prepare_cs(Cs, a_arr)
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_arr
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    numerators = c_arr * a_arr + cs_arr
    return _dz_dm(float(m), k_vals, numerators)


def z0_double_prime(  # noqa: N802
    m: float,
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
    Cs: ArrayLike | None = None,
) -> float:
    """Closed-form second derivative z0''(m) for balanced one-way design.

    Uses the identity
    z''(m) = -2/m^3 + sum_s ( 2 (c_s a_s + Cs_s) k_s^2 / (1 + k_s m)^3 ).
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    cs_arr = _prepare_cs(Cs, a_arr)
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_arr
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    numerators = c_arr * a_arr + cs_arr
    return _d2z_dm2(float(m), k_vals, numerators)


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


def _newton_refine(
    x0: float,
    f: Callable[[float], float],
    fp: Callable[[float], float],
    *,
    max_iter: int = 8,
    tol: float = 1e-14,
) -> float:
    """One-step Newton refinement with simple safeguards.

    If the derivative is ill-conditioned or an update produces NaNs, the
    current iterate is returned unchanged.
    """
    x = float(x0)
    for _ in range(max_iter):
        fx = f(x)
        fpx = fp(x)
        if not (np.isfinite(fx) and np.isfinite(fpx)):
            break
        if abs(fpx) < 1e-18:
            break
        step = fx / fpx
        x_new = x - step
        if not np.isfinite(x_new):
            break
        if abs(x_new - x) < tol:
            return float(x_new)
        x = x_new
    return float(x)


def _stationary_points(
    k_vals: np.ndarray,
    numerators: np.ndarray,
) -> tuple[list[float], Callable[[float], float], Callable[[float], float]]:
    """Locate stationary points of z(m) by bracketing zeros of z'(m)."""

    def derivative(m_val: float) -> float:
        return _dz_dm(m_val, k_vals, numerators)

    def curvature(m_val: float) -> float:
        return _d2z_dm2(m_val, k_vals, numerators)

    grid = _augment_with_singularities(_logspace_grid(), k_vals)
    brackets = _brackets_sign_change(derivative, grid, k_vals)
    roots: list[float] = []
    for left, right in brackets:
        if left == right:
            root = left
        else:
            try:
                root = _bisect(derivative, left, right)
            except RuntimeError:
                continue
        # Optional Newton polish
        root = _newton_refine(root, derivative, curvature)
        if np.isfinite(root):
            roots.append(float(root))
    return roots, derivative, curvature


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


def _crosses_pole(m1: float, m2: float, k_vals: np.ndarray) -> bool:
    """Return True if the interval [m1, m2] crosses a pole 1 + k m = 0."""
    for k in k_vals:
        if k <= 0:
            continue
        s1 = 1.0 + k * m1
        s2 = 1.0 + k * m2
        if s1 == 0.0 or s2 == 0.0 or (s1 < 0.0 and s2 > 0.0) or (s1 > 0.0 and s2 < 0.0):
            return True
    return False


def _root_brackets(
    func: Callable[[float], float],
    points: np.ndarray,
    k_vals: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    brackets: list[tuple[float, float]] = []
    prev_x: float | None = None
    prev_val: float | None = None
    for x in points:
        raw = func(x)
        # Treat infinities as very large finite values preserving the sign
        if not np.isfinite(raw):
            if np.isnan(raw):
                prev_x = None
                prev_val = None
                continue
            val = float(np.sign(raw)) * 1e300
        else:
            val = float(raw)
        if prev_x is not None and prev_val is not None:
            if val == 0.0:
                if k_vals is None or not _crosses_pole(prev_x, x, k_vals):
                    brackets.append((x, x))
            elif prev_val == 0.0:
                if k_vals is None or not _crosses_pole(prev_x, x, k_vals):
                    brackets.append((prev_x, prev_x))
            elif val * prev_val < 0:
                if k_vals is None or not _crosses_pole(prev_x, x, k_vals):
                    brackets.append((prev_x, x))
    prev_x = x
    prev_val = val
    return brackets


def _brackets_sign_change(
    func: Callable[[float], float],
    points: np.ndarray,
    k_vals: np.ndarray,
) -> list[tuple[float, float]]:
    """Find sign-change brackets while guarding against poles.

    This is a more permissive variant used for robust edge/root finding.
    """
    brackets: list[tuple[float, float]] = []
    prev_x: float | None = None
    prev_v: float | None = None
    for x in points:
        try:
            v = float(func(x))
        except Exception:
            v = float("nan")
        if not np.isfinite(v):
            # Skip NaNs only; keep ±inf by clipping to large magnitude preserving sign
            if np.isnan(v):
                prev_x = None
                prev_v = None
                continue
            v = float(np.sign(v)) * 1e300
        if prev_x is not None and prev_v is not None:
            if v == 0.0 or prev_v == 0.0 or (v * prev_v < 0.0):
                if not _crosses_pole(prev_x, x, k_vals):
                    brackets.append((prev_x, x))
        prev_x = x
        prev_v = v
    return brackets


def mp_edge(  # noqa: N803
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
    Cs: ArrayLike | None = None,
) -> float:
    """
    Locate the upper bulk edge of the Marčenko--Pastur distribution.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    cs_arr = _prepare_cs(Cs, a_arr)
    cache_key = f"mp_edge:{_hash_arrays(a_arr, c_arr, d_arr, np.asarray([n_float]), cs_arr)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    value = _mp_edge_impl(a_arr, c_arr, d_arr, n_float, cs_arr)
    _cache_set(cache_key, value)
    return value


def _mp_edge_impl(
    a_arr: np.ndarray,
    c_arr: np.ndarray,
    d_arr: np.ndarray,
    n_float: float,
    cs_arr: np.ndarray,
) -> float:
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_arr
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    numerators = c_arr * a_arr + cs_arr

    roots, derivative, curvature = _stationary_points(k_vals, numerators)

    if not roots:
        # fall back: pick point with minimal |z'(m)| on the search grid
        grid = _augment_with_singularities(_logspace_grid(), k_vals)
        values = [
            (abs(derivative(point)), point)
            for point in grid
            if np.isfinite(derivative(point))
        ]
        if not values:
            raise RuntimeError("Unable to locate a stationary point for z(m).")
        _, candidate = min(values, key=lambda pair: pair[0])
        roots = [candidate]

    # Prefer roots with negative second derivative (concave maximum).
    roots_sorted = sorted(roots, reverse=True)
    for root in roots_sorted:
        curvature_val = curvature(root)
        if np.isfinite(curvature_val) and curvature_val < 0:
            return float(z_of_m(root, a_arr, c_arr, d_arr, n_float, cs_arr))

    # If no root satisfies curvature < 0, return the best available candidate.
    best_root = min(roots_sorted, key=lambda r: abs(derivative(r)))
    return float(z_of_m(best_root, a_arr, c_arr, d_arr, n_float, cs_arr))


def m_edge(  # noqa: N803
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
    Cs: ArrayLike | None = None,
) -> float:
    """Return m_plus where z'(m_plus)=0 and z''(m_plus)<0 (upper edge).

    Parameters mirror :func:`mp_edge`. This provides the stationary m used to
    compute the upper spectral edge via ``z_plus = z0(m_plus)``.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    cs_arr = _prepare_cs(Cs, a_arr)
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_arr
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    numerators = c_arr * a_arr + cs_arr
    roots, derivative, curvature = _stationary_points(k_vals, numerators)
    if not roots:
        raise RuntimeError("No stationary point located for z(m).")
    roots_sorted = sorted(roots, reverse=True)
    for root in roots_sorted:
        if curvature(root) < 0:
            return float(root)
    return float(roots_sorted[0])


def admissible_m_from_lambda(  # noqa: N803
    lam: float,
    a: ArrayLike,
    C: ArrayLike,
    d: ArrayLike,
    N: int | float,
    Cs: ArrayLike | None = None,
) -> float:
    """
    Recover the admissible real root of z(m) = λ with positive slope.
    """
    a_arr, c_arr, d_arr, n_float = _prepare_inputs(a, C, d, N)
    cs_arr = _prepare_cs(Cs, a_arr)
    lam_val = float(lam)
    if not np.isfinite(lam_val):
        raise ValueError("λ must be a finite scalar.")
    cache_key = f"admiss:{_hash_arrays(a_arr, c_arr, d_arr, np.asarray([n_float, lam_val]), cs_arr)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    value = _admissible_m_from_lambda_impl(lam_val, a_arr, c_arr, d_arr, n_float, cs_arr)
    _cache_set(cache_key, value)
    return value


def _admissible_m_from_lambda_impl(
    lam_val: float,
    a_arr: np.ndarray,
    c_arr: np.ndarray,
    d_arr: np.ndarray,
    n_float: float,
    cs_arr: np.ndarray,
) -> float:
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_arr
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    numerators = c_arr * a_arr + cs_arr

    def equation(m_val: float) -> float:
        return z_of_m(m_val, a_arr, c_arr, d_arr, n_float, cs_arr) - lam_val

    grid = _augment_with_singularities(_logspace_grid(), k_vals)
    brackets = _brackets_sign_change(equation, grid, k_vals)
    roots: list[float] = []

    for left, right in brackets:
        if left == right:
            root = left
        else:
            try:
                root = _bisect(equation, left, right)
            except RuntimeError:
                continue
        # Optional Newton polish using derivative of z(m)
        root = _newton_refine(root, equation, lambda m_val: _dz_dm(m_val, k_vals, numerators))
        if np.isfinite(root):
            roots.append(root)

    if not roots:
        raise RuntimeError("No real root found for the supplied λ.")

    def derivative(m_val: float) -> float:
        return _dz_dm(m_val, k_vals, numerators)

    admissible_roots = [
        root for root in roots if derivative(root) > 0 and root < -1e-12
    ]
    if not admissible_roots:
        raise RuntimeError("No admissible root with positive slope found.")

    # Choose the root closest to the real axis (largest m, i.e., least negative).
    best_root = max(admissible_roots)

    # Diagnostics: warn if very close to the edge stationary point or a singularity.
    try:
        m_plus = m_edge(a_arr, c_arr, d_arr, n_float, cs_arr)
        if abs(best_root - m_plus) <= max(1e-8, 1e-6 * abs(m_plus)):
            warnings.warn(
                "Root m(λ) lies extremely close to the spectral edge; results may be unstable.",
                RuntimeWarning,
            )
    except Exception:
        pass
    denom = 1.0 + k_vals * best_root
    if np.min(np.abs(denom)) < 1e-8:
        warnings.warn(
            "Root m(λ) is close to a pole of z(m); numerical conditioning is poor.",
            RuntimeWarning,
        )
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
    Cs: Sequence[float] | None = None,
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

    cs_arr = _prepare_cs(Cs, a_arr)
    m_root = admissible_m_from_lambda(lam, a_arr, c_weights, d_arr, n_float, cs_arr)
    denom_weights = cs_arr if np.any(np.abs(cs_arr) > 0.0) else c_weights
    k_vals = _k_values(a_arr, denom_weights, d_arr, n_float)
    denom = 1.0 + k_vals * m_root
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        raise ValueError("Encountered singularity while computing the t-vector.")
    numerators = c_weights * a_arr + cs_arr
    base_terms = np.zeros_like(a_arr, dtype=np.float64)
    mask = np.abs(c_weights) > 1e-12
    base_terms[mask] = numerators[mask] / (c_weights[mask] * denom[mask])

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


def scale_Cs(Cs: ArrayLike, alpha: float) -> np.ndarray:  # noqa: N802
    """
    Return a scaled copy of the Cs plug-ins by factor ``alpha``.

    Parameters
    ----------
    Cs
        Sequence of Cs values (one per stratum).
    alpha
        Scaling factor; must be finite. Negative factors are allowed for diagnostics
        but typical usage is alpha in (0, +inf).
    """

    cs_arr = np.asarray(Cs, dtype=np.float64)
    if cs_arr.ndim != 1:
        raise ValueError("Cs must be one-dimensional.")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be a finite scalar.")
    return (float(alpha) * cs_arr).astype(np.float64, copy=False)
