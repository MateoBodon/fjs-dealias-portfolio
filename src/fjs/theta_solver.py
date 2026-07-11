from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fjs.mp import t_vec


@dataclass(frozen=True)
class ThetaSolverParams:
    """Closed-form parameters required for the θ root-finding routine."""

    C: np.ndarray
    d: np.ndarray
    N: float
    c: np.ndarray
    order: list[list[int]]
    Cs: np.ndarray | None
    eps: float
    delta: float
    grid_size: int = 64
    tol: float = 1e-8
    max_iter: int = 60


def _normalise_angle(theta: float) -> float:
    """Return θ reduced to the principal interval [0, 2π)."""
    tau = 2.0 * math.pi
    return float((theta % tau + tau) % tau)


def solve_theta_for_t2_zero(
    lambda_hat: float,
    params: ThetaSolverParams,
) -> float | None:
    """
    Solve for θ such that t₂(λ̂, θ) = 0 for k=2 balanced designs.

    Parameters
    ----------
    lambda_hat:
        Eigenvalue candidate associated with a detected spike.
    params:
        Closed-form parameters describing the design (d, c, Cs, etc.).

    Returns
    -------
    float | None
        Angle θ in radians (normalised to [0, 2π)) when successful; otherwise ``None``.
    """

    bulk_fallback = np.asarray(params.C, dtype=np.float64)
    d = np.asarray(params.d, dtype=np.float64)
    c = np.asarray(params.c, dtype=np.float64)
    explicit_scales = (
        None if params.Cs is None else np.asarray(params.Cs, dtype=np.float64)
    )

    if bulk_fallback.shape[0] != 2 or d.shape[0] != 2 or c.shape[0] != 2:
        return None
    if params.order and len(params.order) != 2:
        return None

    eps = float(max(params.eps, 1e-10))
    tol = float(max(params.tol, 1e-12))
    max_iter = max(int(params.max_iter), 12)
    grid_size = max(int(params.grid_size), 32)

    def _t_off(theta_val: float) -> tuple[float, np.ndarray | None]:
        a_vec = [math.cos(theta_val), math.sin(theta_val)]
        try:
            t_vals = t_vec(
                float(lambda_hat),
                a_vec,
                bulk_fallback.tolist(),
                d.tolist(),
                float(params.N),
                c.tolist(),
                params.order,
                None if explicit_scales is None else explicit_scales.tolist(),
            )
        except Exception:
            return float("nan"), None
        t_array = np.asarray(t_vals, dtype=np.float64)
        if t_array.shape[0] != 2:
            return float("nan"), None
        return float(t_array[1]), t_array

    theta_grid = np.linspace(
        0.0, 2.0 * math.pi, num=grid_size, endpoint=False, dtype=np.float64
    )
    f_values = []
    for theta in theta_grid:
        value, _ = _t_off(theta)
        f_values.append(value)

    def _find_brackets() -> list[tuple[float, float, float, float]]:
        tau = 2.0 * math.pi
        brackets: list[tuple[float, float, float, float]] = []
        for idx in range(grid_size):
            theta_left = float(theta_grid[idx])
            theta_right = float(theta_grid[(idx + 1) % grid_size])
            value_left = f_values[idx]
            value_right = f_values[(idx + 1) % grid_size]
            if not (np.isfinite(value_left) and np.isfinite(value_right)):
                continue
            if value_left == 0.0:
                brackets.append((theta_left, theta_left, value_left, value_left))
                continue
            if value_right == 0.0:
                brackets.append((theta_right, theta_right, value_right, value_right))
                continue
            if value_left * value_right < 0.0:
                # Ensure the interval is monotonically increasing
                if theta_right <= theta_left:
                    theta_right += tau
                brackets.append((theta_left, theta_right, value_left, value_right))
        return brackets

    for theta_left, theta_right, f_left, _ in _find_brackets():
        if theta_left == theta_right:
            root = theta_left
        else:
            left = float(theta_left)
            right = float(theta_right)
            f_left_val = float(f_left)
            root = None
            for _ in range(max_iter):
                mid = 0.5 * (left + right)
                f_mid, _ = _t_off(mid)
                if not np.isfinite(f_mid):
                    root = None
                    break
                if abs(f_mid) <= tol:
                    root = mid
                    break
                if f_left_val * f_mid < 0.0:
                    right = mid
                else:
                    left = mid
                    f_left_val = f_mid
                root = mid
        if root is None:
            continue

        theta_root = _normalise_angle(root)
        f_root, t_values_root = _t_off(theta_root)
        if not np.isfinite(f_root) or t_values_root is None:
            continue
        if abs(f_root) > max(eps, tol * 10.0):
            continue
        # A positive target t-value is required for mu=lambda/t to represent a
        # positive covariance eigenvalue. The opposite root is not admissible.
        if float(t_values_root[0]) <= eps:
            continue
        return theta_root

    return None
