from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_PSD_TOL = 1e-10


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)


def _project_box_sum(
    v: NDArray[np.float64],
    lo: float,
    hi: float,
    target: float,
) -> NDArray[np.float64]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Finite box bounds are required for projection.")
    if lo > hi:
        raise ValueError("Lower bound must not exceed upper bound.")

    n = v.size
    lower_sum = lo * n
    upper_sum = hi * n
    if target < lower_sum - 1e-12 or target > upper_sum + 1e-12:
        raise ValueError("Sum target is infeasible under the provided bounds.")

    lambda_low = np.min(v - hi)
    lambda_high = np.max(v - lo)

    def clip_with_shift(shift: float) -> NDArray[np.float64]:
        return np.clip(v - shift, lo, hi)

    for _ in range(100):
        lambda_mid = 0.5 * (lambda_low + lambda_high)
        projected = clip_with_shift(lambda_mid)
        current_sum = projected.sum()
        if abs(current_sum - target) <= 1e-12:
            return projected
        if current_sum > target:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid
    return clip_with_shift(lambda_mid)


def minvar_ridge_box(
    Sigma: NDArray[np.float64],
    *,
    box: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    sum_to_one: bool = True,
    max_iter: int = 3000,
    tol: float = 1e-7,
) -> tuple[NDArray[np.float64], dict[str, float | int | bool]]:
    """Projected-gradient minimum-variance solver with ridge and box bounds."""

    cov = np.asarray(Sigma, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Sigma must be a square matrix.")
    n_assets = cov.shape[0]
    if n_assets == 0:
        raise ValueError("Sigma must be non-empty.")
    lo, hi = box
    if lo > hi:
        raise ValueError("Invalid box bounds: lower exceeds upper.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    cov = _symmetrize(cov)
    penalized = cov + ridge * np.eye(n_assets)
    eigvals = np.linalg.eigvalsh(penalized)
    if eigvals.min() < _PSD_TOL:
        raise ValueError("Penalised covariance must be positive definite.")
    lipschitz = float(eigvals.max())
    step = 1.0 / lipschitz

    if sum_to_one:
        initial_guess = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
        w = _project_box_sum(initial_guess, lo, hi, 1.0)
    else:
        w = np.clip(np.full(n_assets, 1.0 / n_assets, dtype=np.float64), lo, hi)

    converged = False
    for iteration in range(1, max_iter + 1):
        grad = penalized @ w
        candidate = w - step * grad
        if sum_to_one:
            w_next = _project_box_sum(candidate, lo, hi, 1.0)
        else:
            w_next = np.clip(candidate, lo, hi)
        delta = np.linalg.norm(w_next - w, ord=np.inf)
        w = w_next
        if delta < tol:
            converged = True
            break

    objective = float(w @ penalized @ w)
    info = {
        "objective": objective,
        "iterations": iteration,
        "converged": converged,
        "ridge": float(ridge),
        "weight_sum": float(w.sum()),
    }
    return w, info


def turnover(w_prev: NDArray[np.float64], w_new: NDArray[np.float64]) -> float:
    """Compute one-way turnover between consecutive portfolios."""

    prev = np.asarray(w_prev, dtype=np.float64)
    new = np.asarray(w_new, dtype=np.float64)
    if prev.shape != new.shape:
        raise ValueError("Portfolios must share the same shape.")
    return 0.5 * float(np.abs(new - prev).sum())


def apply_turnover_cost(
    var_series: NDArray[np.float64] | list[float],
    w_series: list[NDArray[np.float64]],
    bps: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply turnover costs (in basis points) to a variance or PnL series."""

    if bps < 0:
        raise ValueError("Turnover cost in bps must be non-negative.")
    if len(var_series) != len(w_series):
        raise ValueError("var_series and w_series must have the same length.")

    values = np.asarray(var_series, dtype=np.float64)
    costs = np.zeros_like(values)

    if not w_series:
        return values, costs

    prev_w = np.asarray(w_series[0], dtype=np.float64)
    for idx in range(1, len(w_series)):
        current_w = np.asarray(w_series[idx], dtype=np.float64)
        costs[idx] = turnover(prev_w, current_w) * (bps / 10000.0)
        prev_w = current_w

    adjusted = values - costs
    return adjusted, costs
