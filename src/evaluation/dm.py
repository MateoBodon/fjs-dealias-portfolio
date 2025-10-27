from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def _newey_west_long_run_variance(
    diffs: NDArray[np.float64],
    lags: int,
) -> float:
    n = diffs.size
    if n == 0:
        return float("nan")
    centered = diffs - diffs.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    spectral = gamma0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        spectral += 2.0 * weight * gamma
    return spectral


def dm_test(
    err1: Iterable[float] | NDArray[np.float64],
    err2: Iterable[float] | NDArray[np.float64],
    *,
    h: int = 1,
    use_nw: bool = True,
    lags: int | None = None,
) -> tuple[float, float]:
    """Diebold–Mariano test for equal predictive accuracy."""

    x = np.asarray(list(err1), dtype=np.float64).ravel()
    y = np.asarray(list(err2), dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("err1 and err2 must have the same length.")

    mask = np.isfinite(x) & np.isfinite(y)
    diffs = x[mask] - y[mask]
    n = diffs.size
    if n <= 1:
        return float("nan"), float("nan")
    if h <= 0:
        raise ValueError("Forecast horizon h must be positive.")
    if n <= h:
        return float("nan"), float("nan")

    mean_diff = float(diffs.mean())
    if use_nw:
        if lags is None:
            lags = int(np.floor(n ** (1.0 / 3.0)))
        lags = max(int(lags), 0)
        lags = min(lags, n - 1)
        long_run_var = _newey_west_long_run_variance(diffs, lags)
    else:
        long_run_var = float(np.var(diffs, ddof=1))

    if not np.isfinite(long_run_var) or long_run_var <= 0.0:
        return float("nan"), float("nan")
    variance_scale = float(np.var(diffs, ddof=0))
    tol = 1e-12 * max(1.0, variance_scale)
    if long_run_var <= tol:
        return float("nan"), float("nan")

    dm_stat = mean_diff / sqrt(long_run_var / n)

    # Small-sample correction (Harvey, Leybourne, and Newbold 1997)
    adjustment = sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= adjustment

    from scipy.stats import t as student_t  # defer import

    dof = n - 1
    p_value = 2.0 * (1.0 - student_t.cdf(abs(dm_stat), df=dof))
    return float(dm_stat), float(p_value)
