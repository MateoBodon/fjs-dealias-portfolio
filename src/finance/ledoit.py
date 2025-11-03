from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf


def lw_cov(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the Ledoit–Wolf covariance estimate.

    Parameters
    ----------
    x
        Observation matrix shaped ``(n_samples, n_assets)``.

    Returns
    -------
    numpy.ndarray
        Shrunk covariance estimate.
    """

    data = np.asarray(x, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Input data must be two-dimensional.")
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(data)
    return np.asarray(lw.covariance_, dtype=np.float64)


def ledoit_wolf_shrinkage(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Backward-compatible alias for :func:`lw_cov`."""

    return lw_cov(x)
