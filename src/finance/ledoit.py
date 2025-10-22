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

    if x.ndim != 2:
        raise ValueError("Input data must be two-dimensional.")
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(x)
    return np.asarray(lw.covariance_, dtype=np.float64)


def ledoit_wolf_shrinkage(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Backward-compatible alias for :func:`lw_cov`."""

    return lw_cov(x)
