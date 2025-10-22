from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf


def lw_cov(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the Ledoit–Wolf covariance estimate for observations ``x``."""

    if x.ndim != 2:
        raise ValueError("Input data must be two-dimensional.")
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(x)
    return lw.covariance_


def ledoit_wolf_shrinkage(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compatibility alias for :func:`lw_cov`."""

    return lw_cov(x)
