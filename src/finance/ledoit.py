from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import LedoitWolf

from .shrinkage import _assert_psd_and_symmetric, _symmetrize, _warn_and_fill_nonfinite


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
    data = _warn_and_fill_nonfinite("lw_cov", data)
    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(data)
    sigma = _symmetrize(np.asarray(lw.covariance_, dtype=np.float64))
    _assert_psd_and_symmetric("lw_cov", sigma)
    return sigma


def ledoit_wolf_shrinkage(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Backward-compatible alias for :func:`lw_cov`."""

    return lw_cov(x)
