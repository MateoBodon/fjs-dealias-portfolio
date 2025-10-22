from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_spectrum(
    eigenvalues: NDArray[np.float64],
    *,
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """
    Estimate a smoothed eigenvalue spectrum from noisy observations.

    Parameters
    ----------
    eigenvalues:
        Raw eigenvalues drawn from a covariance or MANOVA estimator.
    bandwidth:
        Optional smoothing bandwidth; defaults to a data-driven rule later on.

    Returns
    -------
    numpy.ndarray
        Smoothed eigenvalue estimates.
    """
    raise NotImplementedError("Spectrum estimation is not implemented yet.")
