from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ledoit_wolf_shrinkage(
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply Ledoit–Wolf shrinkage to a sample covariance estimate.

    Parameters
    ----------
    covariance:
        Sample covariance matrix.

    Returns
    -------
    numpy.ndarray
        Shrunk covariance matrix.
    """
    raise NotImplementedError("Ledoit–Wolf shrinkage is not implemented yet.")
