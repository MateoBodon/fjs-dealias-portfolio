from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def evaluate_portfolio(
    returns: pd.DataFrame,
    weights: NDArray[np.float64],
) -> dict[str, float]:
    """
    Evaluate realised performance metrics for a portfolio.

    Parameters
    ----------
    returns:
        Matrix of asset returns.
    weights:
        Portfolio weights aligned with the column ordering of `returns`.

    Returns
    -------
    dict[str, float]
        Dictionary of scalar evaluation metrics.
    """
    raise NotImplementedError("Portfolio evaluation is not implemented yet.")
