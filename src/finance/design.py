from __future__ import annotations

import numpy as np
import pandas as pd


def build_design_matrix(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Join returns with factor realisations on their common timeline.

    Parameters
    ----------
    returns
        Wide matrix of asset returns indexed by datetime.
    factors
        DataFrame of factor returns indexed by datetime.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame retaining rows present in both inputs.
    """

    if returns.index.inferred_type != "datetime64":
        raise ValueError("returns must use a DatetimeIndex.")
    if factors.index.inferred_type != "datetime64":
        raise ValueError("factors must use a DatetimeIndex.")

    aligned = returns.join(factors, how="inner", rsuffix="_factor").dropna(how="any")
    if aligned.empty:
        raise ValueError("No overlapping observations between returns and factors.")
    return aligned


def groups_from_weeks(index: pd.DatetimeIndex) -> np.ndarray:
    """Assign an integer group id to each timestamp based on its week.

    Parameters
    ----------
    index
        DatetimeIndex that will be grouped by ISO week starting Monday.

    Returns
    -------
    numpy.ndarray
        Integer group labels aligned with ``index``.
    """

    if index.inferred_type != "datetime64":
        raise ValueError("Index must be datetime-like.")
    periods = index.to_period("W-MON")
    mapping = {period: idx for idx, period in enumerate(pd.unique(periods))}
    groups = np.fromiter((mapping[period] for period in periods), dtype=np.intp)
    return groups
