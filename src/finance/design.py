from __future__ import annotations

import numpy as np
import pandas as pd


def build_design_matrix(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Join returns with factor realisations on their common timeline."""

    if returns.index.inferred_type != "datetime64":
        raise ValueError("returns must use a DatetimeIndex.")
    if factors.index.inferred_type != "datetime64":
        raise ValueError("factors must use a DatetimeIndex.")

    aligned = returns.join(factors, how="inner", rsuffix="_factor").dropna(how="any")
    if aligned.empty:
        raise ValueError("No overlapping observations between returns and factors.")
    return aligned


def groups_from_weeks(index: pd.DatetimeIndex) -> np.ndarray:
    """Assign an integer group id to each timestamp based on its week."""

    if index.inferred_type != "datetime64":
        raise ValueError("Index must be datetime-like.")
    periods = index.to_period("W-MON")
    mapping = {period: idx for idx, period in enumerate(pd.unique(periods))}
    groups = np.fromiter((mapping[period] for period in periods), dtype=np.intp)
    return groups
