"""
Grouping utilities for replicated covariance estimation.

Future milestones will populate concrete grouping logic (Day-of-Week,
volatility state buckets).  This file currently exposes the signatures so that
downstream modules and tests can be scaffolded incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

__all__ = ["VolStateConfig", "group_dayofweek", "group_volstate"]


@dataclass(frozen=True)
class VolStateConfig:
    """Configuration for the volatility state grouping."""

    n_bins: int = 4
    method: str = "quantile"


def group_dayofweek(returns: pd.DataFrame) -> pd.Series:
    """
    Assign a Day-of-Week group label to each row in the returns DataFrame.

    Parameters
    ----------
    returns
        DataFrame indexed by date with asset columns.
    """

    raise NotImplementedError("Day-of-Week grouping will be implemented in Sprint 1.")


def group_volstate(
    volatility_proxy: Iterable[float] | pd.Series,
    *,
    config: VolStateConfig | None = None,
) -> pd.Series:
    """
    Bucket each date according to a volatility proxy (e.g., VIX or realised vol).
    """

    raise NotImplementedError("Volatility-state grouping will be implemented in Sprint 1.")
