from __future__ import annotations

import pandas as pd


def compute_log_returns(
    prices: pd.DataFrame,
    *,
    frequency: str = "D",
) -> pd.DataFrame:
    """
    Compute log returns from a price panel.

    Parameters
    ----------
    prices:
        Panel of adjusted prices indexed by timestamp and asset.
    frequency:
        Sampling frequency hint used for downstream annualisation.

    Returns
    -------
    pandas.DataFrame
        Log returns aligned with the input index.
    """
    raise NotImplementedError("Log return computation is not implemented yet.")
