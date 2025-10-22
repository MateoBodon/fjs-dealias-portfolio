from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_market_data(
    path: Path | str,
    *,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load market data from persistent storage into a pandas DataFrame.

    Parameters
    ----------
    path:
        Location of a CSV or Parquet file containing price or return data.
    parse_dates:
        Whether the loader should parse index-like date columns.

    Returns
    -------
    pandas.DataFrame
        Structured market data ready for further processing.
    """
    _ = Path(path)
    raise NotImplementedError("Market data loading has not been implemented yet.")
