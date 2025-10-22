from __future__ import annotations

import pandas as pd


def build_design_matrix(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the design matrix used for factor regressions or MANOVA summarisation.

    Parameters
    ----------
    returns:
        Asset return panel.
    factors:
        Factor realisations to align with the asset returns.

    Returns
    -------
    pandas.DataFrame
        Feature matrix ready for estimation.
    """
    raise NotImplementedError(
        "Factor design matrix construction is not implemented yet."
    )
