from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def check_dealiased_applied(estimates: pd.DataFrame | Iterable[dict]) -> None:
    """Assert de-aliased forecasts differ from aliased when detections exist.

    This guard is meant to catch the failure mode where the risk evaluation
    inadvertently ignores detected spikes (i.e., Σ̂_DA equals the aliased
    forecast for windows where ``n_detections > 0``).

    Parameters
    ----------
    estimates
        Either a pandas DataFrame or an iterable of dict-like rows containing
        a column ``n_detections`` and one or more aliased/de-aliased forecast
        column pairs following the naming convention
        ``"{prefix}_aliased_forecast"`` and ``"{prefix}_dealiased_forecast"``
        (e.g., ``"eq_aliased_forecast"`` and ``"eq_dealiased_forecast"``).

    Raises
    ------
    AssertionError
        If any row with ``n_detections > 0`` has aliased and de-aliased
        forecasts exactly equal across all available strategy prefixes.
    """

    if not isinstance(estimates, pd.DataFrame):
        estimates = pd.DataFrame(list(estimates))

    if estimates.empty:
        return

    if "n_detections" not in estimates.columns:
        # Nothing to check
        return

    # Identify available aliased/de-aliased forecast pairs by prefix
    prefixes: list[str] = []
    for col in estimates.columns:
        if col.endswith("_aliased_forecast"):
            prefix = col[: -len("_aliased_forecast")]
            other = f"{prefix}_dealiased_forecast"
            if other in estimates.columns:
                prefixes.append(prefix)

    if not prefixes:
        return

    df = estimates.copy()
    df["n_detections"] = pd.to_numeric(df["n_detections"], errors="coerce").fillna(0)
    with_detections = df[df["n_detections"] > 0]
    if with_detections.empty:
        return

    tol = 1e-12
    for idx, row in with_detections.iterrows():
        # Check whether all available prefixes have identical forecasts
        all_equal = True
        for prefix in prefixes:
            a = row.get(f"{prefix}_aliased_forecast")
            d = row.get(f"{prefix}_dealiased_forecast")
            if a is None or d is None:
                continue
            if not (np.isfinite(a) and np.isfinite(d)):
                # Skip NaNs for this prefix; they do not prove equivalence
                continue
            if abs(float(a) - float(d)) > tol:
                all_equal = False
                break
        assert (
            not all_equal
        ), (
            "De-aliased forecasts equal to aliased in a window with detections. "
            f"row={idx}"
        )

