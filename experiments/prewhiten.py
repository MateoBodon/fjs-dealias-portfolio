from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from baselines import PrewhitenResult

FACTOR_SETS: dict[str, tuple[str, ...]] = {
    "ff5mom": ("MKT", "SMB", "HML", "RMW", "CMA", "MOM"),
    "ff5": ("MKT", "SMB", "HML", "RMW", "CMA"),
    "mkt": ("MKT",),
}

FACTOR_FALLBACKS: dict[str, tuple[str, ...]] = {
    "ff5mom": ("ff5mom", "ff5", "mkt"),
    "ff5": ("ff5", "mkt"),
}


@dataclass(slots=True, frozen=True)
class PrewhitenTelemetry:
    mode_requested: str
    mode_effective: str
    factor_columns: tuple[str, ...]
    beta_abs_mean: float
    beta_abs_std: float
    beta_abs_median: float
    r2_mean: float
    r2_median: float


def identity_prewhiten_result(
    returns: pd.DataFrame,
    factor_cols: Sequence[str] | None = None,
) -> PrewhitenResult:
    from baselines import PrewhitenResult as _PrewhitenResult

    columns = list(factor_cols or [])
    assets = list(returns.columns)
    betas = pd.DataFrame(
        np.zeros((len(assets), len(columns)), dtype=np.float64),
        index=assets,
        columns=columns,
    )
    intercept = pd.Series(np.zeros(len(assets), dtype=np.float64), index=assets, name="intercept")
    r_squared = pd.Series(np.zeros(len(assets), dtype=np.float64), index=assets, name="r_squared")
    fitted = pd.DataFrame(
        np.zeros_like(returns.to_numpy(dtype=np.float64, copy=True)),
        index=returns.index,
        columns=assets,
    )
    factor_frame = pd.DataFrame(
        np.zeros((returns.shape[0], len(columns)), dtype=np.float64),
        index=returns.index,
        columns=columns,
    )
    return _PrewhitenResult(
        residuals=returns.copy(),
        betas=betas,
        intercept=intercept,
        r_squared=r_squared,
        fitted=fitted,
        factors=factor_frame,
    )


def select_prewhiten_factors(
    factors: pd.DataFrame | None,
    requested: str,
) -> tuple[str, pd.DataFrame | None]:
    if factors is None or factors.empty:
        return "off", None
    requested_key = str(requested or "off").lower()
    if requested_key == "off":
        return "off", None
    candidate_modes = FACTOR_FALLBACKS.get(requested_key, ())
    for mode in candidate_modes:
        required = FACTOR_SETS.get(mode, ())
        if all(col in factors.columns for col in required):
            subset = factors.loc[:, list(required)].copy()
            return mode, subset
    if "MKT" in factors.columns:
        return "mkt", factors.loc[:, ["MKT"]].copy()
    return "off", None


def _beta_abs_stats(betas: pd.DataFrame) -> tuple[float, float, float]:
    if betas.empty:
        return 0.0, 0.0, 0.0
    numeric = betas.select_dtypes(include=["number"])
    if numeric.empty:
        return 0.0, 0.0, 0.0
    values = np.abs(numeric.to_numpy(dtype=np.float64, copy=True))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    median = float(np.median(values))
    return mean, std, median


def compute_prewhiten_telemetry(
    whitening: PrewhitenResult,
    *,
    requested_mode: str,
    effective_mode: str,
) -> PrewhitenTelemetry:
    r2_series = whitening.r_squared if not whitening.r_squared.empty else pd.Series(dtype=np.float64)
    r2_vals = (
        r2_series.to_numpy(dtype=np.float64, copy=True) if not r2_series.empty else np.array([], dtype=np.float64)
    )
    r2_vals = r2_vals[np.isfinite(r2_vals)] if r2_vals.size else r2_vals
    r2_mean = float(np.mean(r2_vals)) if r2_vals.size else 0.0
    r2_median = float(np.median(r2_vals)) if r2_vals.size else 0.0
    beta_mean, beta_std, beta_median = _beta_abs_stats(whitening.betas)
    factor_columns = tuple(whitening.betas.columns.tolist())
    return PrewhitenTelemetry(
        mode_requested=requested_mode,
        mode_effective=effective_mode,
        factor_columns=factor_columns,
        beta_abs_mean=beta_mean,
        beta_abs_std=beta_std,
        beta_abs_median=beta_median,
        r2_mean=r2_mean,
        r2_median=r2_median,
    )


def apply_prewhitening(
    returns: pd.DataFrame,
    *,
    factors: pd.DataFrame | None,
    requested_mode: str,
) -> tuple[PrewhitenResult, PrewhitenTelemetry]:
    requested = str(requested_mode or "off").lower()
    effective_mode, factor_subset = select_prewhiten_factors(factors, requested)
    if effective_mode != "off" and factor_subset is not None:
        from baselines import prewhiten_returns

        try:
            whitening = prewhiten_returns(returns, factor_subset)
            if whitening.residuals.empty:
                raise ValueError("empty residuals")
        except ValueError:
            whitening = identity_prewhiten_result(returns, factor_subset.columns)
            effective_mode = "off"
    else:
        whitening = identity_prewhiten_result(returns)
        effective_mode = "off"
    telemetry = compute_prewhiten_telemetry(
        whitening,
        requested_mode=requested,
        effective_mode=effective_mode,
    )
    return whitening, telemetry

