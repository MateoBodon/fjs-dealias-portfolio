from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PrewhitenResult",
    "load_observed_factors",
    "prewhiten_returns",
]

_FACTOR_FILENAMES: tuple[str, ...] = (
    "factors_ff5_mom.csv",
    "factors_ff5.csv",
    "factors_daily.csv",
    "factors.csv",
)

_COLUMN_ALIASES: dict[str, str] = {
    "Mkt-RF": "MKT",
    "MKTRF": "MKT",
    "MKT_RF": "MKT",
    "MKT": "MKT",
    "SMB": "SMB",
    "HML": "HML",
    "RMW": "RMW",
    "CMA": "CMA",
    "Mom": "MOM",
    "MOM": "MOM",
    "UMD": "MOM",
    "RF": "RF",
    "R_F": "RF",
    "RiskFree": "RF",
}


@dataclass(frozen=True)
class PrewhitenResult:
    residuals: pd.DataFrame
    betas: pd.DataFrame
    intercept: pd.Series
    r_squared: pd.Series
    fitted: pd.DataFrame
    factors: pd.DataFrame


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: _COLUMN_ALIASES.get(col, col) for col in frame.columns if col.lower() != "date"
    }
    result = frame.rename(columns=renamed)
    return result


def _detect_percentage_scale(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    if values.size == 0:
        return False
    max_abs = float(np.nanmax(np.abs(values)))
    return max_abs > 5.0


def _load_candidate(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    columns_lower = [c.lower() for c in frame.columns]
    if "date" in columns_lower:
        date_col = frame.columns[columns_lower.index("date")]
        index = pd.to_datetime(frame.pop(date_col))
    else:
        index = pd.to_datetime(frame.iloc[:, 0])
        frame = frame.iloc[:, 1:]
    frame = _normalise_columns(frame)
    frame.index = index.tz_localize(None)
    frame = frame.sort_index()
    if _detect_percentage_scale(frame):
        frame = frame / 100.0
    numeric = frame.select_dtypes(include=["number"])
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all")
    numeric = numeric.astype(np.float64)
    numeric.index = numeric.index.tz_localize(None)
    numeric = numeric.loc[~numeric.index.duplicated(keep="last")]
    return numeric


def load_observed_factors(
    *,
    returns: pd.DataFrame | None = None,
    path: str | Path | None = None,
    data_dir: Path | None = None,
    required: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load observed factor returns, preferring FF5+MOM datasets when available.

    When no on-disk factor dataset is present and ``returns`` is supplied, the
    fallback builds an in-sample market proxy using equal-weighted asset returns.
    """

    candidate_paths: list[Path] = []
    if path is not None:
        candidate_paths.append(Path(path))
    base_dir = Path(".") if data_dir is None else Path(data_dir)
    candidate_paths.extend(base_dir / name for name in _FACTOR_FILENAMES)

    for candidate in candidate_paths:
        loaded = _load_candidate(candidate)
        if loaded is None:
            continue
        if required:
            missing = [col for col in required if col not in loaded.columns]
            if missing:
                continue
        return loaded

    if returns is None:
        joined = ", ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(
            f"Unable to locate observed factor file. Looked in: {joined}."
        )

    if returns.empty:
        raise ValueError("returns must contain observations when building market proxy.")
    returns = returns.copy()
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must be indexed by DatetimeIndex for proxy construction.")
    numeric = returns.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("returns contain no numeric columns for proxy construction.")
    numeric = numeric.astype(np.float64)
    proxy = numeric.mean(axis=1, skipna=True)
    proxy = proxy.dropna()
    proxy.index = proxy.index.tz_localize(None)
    factors = pd.DataFrame({"MKT": proxy}, index=proxy.index)
    factors.index.name = "date"
    return factors


def _prepare_design_matrix(
    index: pd.DatetimeIndex,
    factors: pd.DataFrame,
    *,
    add_intercept: bool,
) -> tuple[np.ndarray, list[str]]:
    numeric = factors.select_dtypes(include=["number"]).astype(np.float64)
    numeric = numeric.loc[index]
    design_cols = list(numeric.columns)
    X = numeric.to_numpy(dtype=np.float64, copy=True)
    if add_intercept:
        intercept = np.ones((X.shape[0], 1), dtype=np.float64)
        X = np.column_stack([intercept, X])
    return X, design_cols


def _align_returns_factors(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    dropna: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = returns.copy()
    factors = factors.copy()
    returns.index = pd.to_datetime(returns.index).tz_localize(None)
    factors.index = pd.to_datetime(factors.index).tz_localize(None)
    joint = pd.concat([returns, factors], axis=1, join="inner")
    if joint.empty:
        raise ValueError("No overlapping observations between returns and factors.")
    if dropna:
        joint = joint.dropna(axis=0, how="any")
    returns_cols = list(returns.columns)
    factors_cols = [col for col in joint.columns if col not in returns_cols]
    aligned_returns = joint.loc[:, returns_cols].astype(np.float64)
    aligned_factors = joint.loc[:, factors_cols].astype(np.float64)
    if dropna:
        mask = aligned_returns.notna().all(axis=1) & aligned_factors.notna().all(axis=1)
        aligned_returns = aligned_returns.loc[mask]
        aligned_factors = aligned_factors.loc[mask]
    aligned_returns = aligned_returns.dropna(axis=0, how="any")
    aligned_factors = aligned_factors.loc[aligned_returns.index]
    aligned_factors = aligned_factors.dropna(axis=0, how="any")
    aligned_returns = aligned_returns.loc[aligned_factors.index]
    if aligned_returns.shape[0] <= aligned_factors.shape[1]:
        raise ValueError("Not enough observations to estimate factor loadings.")
    return aligned_returns, aligned_factors


def prewhiten_returns(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    add_intercept: bool = True,
    dropna: bool = True,
) -> PrewhitenResult:
    """
    Regress asset returns on observed factors and return residual series.
    """

    if returns.empty:
        raise ValueError("returns DataFrame must contain observations.")
    if factors.empty:
        raise ValueError("factors DataFrame must contain observations.")

    aligned_returns, aligned_factors = _align_returns_factors(
        returns, factors, dropna=dropna
    )
    index = aligned_returns.index
    X, factor_cols = _prepare_design_matrix(index, aligned_factors, add_intercept=add_intercept)
    y = aligned_returns.to_numpy(dtype=np.float64, copy=True)

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coeffs
    residuals = y - fitted

    if add_intercept:
        intercept = coeffs[0, :]
        betas = coeffs[1:, :]
    else:
        intercept = np.zeros(y.shape[1], dtype=np.float64)
        betas = coeffs

    y_centered = y - np.mean(y, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_total = np.sum(y_centered**2, axis=0)
        ss_res = np.sum(residuals**2, axis=0)
        r_squared = 1.0 - ss_res / np.clip(ss_total, 1e-12, None)
    r_squared = np.clip(r_squared, 0.0, 1.0)

    residuals_df = pd.DataFrame(residuals, index=index, columns=aligned_returns.columns)
    fitted_df = pd.DataFrame(fitted, index=index, columns=aligned_returns.columns)
    betas_df = pd.DataFrame(betas.T, index=aligned_returns.columns, columns=factor_cols)
    intercept_series = pd.Series(intercept, index=aligned_returns.columns, name="intercept")
    r2_series = pd.Series(r_squared, index=aligned_returns.columns, name="r_squared")

    return PrewhitenResult(
        residuals=residuals_df,
        betas=betas_df,
        intercept=intercept_series,
        r_squared=r2_series,
        fitted=fitted_df,
        factors=aligned_factors.loc[index],
    )
