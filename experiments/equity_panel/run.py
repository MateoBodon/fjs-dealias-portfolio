# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
# Ensure both project root and src are importable when running as a script
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Progress bar (fallback to no-op if unavailable)
try:  # pragma: no cover - UI nicety
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - best-effort import

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


from finance.eval import (
    oos_variance_forecast,
    risk_metrics,
    rolling_windows,
    variance_forecast_from_components,
)
from finance.io import load_prices_csv, to_daily_returns, load_returns_csv
from finance.portfolio import MinVarMemo, apply_turnover_cost, minvar_ridge_box, turnover
from finance.portfolios import equal_weight
from finance.robust import huberize, winsorize
from finance.returns import balance_weeks
from fjs.balanced import mean_squares
from fjs.balanced_nested import mean_squares_nested
from fjs.dealias import dealias_search
from fjs.gating import count_isolated_outliers, lookup_calibrated_delta, select_top_k
from fjs.mp import estimate_Cs_from_MS, mp_edge
from fjs.spectra import plot_spectrum_with_edges, plot_spike_timeseries
from fjs.robust import edge_from_scatter, huber_scatter, tyler_scatter
from plotting import (
    e1_plot_spectrum_with_mp,
    e2_plot_spike_timeseries,
    e3_plot_var_mse,
    e4_plot_var_coverage,
)
from meta.cache import load_window, save_window, window_key
from meta import runtime
from meta.run_meta import code_signature, write_run_meta
from evaluation import check_dealiased_applied
from evaluation.evaluate import (
    iqr as eval_iqr,
    sign_test_pvalue as eval_sign_test_p,
    plot_variance_error_panel as eval_plot_var_panel,
    plot_coverage_error as eval_plot_cov_err,
    build_metrics_summary as eval_build_metrics_summary,
    qlike as eval_qlike,
    alignment_diagnostics as eval_alignment_diag,
)
from data.panels import (
    BalancedPanel,
    PanelManifest,
    build_balanced_weekday_panel,
    hash_daily_returns,
    load_balanced_panel,
    save_balanced_panel,
    write_manifest as write_panel_manifest,
)

DEFAULT_CONFIG = {
    "data_path": "data/returns_daily.csv",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "window_weeks": 156,
    "horizon_weeks": 4,
    "output_dir": "experiments/equity_panel/outputs",
    "dealias_delta_frac": 0.02,
    "signed_a": True,
    "cs_drop_top_frac": 0.05,
    "cs_sensitivity_frac": 0.0,
    "target_component": 0,
    "a_grid": 180,
    "dealias_eps": 0.03,
    "off_component_leak_cap": 10.0,
    "energy_min_abs": 1e-6,
    "partial_week_policy": "drop",
    "design": "oneway",
    "nested_replicates": 5,
    "oneway_a_solver": "auto",
    "estimator": "dealias",
    "factor_csv": None,
    "minvar_ridge": 1e-4,
    "minvar_box": [0.0, 0.1],
    "turnover_cost_bps": 5.0,
    "minvar_condition_cap": 1e9,
    "edge_mode": "scm",
    "edge_huber_c": 1.5,
    "gating": {
        "enable": True,
        "q_max": 2,
        "require_isolated": True,
        "mode": "fixed",
        "calibration_path": "calibration/edge_delta_thresholds.json",
    },
    "alignment_top_p": 3,
}

def _parse_box_bounds(bounds: Any) -> tuple[float, float]:
    """Normalise min-variance box bounds into a (lo, hi) tuple."""

    if bounds is None:
        return (0.0, 0.1)
    if isinstance(bounds, str):
        parts = [item.strip() for item in bounds.split(",") if item.strip()]
    elif isinstance(bounds, Iterable):
        parts = [str(item).strip() for item in bounds]
    else:
        raise ValueError("minvar_box must be provided as 'lo,hi' or a sequence.")
    if len(parts) != 2:
        raise ValueError("minvar_box requires exactly two values (lo, hi).")
    lo, hi = float(parts[0]), float(parts[1])
    if lo > hi:
        raise ValueError("minvar_box lower bound must not exceed the upper bound.")
    return lo, hi


CODE_SIGNATURE = code_signature()


@dataclass
class PreparedWindowStats:
    y_fit: np.ndarray
    groups: np.ndarray
    stats: dict[str, Any]
    ms_list: list[np.ndarray]
    sigma_list: list[np.ndarray]
    design_override: dict[str, Any] | None
    design_c: np.ndarray
    design_d: np.ndarray
    design_N: float
    design_order: list[list[int]]
    nested_replicates: int | None
    cache_payload: dict[str, Any] | None


def _load_prepared_from_cache(
    cached_stats: dict[str, Any] | None,
    design_mode: str,
    y_fit_raw: np.ndarray,
    code_signature_hash: str | None,
    expected_nested_replicates: int | None,
) -> PreparedWindowStats | None:
    if cached_stats is None:
        return None
    try:
        if cached_stats.get("design_mode") != design_mode:
            return None
        cached_signature = cached_stats.get("code_signature")
        if code_signature_hash and cached_signature != code_signature_hash:
            return None
        if cached_signature and not code_signature_hash:
            return None
        component_count = int(cached_stats.get("components", 0))
        if component_count <= 0:
            return None
        cached_nested = cached_stats.get("nested_replicates")
        if expected_nested_replicates is not None:
            if cached_nested is None:
                return None
            if int(cached_nested) != int(expected_nested_replicates):
                return None
        elif cached_nested not in (None, 0):
            # Current run expects oneway but cache stored nested stats.
            return None
        valid_indices = np.asarray(cached_stats.get("valid_indices"), dtype=np.intp)
        if valid_indices.ndim != 1 or valid_indices.size == 0:
            return None
        if valid_indices.max(initial=-1) >= y_fit_raw.shape[0]:
            return None
        y_fit = y_fit_raw[valid_indices]
        groups = np.arange(valid_indices.size, dtype=np.intp)

        stats_local: dict[str, Any] = {}
        ms_list: list[np.ndarray] = []
        sigma_list: list[np.ndarray] = []
        for idx in range(component_count):
            ms_key = f"MS{idx + 1}"
            sigma_key = f"Sigma{idx + 1}_hat"
            if ms_key not in cached_stats or sigma_key not in cached_stats:
                return None
            ms_arr = np.asarray(cached_stats[ms_key], dtype=np.float64)
            sigma_arr = np.asarray(cached_stats[sigma_key], dtype=np.float64)
            ms_list.append(ms_arr)
            sigma_list.append(sigma_arr)
            stats_local[ms_key] = ms_arr
            stats_local[sigma_key] = sigma_arr

        for key in ("I", "J", "n", "replicates"):
            if key in cached_stats:
                stats_local[key] = int(cached_stats[key])

        design_c = np.asarray(cached_stats.get("design_c"), dtype=np.float64)
        design_d = np.asarray(cached_stats.get("design_d"), dtype=np.float64)
        design_N = float(cached_stats.get("design_N", 0.0))
        if design_c.size == 0 and "J" in stats_local:
            design_c = np.array([float(stats_local["J"]), 1.0], dtype=np.float64)
        if design_d.size == 0 and "I" in stats_local and "n" in stats_local:
            design_d = np.array(
                [float(stats_local["I"] - 1), float(stats_local["n"] - stats_local["I"])],
                dtype=np.float64,
            )
        if design_N <= 0.0 and "J" in stats_local:
            design_N = float(stats_local["J"])
        design_order = cached_stats.get("design_order")
        if not isinstance(design_order, list):
            design_order = [[idx + 1 for idx in range(component_count)]]
        design_override = None
        if design_mode == "nested":
            design_override = {
                "c": design_c,
                "C": np.ones_like(design_c, dtype=np.float64),
                "d": design_d,
                "N": design_N,
                "order": design_order,
            }

        nested_value = (
            None if cached_nested is None else int(cached_nested)
        )

        return PreparedWindowStats(
            y_fit=y_fit,
            groups=groups,
            stats=stats_local,
            ms_list=ms_list,
            sigma_list=sigma_list,
            design_override=design_override,
            design_c=design_c,
            design_d=design_d,
            design_N=design_N,
            design_order=design_order,
            nested_replicates=nested_value,
            cache_payload=None,
        )
    except Exception:
        return None


def _compute_oneway_prepared(
    fit_blocks: list[pd.DataFrame],
    y_fit_raw: np.ndarray,
    replicates: int,
    code_signature_hash: str | None,
) -> PreparedWindowStats:
    group_indices = np.repeat(np.arange(len(fit_blocks)), replicates)
    if group_indices.shape[0] != y_fit_raw.shape[0]:
        raise ValueError("Unexpected imbalance in one-way design.")

    stats_raw = mean_squares(y_fit_raw, group_indices)
    ms1 = stats_raw["MS1"].astype(np.float64)
    ms2 = stats_raw["MS2"].astype(np.float64)
    sigma1 = stats_raw["Sigma1_hat"].astype(np.float64)
    sigma2 = stats_raw["Sigma2_hat"].astype(np.float64)

    stats_local = dict(stats_raw)
    stats_local["MS1"] = ms1
    stats_local["MS2"] = ms2
    stats_local["Sigma1_hat"] = sigma1
    stats_local["Sigma2_hat"] = sigma2
    stats_local.setdefault("replicates", replicates)

    design_c = np.array([float(stats_local["J"]), 1.0], dtype=np.float64)
    design_d = np.array(
        [float(stats_local["I"] - 1), float(stats_local["n"] - stats_local["I"])],
        dtype=np.float64,
    )
    design_N = float(stats_local["J"])
    design_order = [[1, 2], [2]]
    valid_indices = np.arange(y_fit_raw.shape[0], dtype=np.intp)
    groups = np.arange(valid_indices.size, dtype=np.intp)

    cache_payload: dict[str, Any] = {
        "design_mode": "oneway",
        "components": 2,
        "I": int(stats_local["I"]),
        "J": int(stats_local["J"]),
        "n": int(stats_local["n"]),
        "replicates": int(stats_local["J"]),
        "design_c": design_c,
        "design_d": design_d,
        "design_N": design_N,
        "design_order": design_order,
        "valid_indices": valid_indices,
        "MS1": ms1,
        "MS2": ms2,
        "Sigma1_hat": sigma1,
        "Sigma2_hat": sigma2,
        "nested_replicates": None,
        "code_signature": code_signature_hash,
    }

    return PreparedWindowStats(
        y_fit=y_fit_raw,
        groups=groups,
        stats=stats_local,
        ms_list=[ms1, ms2],
        sigma_list=[sigma1, sigma2],
        design_override=None,
        design_c=design_c,
        design_d=design_d,
        design_N=design_N,
        design_order=design_order,
        nested_replicates=None,
        cache_payload=cache_payload,
    )


def _compute_nested_prepared(
    fit_blocks: list[pd.DataFrame],
    y_fit_raw: np.ndarray,
    expected_reps: int,
    code_signature_hash: str | None,
) -> tuple[PreparedWindowStats | None, dict[str, Any] | None]:
    if expected_reps <= 1:
        raise ValueError("Nested design requires at least two replicates per cell.")
    if not fit_blocks:
        return (
            None,
            {
                "exit_reason": "No weekly blocks available for nested preparation.",
                "years_kept": 0,
                "weeks_common": 0,
                "replicates": int(expected_reps),
            },
        )

    date_arrays = [block.index.to_numpy(dtype="datetime64[ns]") for block in fit_blocks]
    dates_flat = np.concatenate(date_arrays)
    if dates_flat.size != y_fit_raw.shape[0]:
        raise ValueError("Mismatch between date index and observation count.")
    dt_index = pd.DatetimeIndex(dates_flat)
    iso = dt_index.isocalendar()
    year_labels = iso["year"].to_numpy()
    week_labels = iso["week"].to_numpy()
    idx_array = np.arange(y_fit_raw.shape[0], dtype=np.intp)

    labels_df = pd.DataFrame(
        {
            "year": year_labels,
            "week": week_labels,
            "idx": idx_array,
        }
    )
    total_years = int(pd.unique(year_labels).size)
    total_weeks = int(pd.unique(week_labels).size)
    counts = labels_df.groupby(["year", "week"])["idx"].transform("count")
    labels_df["valid_reps"] = counts == int(expected_reps)
    labels_valid = labels_df[labels_df["valid_reps"]]
    if labels_valid.empty:
        max_count = int(counts.max()) if counts.size else 0
        return (
            None,
            {
                "exit_reason": (
                    f"No (year, week) cells matched expected replicates={expected_reps} "
                    f"(observed max {max_count})."
                ),
                "years_kept": total_years,
                "weeks_common": 0,
                "replicates": int(expected_reps),
                "replicates_observed": max_count,
            },
        )

    week_sets_series = (
        labels_valid.drop_duplicates(subset=["year", "week"])
        .groupby("year")["week"]
        .apply(lambda series: set(int(value) for value in series.tolist()))
    )
    if week_sets_series.empty:
        return (
            None,
            {
                "exit_reason": "No ISO weeks remain after filtering to complete replicate cells.",
                "years_kept": int(total_years),
                "weeks_common": 0,
                "replicates": int(expected_reps),
            },
        )
    common_weeks = set.intersection(*week_sets_series.tolist())
    if not common_weeks:
        return (
            None,
            {
                "exit_reason": "Years share no common ISO weeks after replicate filtering.",
                "years_kept": int(len(week_sets_series)),
                "weeks_common": 0,
                "replicates": int(expected_reps),
            },
        )

    labels_common = labels_valid[labels_valid["week"].isin(common_weeks)].copy()
    if labels_common.empty:
        return (
            None,
            {
                "exit_reason": "Nested labels empty after intersecting common ISO weeks.",
                "years_kept": int(len(week_sets_series)),
                "weeks_common": int(len(common_weeks)),
                "replicates": int(expected_reps),
            },
        )
    labels_common.sort_values("idx", inplace=True)

    counts_cells = (
        labels_common.groupby(["year", "week"])["idx"].size().to_numpy(dtype=np.intp)
    )
    if counts_cells.size and np.any(counts_cells != int(expected_reps)):
        observed = sorted({int(val) for val in counts_cells.tolist()})
        return (
            None,
            {
                "exit_reason": "Replicate mismatch after filtering "
                f"(expected {expected_reps}, observed {observed}).",
                "years_kept": int(labels_common["year"].nunique()),
                "weeks_common": int(len(common_weeks)),
                "replicates": int(expected_reps),
                "replicates_observed": int(max(observed)) if observed else 0,
            },
        )

    counts_per_year = labels_common.groupby("year")["week"].nunique()
    if counts_per_year.empty or counts_per_year.nunique() != 1:
        min_weeks = int(counts_per_year.min()) if not counts_per_year.empty else 0
        max_weeks = int(counts_per_year.max()) if not counts_per_year.empty else 0
        return (
            None,
            {
                "exit_reason": "Common ISO week count differs by year "
                f"(min {min_weeks}, max {max_weeks}).",
                "years_kept": int(counts_per_year.index.size),
                "weeks_common": int(max_weeks),
                "replicates": int(expected_reps),
            },
        )
    weeks_per_year = int(counts_per_year.iloc[0])
    if weeks_per_year < 2:
        return (
            None,
            {
                "exit_reason": f"Require at least 2 common ISO weeks per year; found {weeks_per_year}.",
                "years_kept": int(counts_per_year.index.size),
                "weeks_common": int(weeks_per_year),
                "replicates": int(expected_reps),
            },
        )
    unique_years_common = labels_common["year"].nunique()
    if unique_years_common < 2:
        return (
            None,
            {
                "exit_reason": f"Nested detection requires at least 2 years; found {unique_years_common}.",
                "years_kept": int(unique_years_common),
                "weeks_common": int(weeks_per_year),
                "replicates": int(expected_reps),
            },
        )

    indices_final = labels_common["idx"].to_numpy(dtype=np.intp)
    year_final = labels_common["year"].to_numpy()
    week_final = labels_common["week"].to_numpy()
    y_fit = y_fit_raw[indices_final]

    try:
        (ms1, ms2, ms3), metadata = mean_squares_nested(
            y_fit,
            year_final,
            week_final,
            int(expected_reps),
        )
    except ValueError as exc:
        return (
            None,
            {
                "exit_reason": f"mean_squares_nested failure: {exc}",
                "years_kept": int(unique_years_common),
                "weeks_common": int(weeks_per_year),
                "replicates": int(expected_reps),
            },
        )

    ms1 = ms1.astype(np.float64)
    ms2 = ms2.astype(np.float64)
    ms3 = ms3.astype(np.float64)
    sigma1 = ((ms1 - ms2) / float(metadata.J * metadata.replicates)).astype(np.float64, copy=False)
    sigma2 = ((ms2 - ms3) / float(metadata.replicates)).astype(np.float64, copy=False)
    sigma3 = ms3.copy()

    stats_local: dict[str, Any] = {
        "MS1": ms1,
        "MS2": ms2,
        "MS3": ms3,
        "Sigma1_hat": sigma1,
        "Sigma2_hat": sigma2,
        "Sigma3_hat": sigma3,
        "I": metadata.I,
        "J": metadata.J,
        "n": metadata.n,
        "replicates": metadata.replicates,
    }

    design_c = metadata.c.astype(np.float64, copy=False)
    design_d = metadata.d.astype(np.float64, copy=False)
    design_N = float(metadata.N)
    design_order = [[1, 2, 3], [2, 3], [3]]
    design_override = {
        "c": design_c,
        "C": np.ones_like(design_c, dtype=np.float64),
        "d": design_d,
        "N": design_N,
        "order": design_order,
    }

    groups = np.arange(y_fit.shape[0], dtype=np.intp)
    cache_payload: dict[str, Any] = {
        "design_mode": "nested",
        "components": 3,
        "I": metadata.I,
        "J": metadata.J,
        "n": metadata.n,
        "replicates": metadata.replicates,
        "design_c": design_c,
        "design_d": design_d,
        "design_N": design_N,
        "design_order": design_order,
        "valid_indices": indices_final,
        "MS1": ms1,
        "MS2": ms2,
        "MS3": ms3,
        "Sigma1_hat": sigma1,
        "Sigma2_hat": sigma2,
        "Sigma3_hat": sigma3,
        "nested_replicates": metadata.J,
        "code_signature": code_signature_hash,
    }

    return (
        PreparedWindowStats(
            y_fit=y_fit,
            groups=groups,
            stats=stats_local,
            ms_list=[ms1, ms2, ms3],
            sigma_list=[sigma1, sigma2, sigma3],
            design_override=design_override,
            design_c=design_c,
            design_d=design_d,
            design_N=design_N,
            design_order=design_order,
            nested_replicates=metadata.J,
            cache_payload=cache_payload,
        ),
        None,
    )


def _compute_grouped_design_prepared(
    y_fit_raw: np.ndarray,
    group_labels: np.ndarray,
    *,
    design_mode: str,
    code_signature_hash: str | None,
) -> PreparedWindowStats:
    """Compute MANOVA statistics for alternate balanced daily groupings."""

    if y_fit_raw.ndim != 2 or y_fit_raw.shape[0] == 0:
        raise ValueError("y_fit_raw must be a non-empty two-dimensional array.")
    groups = np.asarray(group_labels, dtype=np.intp)
    if groups.ndim != 1:
        raise ValueError("group_labels must be one-dimensional.")
    if groups.shape[0] != y_fit_raw.shape[0]:
        raise ValueError("group_labels must align with observations.")

    stats_raw = mean_squares(y_fit_raw, groups)
    ms1 = stats_raw["MS1"].astype(np.float64)
    ms2 = stats_raw["MS2"].astype(np.float64)
    sigma1 = stats_raw["Sigma1_hat"].astype(np.float64)
    sigma2 = stats_raw["Sigma2_hat"].astype(np.float64)

    stats_local = dict(stats_raw)
    stats_local["MS1"] = ms1
    stats_local["MS2"] = ms2
    stats_local["Sigma1_hat"] = sigma1
    stats_local["Sigma2_hat"] = sigma2

    design_c = np.array([float(stats_local["J"]), 1.0], dtype=np.float64)
    design_d = np.array(
        [float(stats_local["I"] - 1), float(stats_local["n"] - stats_local["I"])],
        dtype=np.float64,
    )
    design_N = float(stats_local["J"])
    design_order = [[1, 2], [2]]

    valid_indices = np.arange(y_fit_raw.shape[0], dtype=np.intp)
    cache_payload = {
        "design_mode": str(design_mode),
        "components": 2,
        "I": int(stats_local["I"]),
        "J": int(stats_local["J"]),
        "n": int(stats_local["n"]),
        "replicates": int(stats_local["J"]),
        "design_c": design_c,
        "design_d": design_d,
        "design_N": design_N,
        "design_order": design_order,
        "valid_indices": valid_indices,
        "MS1": ms1,
        "MS2": ms2,
        "Sigma1_hat": sigma1,
        "Sigma2_hat": sigma2,
        "nested_replicates": None,
        "code_signature": code_signature_hash,
    }

    return PreparedWindowStats(
        y_fit=y_fit_raw,
        groups=groups,
        stats=stats_local,
        ms_list=[ms1, ms2],
        sigma_list=[sigma1, sigma2],
        design_override=None,
        design_c=design_c,
        design_d=design_d,
        design_N=design_N,
        design_order=design_order,
        nested_replicates=None,
        cache_payload=cache_payload,
    )


def _prepare_window_stats(
    design_mode: str,
    fit_blocks: list[pd.DataFrame],
    replicates: int,
    *,
    cached_stats: dict[str, Any] | None = None,
    nested_replicates: int | None = None,
) -> tuple[PreparedWindowStats | None, dict[str, Any] | None]:
    y_fit_raw = np.vstack([block.to_numpy(dtype=np.float64) for block in fit_blocks])
    expected_nested = (
        int(nested_replicates)
        if (design_mode == "nested" and nested_replicates is not None)
        else None
    )
    prepared_cached = _load_prepared_from_cache(
        cached_stats, design_mode, y_fit_raw, CODE_SIGNATURE, expected_nested
    )
    if prepared_cached is not None:
        return prepared_cached, None

    if design_mode == "nested":
        reps = int(nested_replicates or replicates)
        return _compute_nested_prepared(fit_blocks, y_fit_raw, reps, CODE_SIGNATURE)

    return (
        _compute_oneway_prepared(fit_blocks, y_fit_raw, replicates, CODE_SIGNATURE),
        None,
    )

def load_config(path: Path | str) -> dict[str, Any]:
    """Load experiment configuration, falling back to defaults."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    merged = DEFAULT_CONFIG | data
    default_gating = DEFAULT_CONFIG.get("gating", {}) or {}
    user_gating = data.get("gating") or {}
    if not isinstance(user_gating, dict):
        raise ValueError("gating configuration must be a mapping when provided.")
    # Ensure a copy so per-run mutation doesn't affect defaults
    merged["gating"] = {**default_gating, **user_gating}
    return merged


def _generate_synthetic_prices(path: Path) -> None:
    """Create a synthetic price panel for quick smoke testing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2024-12-31")
    dates = pd.date_range(start, end, freq="B")
    tickers = [f"T{idx:04d}" for idx in range(200)]

    market = rng.normal(scale=0.005, size=len(dates))
    records = []
    for ticker in tickers:
        beta = rng.normal(scale=0.5)
        idiosyncratic = rng.normal(scale=0.01, size=len(dates))
        log_returns = beta * market + idiosyncratic
        prices = 100 * np.exp(np.cumsum(log_returns))
        records.append(
            pd.DataFrame({"date": dates, "ticker": ticker, "price_close": prices})
        )

    full = pd.concat(records, ignore_index=True)
    full.to_csv(path, index=False)


def _mp_edges(
    noise_variance: float, n_assets: int, n_samples: int
) -> tuple[float, float]:
    """Return approximate Marčenko–Pastur bulk edges."""

    aspect_ratio = n_assets / max(n_samples, 1)
    sqrt_ratio = np.sqrt(aspect_ratio)
    upper = noise_variance * (1.0 + sqrt_ratio) ** 2
    lower = noise_variance * max(0.0, (1.0 - sqrt_ratio) ** 2)
    return lower, upper


def _prepare_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load daily returns from returns CSV or derive from prices CSV.

    If the configured path does not exist, a synthetic prices CSV is generated
    (legacy behavior) and returns are computed from prices.
    """

    data_path = Path(config["data_path"])
    if data_path.exists():
        # Peek at header to decide schema
        try:
            head = pd.read_csv(data_path, nrows=1)
        except Exception:
            head = pd.DataFrame()
        if {"date", "ticker", "ret"}.issubset(set(head.columns)):
            return load_returns_csv(data_path)
        # else assume prices schema
        prices = load_prices_csv(str(data_path))
        return to_daily_returns(prices)
    # Fallback: synthesize a price file and compute returns
    _generate_synthetic_prices(data_path)
    prices = load_prices_csv(str(data_path))
    return to_daily_returns(prices)


def _apply_preprocessing(
    daily_returns: pd.DataFrame,
    *,
    winsorize_q: float | None,
    huber_c: float | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply optional robustness preprocessing to daily returns."""

    flags: dict[str, str] = {}
    processed = daily_returns.copy()
    if winsorize_q is not None:
        processed = winsorize(processed, winsorize_q)
        flags["winsorize"] = f"{winsorize_q:.4g}"
    if huber_c is not None:
        processed = huberize(processed, huber_c)
        flags["huber"] = f"{huber_c:.4g}"
    return processed, flags


def _run_param_ablation(
    daily_returns: pd.DataFrame,
    output_dir: Path,
    *,
    partial_week_policy: str,
    target_component: int,
    base_delta: float,
    base_delta_frac: float | None,
    base_eps: float,
    base_eta: float,
    signed_a: bool,
    off_component_leak_cap: float | None,
    energy_min_abs: float | None,
    oneway_a_solver: str,
    preprocess_flags: Mapping[str, str] | None = None,
    grid_overrides: Mapping[str, Iterable[Any]] | None = None,
) -> None:
    """Grid sweep over detection parameters; emit CSV and heatmaps (E5).

    This routine uses a shorter rolling setup for speed and reproducibility.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_cache_dir = output_dir / "ablation_panel"
    panel_cache_dir.mkdir(parents=True, exist_ok=True)
    balanced_panel = _load_or_build_balanced_panel(
        daily_returns,
        days_per_week=5,
        partial_week_policy=partial_week_policy,
        output_dir=panel_cache_dir,
        precompute_panel=False,
        preprocess_flags=preprocess_flags,
    )
    weekly_balanced = balanced_panel.weekly
    week_map = balanced_panel.week_map
    replicates = balanced_panel.replicates
    # Use a compact rolling scheme for ablations
    # Choose a compact rolling setup that guarantees at least one window when possible
    total_weeks = int(weekly_balanced.shape[0])
    horizon_weeks = 1
    window_weeks = max(4, min(12, max(2, total_weeks - horizon_weeks)))
    windows = (
        list(rolling_windows(weekly_balanced, window_weeks, horizon_weeks))
        if total_weeks > horizon_weeks
        else []
    )
    if not windows:
        # Emit an empty summary so callers can rely on the artifact
        empty = pd.DataFrame(
            columns=[
                "delta_frac",
                "eps",
                "a_grid",
                "eta",
                "detection_rate",
                "mse_alias",
                "mse_de",
            ]
        )
        empty.to_csv(output_dir / "ablation_summary.csv", index=False)
        return

    def _normalise(values: Iterable[Any] | None, *, to_int: bool = False) -> list[float] | list[int]:
        if values is None:
            return []
        cleaned: list[float] = []
        for value in values:
            try:
                cleaned.append(float(value))
            except (TypeError, ValueError):
                continue
        if not cleaned:
            return []
        unique = list(dict.fromkeys(cleaned))
        unique.sort()
        if to_int:
            return [int(round(item)) for item in unique]
        return unique

    delta_fracs = [0.02, 0.03, 0.05]
    eps_vals = [0.02, 0.03, 0.05]
    a_grids = [72, 120, 144]
    etas = [0.4, 1.0]

    if grid_overrides:
        delta_override = _normalise(grid_overrides.get("delta_frac"))
        if delta_override:
            delta_fracs = [float(val) for val in cast(Iterable[float], delta_override)]
        eps_override = _normalise(grid_overrides.get("eps"))
        if eps_override:
            eps_vals = [float(val) for val in cast(Iterable[float], eps_override)]
        eta_override = _normalise(grid_overrides.get("eta"))
        if eta_override:
            etas = [float(val) for val in cast(Iterable[float], eta_override)]
        a_override = _normalise(grid_overrides.get("a_grid"), to_int=True)
        if a_override:
            a_grids = [int(val) for val in cast(Iterable[int], a_override)]

    records: list[dict[str, Any]] = []
    for df in delta_fracs:
        for eps in eps_vals:
            for ag in a_grids:
                for eta in etas:
                    det_count = 0
                    mse_alias_list: list[float] = []
                    mse_de_list: list[float] = []
                    for fit, hold in windows:
                        fit_blocks_raw = [week_map[idx] for idx in fit.index if idx in week_map]
                        hold_blocks_raw = [week_map[idx] for idx in hold.index if idx in week_map]
                        if len(fit_blocks_raw) != len(fit.index) or len(hold_blocks_raw) != len(hold.index):
                            continue
                        # Intersect tickers across the native per-week frames in this window
                        tickers_sets = [set(df.columns) for df in (fit_blocks_raw + hold_blocks_raw)]
                        ordered_tickers = sorted(set.intersection(*tickers_sets)) if tickers_sets else []
                        if not ordered_tickers:
                            continue
                        fit_blocks = [
                            df.loc[:, ordered_tickers].to_numpy(dtype=np.float64) for df in fit_blocks_raw
                        ]
                        hold_blocks = [
                            df.loc[:, ordered_tickers].to_numpy(dtype=np.float64) for df in hold_blocks_raw
                        ]
                        y_fit_daily = np.vstack(fit_blocks)
                        y_hold_daily = np.vstack(hold_blocks)
                        groups_fit = np.repeat(np.arange(len(fit_blocks)), replicates)
                        off_cap = off_component_leak_cap
                        detections = dealias_search(
                            y_fit_daily,
                            groups_fit,
                            target_r=target_component,
                            delta=0.0,
                            delta_frac=df,
                            eps=eps,
                            stability_eta_deg=eta,
                            use_tvector=True,
                            nonnegative_a=not signed_a,
                            a_grid=int(ag),
                            scan_basis="sigma",
                            off_component_leak_cap=(
                                None if off_cap is None else float(off_cap)
                            ),
                            energy_min_abs=energy_min_abs,
                            oneway_a_solver=oneway_a_solver,
                        )
                        det_count += int(bool(detections))
                        # Equal-weight weights for speed/consistency
                        w = np.full(
                            y_fit_daily.shape[1],
                            1.0 / y_fit_daily.shape[1],
                            dtype=np.float64,
                        )
                        f_alias, r_alias = variance_forecast_from_components(
                            y_fit_daily, y_hold_daily, replicates, w
                        )
                        f_de, r_de = variance_forecast_from_components(
                            y_fit_daily,
                            y_hold_daily,
                            replicates,
                            w,
                            detections=detections,
                        )
                        realized = r_de if np.isfinite(r_de) else r_alias
                        if np.isfinite(realized):
                            mse_alias_list.append(float((f_alias - realized) ** 2))
                            mse_de_list.append(float((f_de - realized) ** 2))

                    record = {
                        "delta_frac": df,
                        "eps": eps,
                        "a_grid": int(ag),
                        "eta": eta,
                        "detection_rate": det_count / max(len(windows), 1),
                        "mse_alias": (
                            float(np.mean(mse_alias_list))
                            if mse_alias_list
                            else float("nan")
                        ),
                        "mse_de": (
                            float(np.mean(mse_de_list)) if mse_de_list else float("nan")
                        ),
                    }
                    records.append(record)

    ablation_df = pd.DataFrame(records)
    ablation_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    # Simple heatmaps for detection rate and MSE delta at a fixed eta
    try:
        for eta in etas:
            subset = ablation_df[ablation_df["eta"] == eta]
            if subset.empty:
                continue
            pivot_det = subset.pivot_table(
                index="delta_frac",
                columns="eps",
                values="detection_rate",
                aggfunc="mean",
            )
            # Compute mean MSE gain matrix explicitly to avoid closure over loop var
            mse_gain = subset.copy()
            mse_gain["mse_gain"] = mse_gain["mse_alias"] - mse_gain["mse_de"]
            pivot_gain = mse_gain.pivot_table(
                index="delta_frac",
                columns="eps",
                values="mse_gain",
                aggfunc=lambda s: float(np.nanmean(s)),
            )
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(pivot_det.values, cmap="viridis", aspect="auto")
            axes[0].set_title(f"Detection rate (eta={eta})")
            axes[0].set_xticks(range(pivot_det.shape[1]))
            axes[0].set_xticklabels(pivot_det.columns)
            axes[0].set_yticks(range(pivot_det.shape[0]))
            axes[0].set_yticklabels(pivot_det.index)
            fig.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(pivot_gain.values, cmap="RdBu", aspect="auto")
            axes[1].set_title(f"MSE gain (alias - de) (eta={eta})")
            axes[1].set_xticks(range(pivot_gain.shape[1]))
            axes[1].set_xticklabels(pivot_gain.columns)
            axes[1].set_yticks(range(pivot_gain.shape[0]))
            axes[1].set_yticklabels(pivot_gain.index)
            fig.colorbar(im1, ax=axes[1])
            fig.tight_layout()
            fig.savefig(
                output_dir / f"E5_ablation_eta{eta:.1f}.png", bbox_inches="tight"
            )
            plt.close(fig)
    except Exception:
        # Best-effort plotting; CSV is the primary artifact
        pass


## Stats moved to evaluation.evaluate (eval_iqr, eval_sign_test_p)


## Plotters moved to evaluation.evaluate (eval_plot_var_panel, eval_plot_cov_err)


def _load_or_build_balanced_panel(
    daily_returns: pd.DataFrame,
    *,
    days_per_week: int,
    partial_week_policy: str,
    output_dir: Path | None,
    precompute_panel: bool,
    preprocess_flags: Mapping[str, str] | None = None,
) -> BalancedPanel:
    """Load a cached balanced panel or build a fresh one from daily returns."""

    if partial_week_policy not in {"drop", "impute"}:
        raise ValueError("partial_week_policy must be either 'drop' or 'impute'.")

    data_hash = hash_daily_returns(daily_returns)
    expected_flags = {str(k): str(v) for k, v in (preprocess_flags or {}).items()}
    cache_path: Path | None = None
    manifest_path: Path | None = None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "panel_balanced.pkl"
        manifest_path = output_dir / "panel_manifest.json"
        if cache_path.exists() and manifest_path.exists():
            try:
                manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = PanelManifest.from_dict(manifest_raw)
                if (
                    manifest.data_hash == data_hash
                    and manifest.partial_week_policy == partial_week_policy
                    and manifest.days_per_week == days_per_week
                    and manifest.preprocess_flags == expected_flags
                ):
                    cached_panel = load_balanced_panel(cache_path)
                    if (
                        cached_panel.manifest.data_hash == manifest.data_hash
                        and cached_panel.replicates == days_per_week
                        and cached_panel.manifest.preprocess_flags == expected_flags
                    ):
                        # Refresh manifest formatting in case schema evolved.
                        write_panel_manifest(cached_panel.manifest, manifest_path)
                        return cached_panel
            except Exception:
                pass

    balanced = build_balanced_weekday_panel(
        daily_returns,
        days_per_week=days_per_week,
        partial_week_policy=partial_week_policy,  # type: ignore[arg-type]
        preprocess_flags=expected_flags,
    )

    if manifest_path is not None:
        try:
            write_panel_manifest(balanced.manifest, manifest_path)
        except Exception:
            pass
    if precompute_panel and cache_path is not None:
        try:
            save_balanced_panel(balanced, cache_path)
        except Exception:
            pass

    return balanced


def _run_single_period(
    daily_returns: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    output_dir: Path,
    window_weeks: int,
    horizon_weeks: int,
    delta: float,
    delta_frac: float | None,
    eps: float,
    stability_eta: float,
    signed_a: bool,
    target_component: int,
    partial_week_policy: str,
    precompute_panel: bool,
    cache_dir: Path | None,
    resume_cache: bool,
    cs_drop_top_frac: float,
    cs_sensitivity_frac: float,
    off_component_leak_cap: float | None,
    sigma_ablation: bool,
    label: str,
    crisis_label: str | None = None,
    design_mode: str,
    nested_replicates: int,
    oneway_a_solver: str,
    estimator: str,
    progress: bool = True,
    a_grid: int = 120,
    energy_min_abs: float | None = None,
    factor_returns: pd.DataFrame | None = None,
    minvar_ridge: float = 1e-4,
    minvar_box: tuple[float, float] = (0.0, 0.1),
    turnover_cost_bps: float = 5.0,
    minvar_condition_cap: float = 1e9,
    preprocess_flags: Mapping[str, str] | None = None,
    gating: Mapping[str, Any] | None = None,
    alignment_top_p: int = 3,
    edge_mode: str = "scm",
    edge_huber_c: float = 1.5,
) -> None:
    """Execute the rolling evaluation for a single date range."""

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError("start date must be on or before end date.")

    mask = (daily_returns.index >= start_ts) & (daily_returns.index <= end_ts)
    daily_subset = daily_returns.loc[mask]
    if daily_subset.empty:
        raise ValueError(
            f"No data available within the window {start_ts.date()} to {end_ts.date()}."
        )

    balanced_panel = _load_or_build_balanced_panel(
        daily_subset,
        days_per_week=5,
        partial_week_policy=partial_week_policy,
        output_dir=output_dir,
        precompute_panel=precompute_panel,
        preprocess_flags=preprocess_flags,
    )
    weekly_balanced = balanced_panel.weekly
    week_map = balanced_panel.week_map
    replicates = balanced_panel.replicates
    design_mode = design_mode.lower()
    valid_designs = {"oneway", "nested", "dow", "vol"}
    if design_mode not in valid_designs:
        raise ValueError(f"design_mode must be one of {sorted(valid_designs)}.")
    if design_mode == "nested":
        nested_reps_value = int(nested_replicates) if nested_replicates > 0 else int(replicates)
    else:
        nested_reps_value = int(replicates)
    solver_mode = oneway_a_solver.lower()
    if solver_mode not in {"auto", "rootfind", "grid"}:
        raise ValueError("oneway_a_solver must be 'auto', 'rootfind', or 'grid'.")
    estimator_mode = (estimator or "dealias").strip().lower()
    edge_mode_cfg = (edge_mode or "scm").strip().lower()
    edge_huber_c_val = float(edge_huber_c)
    try:
        delta_frac_config = float(delta_frac) if delta_frac is not None else None
    except (TypeError, ValueError):
        delta_frac_config = None
    if delta_frac_config is not None and delta_frac_config < 0.0:
        delta_frac_config = 0.0
    solver_usage: set[str] = set()
    tickers = balanced_panel.ordered_tickers
    dropped_weeks = balanced_panel.dropped_weeks
    total_weeks = int(weekly_balanced.shape[0])
    if total_weeks < window_weeks + horizon_weeks:
        # Auto-shrink to a minimal viable rolling scheme when possible
        if total_weeks >= 3:
            window_weeks = max(2, total_weeks - 1)
            horizon_weeks = 1
        else:
            raise ValueError(
                "Not enough balanced weeks for the requested rolling evaluation window."
            )

    p_assets = len(tickers)
    equal_weights = equal_weight(p_assets)

    minvar_lo_bound = float(minvar_box[0])
    minvar_hi_bound = float(minvar_box[1])
    box_bounds = (minvar_lo_bound, minvar_hi_bound)
    solver_stats: dict[str, dict[str, Any]] = defaultdict(dict)

    cov_weekly = np.cov(
        weekly_balanced.to_numpy(dtype=np.float64), rowvar=False, ddof=1
    )
    eigenvalues = np.linalg.eigvalsh(cov_weekly)
    avg_noise = float(np.median(np.diag(cov_weekly)))
    edges = _mp_edges(
        avg_noise,
        n_assets=cov_weekly.shape[0],
        n_samples=weekly_balanced.shape[0],
    )
    plot_title = f"{label.title()} weekly covariance spectrum"
    plot_spectrum_with_edges(
        eigenvalues,
        edges=edges,
        out_path=output_dir / "spectrum.png",
        title=plot_title,
        highlight_threshold=max(edges) if edges else None,
    )
    plot_spectrum_with_edges(
        eigenvalues,
        edges=edges,
        out_path=output_dir / "spectrum.pdf",
        title=plot_title,
        highlight_threshold=max(edges) if edges else None,
    )
    # Also save E1 into experiments/<run>/figures
    try:
        run_name = output_dir.parents[1].name if len(output_dir.parents) >= 2 else output_dir.parent.name
        e1_plot_spectrum_with_mp(eigenvalues, edges, run=run_name, title=plot_title)
    except Exception:
        pass

    minvar_memo = MinVarMemo()

    def _equal_weight_weights(covariance: np.ndarray) -> np.ndarray:
        n = int(covariance.shape[0])
        if n <= 0:
            return np.array([], dtype=np.float64)
        return np.full(n, 1.0 / float(n), dtype=np.float64)

    minvar_label_box = "Min-Variance (box)"
    minvar_label_long = "Min-Variance (long-only)"

    def _min_var_weights(covariance: np.ndarray) -> np.ndarray:
        weights, info = minvar_ridge_box(
            covariance,
            box=box_bounds,
            ridge=float(minvar_ridge),
            cache=minvar_memo,
        )
        solver_stats[minvar_label_box] = info
        return np.asarray(weights, dtype=np.float64)

    def _min_var_longonly_weights(covariance: np.ndarray) -> np.ndarray:
        weights, info = minvar_ridge_box(
            covariance,
            box=box_bounds,
            ridge=float(minvar_ridge),
            cache=minvar_memo,
        )
        solver_stats[minvar_label_long] = info
        return np.asarray(weights, dtype=np.float64)

    strategies: dict[str, dict[str, Any]] = {
        "Equal Weight": {
            "prefix": "eq",
            "get_weights": _equal_weight_weights,
            "available": True,
        },
        minvar_label_box: {
            "prefix": "mv",
            "get_weights": _min_var_weights,
            "available": True,
        },
        minvar_label_long: {
            "prefix": "mvlo",
            "get_weights": _min_var_longonly_weights,
            "available": True,
        },
    }

    errors_by_combo: dict[str, dict[int, float]] = defaultdict(dict)
    qlike_by_combo: dict[str, dict[int, float]] = defaultdict(dict)
    var95_by_combo: dict[str, list[float]] = defaultdict(list)
    realised_returns_by_combo: dict[str, list[float]] = defaultdict(list)
    realized_by_combo_raw: dict[str, dict[int, float]] = defaultdict(dict)
    strategy_success: dict[str, bool] = {name: False for name in strategies}
    strategy_estimators: dict[str, set[str]] = defaultdict(set)
    weights_history: dict[str, list[np.ndarray]] = {name: [] for name in strategies}
    strategy_windows: dict[str, list[int]] = {name: [] for name in strategies}
    turnover_cost_history: dict[str, list[float]] = {name: [] for name in strategies}
    prev_weights: dict[str, np.ndarray | None] = {name: None for name in strategies}
    records: list[dict[str, Any]] = []
    rejection_totals: dict[str, int] = {}
    edge_margin_values: list[float] = []
    detection_windows = 0
    substituted_windows = 0
    gating_cfg = dict(gating or {})
    gating_mode_value = str(gating_cfg.get("mode", "fixed") or "fixed").strip().lower()
    if gating_mode_value not in {"fixed", "calibrated"}:
        gating_mode_value = "fixed"
    calibration_path_cfg = gating_cfg.get("calibration_path")
    if calibration_path_cfg is not None:
        calibration_path = Path(str(calibration_path_cfg)).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = (PROJECT_ROOT / calibration_path).resolve()
    else:
        calibration_path = (PROJECT_ROOT / "calibration" / "edge_delta_thresholds.json").resolve()
    gating_enabled = bool(gating_cfg.get("enable", True))
    try:
        gating_q_max = int(gating_cfg.get("q_max", 0) or 0)
    except (TypeError, ValueError):
        gating_q_max = 0
    if gating_q_max < 0:
        gating_q_max = 0
    gating_require_isolated = bool(gating_cfg.get("require_isolated", True))
    gating_skip_reasons: dict[str, int] = {}
    gating_discard_log: list[dict[str, Any]] = []
    delta_usage_records: list[dict[str, Any]] = []
    delta_used_values: list[float] = []
    calibration_misses: set[tuple[str, int, int]] = set()
    design_logged = False
    try:
        alignment_top_p = int(alignment_top_p)
    except (TypeError, ValueError):
        alignment_top_p = 3
    if alignment_top_p <= 0:
        alignment_top_p = 3
    nested_skip_reasons: dict[str, int] = {}
    nested_skip_detail_map: dict[str, dict[str, Any]] = {}

    baseline_name = "Equal Weight"
    baseline_alias_key = f"{baseline_name}::Aliased"
    var_forecasts_alias_baseline: list[float] = []
    var_forecasts_de_baseline: list[float] = []
    var_forecasts_lw_baseline: list[float] = []
    factor_warned = False
    factor_obs_warned = False
    poet_warned = False

    total_windows = weekly_balanced.shape[0] - (window_weeks + horizon_weeks) + 1
    window_iter = rolling_windows(weekly_balanced, window_weeks, horizon_weeks)
    window_cache_dir = cache_dir / label if cache_dir is not None else None
    if progress and total_windows > 0:
        window_iter = tqdm(
            window_iter,
            total=total_windows,
            desc=f"Rolling windows ({label})",
            unit="window",
        )  # type: ignore
    for window_idx, (fit, hold) in enumerate(window_iter):
        if hold.empty:
            continue

        # Per-window universe intersection to maximize balanced weeks
        fit_blocks_raw = [week_map[idx] for idx in fit.index if idx in week_map]
        hold_blocks_raw = [week_map[idx] for idx in hold.index if idx in week_map]
        if len(fit_blocks_raw) != len(fit.index) or len(hold_blocks_raw) != len(hold.index):
            continue
        # Intersect tickers across the native per-week frames for current window
        tickers_sets = [set(df.columns) for df in (fit_blocks_raw + hold_blocks_raw)]
        if tickers_sets:
            candidate = set.intersection(*tickers_sets)
            # Constrain to columns available in the rectangular weekly panel
            ordered_tickers = sorted(candidate & set(weekly_balanced.columns))
        else:
            ordered_tickers = []
        if not ordered_tickers:
            continue
        # Reindex each weekly frame to the intersection and stack
        fit_blocks = [df.loc[:, ordered_tickers] for df in fit_blocks_raw]
        hold_blocks = [df.loc[:, ordered_tickers] for df in hold_blocks_raw]

        fit_block_arrays = [block.to_numpy(dtype=np.float64) for block in fit_blocks]
        hold_block_arrays = [block.to_numpy(dtype=np.float64) for block in hold_blocks]
        if not fit_block_arrays:
            continue
        y_fit_weekly_order = np.vstack(fit_block_arrays)
        if hold_block_arrays:
            y_hold_daily = np.vstack(hold_block_arrays)
        else:
            y_hold_daily = np.empty((0, y_fit_weekly_order.shape[1]), dtype=np.float64)

        cache_key: str | None = None
        cached_stats: dict[str, Any] | None = None
        if window_cache_dir is not None:
            week_list = [ts.strftime("%Y-%m-%d") for ts in fit.index]
            nested_key = nested_reps_value if design_mode == "nested" else None
            cache_key = window_key(
                balanced_panel.manifest,
                week_list,
                ordered_tickers,
                replicates,
                code_signature=CODE_SIGNATURE,
                design=design_mode,
                nested_replicates=nested_key,
                oneway_a_solver=solver_mode,
                estimator=estimator_mode,
                preprocess_flags=balanced_panel.manifest.preprocess_flags,
            )
            if resume_cache:
                cached_stats = load_window(window_cache_dir, cache_key) or None

        prepared: PreparedWindowStats | None = None
        prep_info: dict[str, Any] | None = None

        if design_mode in {"dow", "vol"}:
            if design_mode == "dow":
                groups_override = np.tile(
                    np.arange(replicates, dtype=np.intp),
                    len(fit_block_arrays),
                )
            else:
                groups_override = np.repeat(
                    np.arange(len(fit_block_arrays), dtype=np.intp),
                    replicates,
                )
            if groups_override.shape[0] != y_fit_weekly_order.shape[0]:
                continue
            try:
                prepared = _compute_grouped_design_prepared(
                    y_fit_weekly_order,
                    groups_override,
                    design_mode=design_mode,
                    code_signature_hash=CODE_SIGNATURE,
                )
            except ValueError:
                continue
            if not design_logged:
                counts = np.bincount(groups_override)
                print(
                    f"[design:{design_mode}] window={window_idx} group_sizes={counts.tolist()} total_obs={int(groups_override.size)}"
                )
                design_logged = True
        else:
            prepared, prep_info = _prepare_window_stats(
                design_mode,
                fit_blocks,
                replicates,
                cached_stats=cached_stats if resume_cache else None,
                nested_replicates=nested_reps_value,
            )

        if prepared is None:
            if design_mode == "nested" and prep_info:
                reason_value = str(prep_info.get("exit_reason", "unknown"))
                nested_skip_reasons[reason_value] = (
                    nested_skip_reasons.get(reason_value, 0) + 1
                )
                detail = nested_skip_detail_map.setdefault(
                    reason_value,
                    {
                        "exit_reason": reason_value,
                        "windows": 0,
                        "years_kept": 0,
                        "weeks_common": 0,
                        "replicates": int(prep_info.get("replicates", nested_reps_value)),
                    },
                )
                detail["windows"] = int(detail.get("windows", 0)) + 1
                years_val = int(prep_info.get("years_kept", 0))
                weeks_val = int(prep_info.get("weeks_common", 0))
                detail["years_kept"] = max(int(detail.get("years_kept", 0)), years_val)
                detail["weeks_common"] = max(int(detail.get("weeks_common", 0)), weeks_val)
                if "replicates_observed" in prep_info:
                    detail["replicates_observed"] = max(
                        int(prep_info.get("replicates_observed", 0)),
                        int(detail.get("replicates_observed", 0)),
                    )
            continue

        y_fit_daily = prepared.y_fit
        groups_fit = prepared.groups
        stats_local = prepared.stats
        ms_list = prepared.ms_list
        sigma_list = prepared.sigma_list
        design_override = prepared.design_override
        design_c_local = prepared.design_c
        design_d_local = prepared.design_d
        design_N_local = prepared.design_N
        _design_order_local = prepared.design_order

        if (
            prepared.cache_payload is not None
            and window_cache_dir is not None
            and cache_key is not None
        ):
            save_window(window_cache_dir, cache_key, prepared.cache_payload)

        edge_scale_val = 1.0
        edge_scm_val = float("nan")
        edge_tyler_val = float("nan")
        edge_selected_val = float("nan")
        n_fit_samples = int(y_fit_daily.shape[0])
        p_dim = int(y_fit_daily.shape[1]) if y_fit_daily.ndim == 2 else 0
        if p_dim > 0 and n_fit_samples > 0:
            try:
                scatter_scm = np.cov(y_fit_daily, rowvar=False, ddof=1)
                scatter_scm = 0.5 * (scatter_scm + scatter_scm.T)
                edge_scm_val = edge_from_scatter(scatter_scm, p_dim, n_fit_samples)
            except Exception:
                scatter_scm = None
                edge_scm_val = float("nan")
            try:
                scatter_tyler = tyler_scatter(y_fit_daily)
                scatter_tyler = 0.5 * (scatter_tyler + scatter_tyler.T)
                edge_tyler_val = edge_from_scatter(
                    scatter_tyler,
                    p_dim,
                    n_fit_samples,
                )
            except Exception:
                scatter_tyler = None
                edge_tyler_val = float("nan")
            if edge_mode_cfg == "tyler":
                edge_selected_val = edge_tyler_val
            elif edge_mode_cfg == "huber":
                try:
                    scatter_huber = huber_scatter(y_fit_daily, edge_huber_c_val)
                    scatter_huber = 0.5 * (scatter_huber + scatter_huber.T)
                    edge_selected_val = edge_from_scatter(
                        scatter_huber,
                        p_dim,
                        n_fit_samples,
                    )
                except Exception:
                    scatter_huber = None
                    edge_selected_val = float("nan")
            else:
                edge_selected_val = edge_scm_val
            if (
                np.isfinite(edge_scm_val)
                and edge_scm_val > 0.0
                and np.isfinite(edge_selected_val)
                and edge_selected_val > 0.0
            ):
                edge_scale_val = float(edge_selected_val / edge_scm_val)
                if not np.isfinite(edge_scale_val) or edge_scale_val <= 0.0:
                    edge_scale_val = 1.0
            else:
                edge_scale_val = 1.0
        edge_band_min = float("nan")
        edge_band_max = float("nan")
        edge_candidates = [
            float(val)
            for val in (edge_scm_val, edge_tyler_val, edge_selected_val)
            if np.isfinite(val) and val > 0.0
        ]
        if edge_candidates:
            edge_band_min = float(min(edge_candidates))
            edge_band_max = float(max(edge_candidates))

        edge_scale_used = edge_scale_val
        if not np.isfinite(edge_scale_used) or edge_scale_used <= 0.0:
            edge_scale_used = 1.0

        base_delta_frac_val = float(delta_frac_config) if delta_frac_config is not None else 0.0
        delta_frac_calibrated = None
        if gating_mode_value == "calibrated" and p_dim > 0 and n_fit_samples > 0:
            delta_frac_calibrated = lookup_calibrated_delta(
                edge_mode=edge_mode_cfg,
                p=p_dim,
                t=n_fit_samples,
                calibration_path=calibration_path,
            )
            if delta_frac_calibrated is None:
                miss_key = (edge_mode_cfg, p_dim, n_fit_samples)
                if miss_key not in calibration_misses:
                    calibration_misses.add(miss_key)
                    print(
                        (
                            f"[gate] Missing calibrated delta for edge={edge_mode_cfg} "
                            f"(p={p_dim}, T={n_fit_samples}); using config delta_frac={base_delta_frac_val:.4f}"
                        ),
                        file=sys.stderr,
                    )
        if delta_frac_calibrated is not None:
            delta_frac_used_value = max(base_delta_frac_val, float(delta_frac_calibrated))
        else:
            delta_frac_used_value = base_delta_frac_val
        if delta_frac_used_value < 0.0 or not np.isfinite(delta_frac_used_value):
            delta_frac_used_value = 0.0
        delta_used_values.append(float(delta_frac_used_value))
        delta_usage_records.append(
            {
                "window": int(window_idx),
                "p": int(p_dim),
                "t": int(n_fit_samples),
                "delta_frac_used": float(delta_frac_used_value),
                **(
                    {"delta_frac_calibrated": float(delta_frac_calibrated)}
                    if delta_frac_calibrated is not None
                    else {}
                ),
            }
        )

        off_cap = off_component_leak_cap
        diag_local: dict[str, int] = {}
        detections = dealias_search(
            y_fit_daily,
            groups_fit,
            target_r=target_component,
            delta=delta,
            delta_frac=float(delta_frac_used_value),
            eps=eps,
            energy_min_abs=energy_min_abs,
            stability_eta_deg=stability_eta,
            use_tvector=True,
            nonnegative_a=not signed_a,
            a_grid=int(a_grid),
            cs_drop_top_frac=float(cs_drop_top_frac),
            cs_sensitivity_frac=float(cs_sensitivity_frac),
            scan_basis="sigma",
            off_component_leak_cap=(
                None if off_cap is None else float(off_cap)
            ),
            diagnostics=diag_local,
            stats=stats_local,
            design=design_override,
            oneway_a_solver=solver_mode,
            edge_scale=edge_scale_used,
            edge_mode=edge_mode_cfg,
        )
        for key, value in diag_local.items():
            rejection_totals[key] = rejection_totals.get(key, 0) + int(value)
        detections = list(detections or [])
        window_skip_reason: str | None = None
        gate_discard_detail: list[dict[str, float]] = []
        isolated_count_raw = count_isolated_outliers(detections, None, None)

        if gating_enabled and detections:
            candidate_pool: list[dict[str, Any]] = [
                det for det in detections if isinstance(det, dict)
            ]
            if gating_require_isolated:
                if isolated_count_raw == 0:
                    window_skip_reason = "no_isolated_spike"
                    gating_skip_reasons[window_skip_reason] = (
                        gating_skip_reasons.get(window_skip_reason, 0) + 1
                    )
                    candidate_pool = []
                else:
                    filtered_pool: list[dict[str, Any]] = []
                    for det in candidate_pool:
                        try:
                            pre_val = int(det.get("pre_outlier_count", 0))
                        except (TypeError, ValueError):
                            pre_val = 0
                        if pre_val == 1:
                            filtered_pool.append(det)
                    if filtered_pool:
                        candidate_pool = filtered_pool
            if candidate_pool and gating_q_max > 0 and len(candidate_pool) > gating_q_max:
                selected, discarded = select_top_k(candidate_pool, gating_q_max)
                candidate_pool = list(selected)
                if discarded:
                    gate_discard_detail = []
                    for det in discarded:
                        if not isinstance(det, dict):
                            continue
                        lambda_val = det.get("lambda_hat")
                        mu_val = det.get("mu_hat")
                        score_val = det.get("target_energy", 0.0)
                        try:
                            energy_val = float(det.get("target_energy", 0.0))
                        except (TypeError, ValueError):
                            energy_val = 0.0
                        try:
                            stability_val = float(det.get("stability_margin", 0.0))
                        except (TypeError, ValueError):
                            stability_val = 0.0
                        try:
                            mu_float = float(mu_val)
                        except (TypeError, ValueError):
                            mu_float = float("nan")
                        try:
                            delta_frac_val = float(det.get("delta_frac", float("nan")))
                        except (TypeError, ValueError):
                            delta_frac_val = float("nan")
                        try:
                            leak_val = float(det.get("off_component_ratio", float("nan")))
                        except (TypeError, ValueError):
                            leak_val = float("nan")
                        if not np.isfinite(energy_val):
                            energy_val = 0.0
                        if not np.isfinite(stability_val):
                            stability_val = 0.0
                        score_val = max(energy_val, 0.0) * max(stability_val, 0.0)
                        gate_discard_detail.append(
                            {
                                "lambda_hat": (
                                    float(lambda_val)
                                    if isinstance(lambda_val, (float, int, np.floating, np.integer))
                                    else float("nan")
                                ),
                                "mu_hat": float(mu_float),
                                "stability_margin": float(stability_val),
                                "target_energy": float(energy_val),
                                "off_component_ratio": float(leak_val),
                                "delta_frac": float(delta_frac_val),
                                "score": float(score_val),
                                "accepted": False,
                            }
                        )
                    gating_discard_log.append(
                        {"window": int(window_idx), "discarded": gate_discard_detail}
                    )
                    lambda_str = ", ".join(
                        f"{item['lambda_hat']:.4f}"
                        for item in gate_discard_detail
                        if np.isfinite(item.get("lambda_hat", float("nan")))
                    )
                    print(
                        f"[gate] Window {window_idx}: discarded {len(gate_discard_detail)} detection(s) "
                        f"(lambda={lambda_str or 'n/a'})",
                        file=sys.stderr,
                    )
            if window_skip_reason:
                detections = []
            else:
                detections = candidate_pool

        if detections:
            detection_windows += 1
            substituted_windows += 1
            for det in detections:
                edge_val = det.get("edge_margin") if isinstance(det, dict) else None
                if edge_val is not None and np.isfinite(edge_val):
                    edge_margin_values.append(float(edge_val))

        # Optional per-window diagnostics: MP edge vs top eigenvalue across angles (two-component only)
        try:
            if len(ms_list) != 2 or len(sigma_list) != 2:
                raise ValueError("Skip diagnostics for designs with more than two components.")
            p_dim = int(ms_list[0].shape[0])
            drop_top = min(p_dim - 1, max(1, int(round(p_dim * float(cs_drop_top_frac)))))
            cs_vec_local = estimate_Cs_from_MS(
                ms_list,
                design_d_local.tolist(),
                design_c_local.tolist(),
                drop_top=drop_top,
            )

            sigma_total = sigma_list[0] + sigma_list[1]
            try:
                eigvals_total = np.linalg.eigvalsh(0.5 * (sigma_total + sigma_total.T))
                lam_mean = float(np.mean(eigvals_total)) if eigvals_total.size else float("nan")
            except Exception:
                lam_mean = float("nan")
            if np.isfinite(lam_mean) and lam_mean > 0.0:
                C_diag = np.full_like(design_c_local, lam_mean, dtype=np.float64)
            else:
                C_diag = np.ones_like(design_c_local, dtype=np.float64)

            ms1_local = ms_list[0]
            ms2_local = ms_list[1]
            sigma1_local = sigma_list[0]
            sigma2_local = sigma_list[1]

            angles = np.linspace(0.0, 2.0 * np.pi, num=int(a_grid), endpoint=False, dtype=np.float64)
            rows = []
            for theta in angles:
                a_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
                try:
                    z_plus = mp_edge(
                        a_vec.tolist(),
                        C_diag.tolist(),
                        design_d_local.tolist(),
                        design_N_local,
                        Cs=cs_vec_local,
                    )
                except Exception:
                    continue
                sigma_a = float(a_vec[0]) * sigma1_local + float(a_vec[1]) * sigma2_local
                try:
                    lam_top = float(np.linalg.eigvalsh(sigma_a)[-1])
                except Exception:
                    continue
                delta_frac_term = float(delta_frac_used_value) * z_plus if delta_frac_used_value else 0.0
                threshold = z_plus + max(float(delta), delta_frac_term)
                rows.append(
                    {
                        "theta": float(theta),
                        "z_plus": float(z_plus),
                        "lambda_top": lam_top,
                        "edge_margin": float(lam_top - threshold),
                    }
                )
            if rows:
                diag_path = output_dir / f"edge_diag_window{window_idx:03d}.csv"
                pd.DataFrame(rows).to_csv(diag_path, index=False)
        except Exception:
            pass

        fit_matrix = fit.loc[:, ordered_tickers].to_numpy(dtype=np.float64)
        hold_matrix = hold.loc[:, ordered_tickers].to_numpy(dtype=np.float64)
        if fit_matrix.shape[0] < 2:
            continue
        cov_fit = np.cov(fit_matrix, rowvar=False, ddof=1)
        if not np.all(np.isfinite(cov_fit)):
            continue

        window_record: dict[str, Any] = {
            "label": label,
            "fit_start": fit.index[0],
            "fit_end": fit.index[-1],
            "hold_start": hold.index[0],
            "hold_end": hold.index[-1],
            "n_detections": len(detections),
            "skip_reason": window_skip_reason or "",
            "isolated_spikes": int(isolated_count_raw),
            "gate_discarded_count": len(gate_discard_detail),
            "edge_mode": edge_mode_cfg,
            "gating_mode": gating_mode_value,
            "edge_scale": float(edge_scale_used),
            "edge_scm": float(edge_scm_val),
            "edge_tyler": float(edge_tyler_val),
            "edge_selected": float(edge_selected_val),
            "edge_band_min": float(edge_band_min),
            "edge_band_max": float(edge_band_max),
            "delta_frac_used": float(delta_frac_used_value),
            "delta_frac_config": (
                float(delta_frac_config) if delta_frac_config is not None else float("nan")
            ),
            "delta_frac_calibrated": (
                float(delta_frac_calibrated) if delta_frac_calibrated is not None else float("nan")
            ),
            "p_dim": int(p_dim),
            "n_fit_samples": int(n_fit_samples),
            "stability_eta_deg": float(stability_eta),
            "off_component_cap": (
                float(off_component_leak_cap)
                if off_component_leak_cap is not None
                else float("nan")
            ),
        }
        if gate_discard_detail:
            window_record["gate_discarded"] = json.dumps(gate_discard_detail)
        for strategy_meta in strategies.values():
            prefix = strategy_meta["prefix"]
            window_record[f"{prefix}_turnover"] = float("nan")
            window_record[f"{prefix}_turnover_cost"] = float("nan")
            window_record[f"{prefix}_mv_cond_penalized"] = float("nan")
            window_record[f"{prefix}_mv_cond_original"] = float("nan")
            window_record[f"{prefix}_mv_iterations"] = float("nan")
            window_record[f"{prefix}_mv_converged"] = False
            window_record[f"{prefix}_mv_condition_flag"] = False

        # Log top detection (by lambda_hat) for diagnostics/time series
        if detections:
            det_sorted = sorted(
                detections, key=lambda d: float(d["lambda_hat"]), reverse=True
            )
            def _safe_num(value: Any) -> float | None:
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    return None
                return val if np.isfinite(val) else None

            detail_payload: list[dict[str, Any]] = []
            for det in det_sorted:
                if not isinstance(det, dict):
                    continue
                solver = det.get("solver_used")
                if solver:
                    solver_usage.add(str(solver))
                t_vals = det.get("t_values") or []
                detail_payload.append(
                    {
                        "lambda_hat": _safe_num(det.get("lambda_hat")),
                        "mu_hat": _safe_num(det.get("mu_hat")),
                        "z_plus": _safe_num(det.get("z_plus")),
                        "edge_margin": _safe_num(det.get("edge_margin")),
                        "buffer_margin": _safe_num(det.get("buffer_margin")),
                        "target_energy": _safe_num(det.get("target_energy")),
                        "stability_margin": _safe_num(det.get("stability_margin")),
                        "off_component_ratio": _safe_num(det.get("off_component_ratio")),
                        "delta_frac": _safe_num(det.get("delta_frac")),
                        "t_values": [float(val) for val in t_vals],
                        "admissible_root": bool(det.get("admissible_root", False)),
                        "a": [float(val) for val in det.get("a", [])],
                        "components": [
                            float(val) for val in (det.get("components") or [])
                        ],
                        "solver_used": det.get("solver_used"),
                        "accepted": True,
                    }
                )
            window_record["detections_detail"] = json.dumps(detail_payload)

            top = det_sorted[0]
            window_record["top_lambda_hat"] = float(top["lambda_hat"])
            window_record["top_mu_hat"] = float(top["mu_hat"])
            top_a_values = [float(val) for val in top.get("a", [])]
            window_record["top_a"] = json.dumps(top_a_values)
            if top_a_values:
                window_record["top_a0"] = top_a_values[0]
            if len(top_a_values) > 1:
                window_record["top_a1"] = top_a_values[1]
            window_record["top_stability_margin"] = float(top["stability_margin"])
            window_record["top_solver_used"] = top.get("solver_used")
            edge_margin_val = _safe_num(top.get("edge_margin"))
            buffer_margin_val = _safe_num(top.get("buffer_margin"))
            window_record["top_edge_margin"] = (
                float(edge_margin_val)
                if edge_margin_val is not None
                else float("nan")
            )
            window_record["top_buffer_margin"] = (
                float(buffer_margin_val)
                if buffer_margin_val is not None
                else float("nan")
            )
            top_t_vals = top.get("t_values") if isinstance(top, dict) else None
            window_record["top_t_vector_abs"] = json.dumps(
                [float(val) for val in (top_t_vals or [])]
            )
            window_record["top_admissible_root"] = bool(
                top.get("admissible_root", False)
            )
            alignment_angle = float("nan")
            energy_mu = float("nan")
            eigvec_val = top.get("eigvec")
            if isinstance(eigvec_val, np.ndarray):
                try:
                    angle_deg, energy_val = eval_alignment_diag(
                        cov_fit,
                        np.asarray(eigvec_val, dtype=np.float64),
                        top_p=alignment_top_p,
                    )
                    alignment_angle = float(angle_deg)
                    energy_mu = float(energy_val)
                except Exception:
                    pass
            window_record["angle_min_deg"] = alignment_angle
            window_record["energy_mu"] = energy_mu
            # Optional diagnostics populated by dealias_search
            window_record["top_z_plus"] = (
                float(top.get("z_plus", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
            window_record["top_z_plus_scm"] = (
                float(top.get("z_plus_scm", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
            window_record["top_edge_scale"] = (
                float(top.get("edge_scale", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
            window_record["top_threshold_main"] = (
                float(top.get("threshold_main", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
            window_record["top_off_component_ratio"] = (
                float(top.get("off_component_ratio", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
            components = top.get("components") if isinstance(top, dict) else None
            window_record["top_component_sigma1"] = (
                float(components[target_component])
                if isinstance(components, (list, tuple))
                and len(components) > target_component
                else float("nan")
            )
            window_record["top_component_sigma2"] = (
                float(components[1])
                if isinstance(components, (list, tuple)) and len(components) > 1
                else float("nan")
            )
        else:
            window_record["top_lambda_hat"] = float("nan")
            window_record["top_mu_hat"] = float("nan")
            window_record["top_a0"] = float("nan")
            window_record["top_a1"] = float("nan")
            window_record["top_stability_margin"] = float("nan")
            window_record["top_edge_margin"] = float("nan")
            window_record["top_buffer_margin"] = float("nan")
            window_record["top_t_vector_abs"] = json.dumps([])
            window_record["top_admissible_root"] = False
            window_record["detections_detail"] = "[]"
            window_record["top_z_plus"] = float("nan")
            window_record["top_z_plus_scm"] = float("nan")
            window_record["top_edge_scale"] = float("nan")
            window_record["top_threshold_main"] = float("nan")
            window_record["top_off_component_ratio"] = float("nan")
            window_record["top_component_sigma1"] = float("nan")
            window_record["top_component_sigma2"] = float("nan")
            window_record["angle_min_deg"] = float("nan")
            window_record["energy_mu"] = float("nan")
            window_record["angle_min_deg"] = float("nan")
            window_record["energy_mu"] = float("nan")

        # Always record the top aliased Σ1 eigenvalue (for E2-alt)
        try:
            stats_local = mean_squares(y_fit_daily, groups_fit)
            sigma1_local = stats_local["Sigma1_hat"].astype(np.float64)
            window_record["top_sigma1_eigval"] = float(
                np.linalg.eigvalsh(sigma1_local)[-1]
            )
        except Exception:
            window_record["top_sigma1_eigval"] = float("nan")

        window_record["window_index"] = int(window_idx)

        for strategy_label, cfg in strategies.items():
            if not cfg.get("available", True):
                continue
            try:
                weights = np.asarray(cfg["get_weights"](cov_fit), dtype=np.float64).reshape(-1)
            except ImportError:
                cfg["available"] = False
                continue
            except Exception:
                continue

            p_assets_window = len(ordered_tickers)
            if weights.size != p_assets_window or not np.all(np.isfinite(weights)):
                continue
            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or abs(weight_sum) < 1e-12:
                continue
            if not np.isclose(weight_sum, 1.0):
                weights = weights / weight_sum

            prefix = cfg["prefix"]
            solver_info = solver_stats.get(strategy_label)
            if solver_info:
                cond_pen = float(solver_info.get("cond_penalized", float("nan")))
                cond_orig = float(solver_info.get("cond_original", float("nan")))
                window_record[f"{prefix}_mv_cond_penalized"] = cond_pen
                window_record[f"{prefix}_mv_cond_original"] = cond_orig
                window_record[f"{prefix}_mv_iterations"] = float(solver_info.get("iterations", float("nan")))
                window_record[f"{prefix}_mv_converged"] = bool(solver_info.get("converged", False))
                condition_flag = (
                    not bool(solver_info.get("converged", False))
                    or (np.isfinite(cond_pen) and cond_pen > float(minvar_condition_cap))
                )
                window_record[f"{prefix}_mv_condition_flag"] = bool(condition_flag)
                solver_stats[strategy_label] = {}
                if condition_flag:
                    continue
            else:
                window_record[f"{prefix}_mv_condition_flag"] = False

            strategy_success[strategy_label] = True
            strategy_windows[strategy_label].append(window_idx)
            weights_history[strategy_label].append(weights.copy())

            prev = prev_weights[strategy_label]
            if prev is not None and prev.shape == weights.shape:
                turnover_value = turnover(prev, weights)
            else:
                turnover_value = 0.0
            prev_weights[strategy_label] = weights.copy()
            turnover_cost = turnover_value * float(turnover_cost_bps) / 10000.0
            turnover_cost_history[strategy_label].append(turnover_cost)
            window_record[f"{prefix}_turnover"] = float(turnover_value)
            window_record[f"{prefix}_turnover_cost"] = float(turnover_cost)

            forecast_alias, realised_alias_raw = variance_forecast_from_components(
                y_fit_daily,
                y_hold_daily,
                replicates,
                weights,
            )
            forecast_dealias, realised_de_raw = variance_forecast_from_components(
                y_fit_daily,
                y_hold_daily,
                replicates,
                weights,
                detections=detections,
            )
            forecast_lw, realised_lw_raw = oos_variance_forecast(
                fit_matrix,
                hold_matrix,
                weights,
                estimator="lw",
            )
            forecast_scm, realised_scm_raw = oos_variance_forecast(
                fit_matrix,
                hold_matrix,
                weights,
                estimator="scm",
            )

            estimator_outputs: list[tuple[str, float, float, float | None]] = [
                ("Aliased", float(forecast_alias), float(realised_alias_raw), None),
                (
                    "De-aliased",
                    float(forecast_dealias),
                    float(realised_de_raw),
                    float(realised_alias_raw),
                ),
                (
                    "Ledoit-Wolf",
                    float(forecast_lw),
                    float(realised_lw_raw),
                    float(realised_alias_raw),
                ),
                (
                    "SCM",
                    float(forecast_scm),
                    float(realised_scm_raw),
                    float(realised_alias_raw),
                ),
            ]

            try:
                forecast_oas, realised_oas_raw = oos_variance_forecast(
                    fit_matrix,
                    hold_matrix,
                    weights,
                    estimator="oas",
                )
                estimator_outputs.append(
                    (
                        "OAS",
                        float(forecast_oas),
                        float(realised_oas_raw),
                        float(realised_alias_raw),
                    )
                )
            except Exception:
                pass

            try:
                forecast_cc, realised_cc_raw = oos_variance_forecast(
                    fit_matrix,
                    hold_matrix,
                    weights,
                    estimator="cc",
                )
                estimator_outputs.append(
                    (
                        "Constant-Correlation",
                        float(forecast_cc),
                        float(realised_cc_raw),
                        float(realised_alias_raw),
                    )
                )
            except Exception:
                pass

            try:
                forecast_tyler, realised_tyler_raw = oos_variance_forecast(
                    fit_matrix,
                    hold_matrix,
                    weights,
                    estimator="tyler_shrink",
                )
                estimator_outputs.append(
                    (
                        "Tyler-Shrink",
                        float(forecast_tyler),
                        float(realised_tyler_raw),
                        float(realised_alias_raw),
                    )
                )
            except Exception:
                pass

            if factor_returns is not None:
                try:
                    forecast_factor, realised_factor_raw = oos_variance_forecast(
                        fit_matrix,
                        hold_matrix,
                        weights,
                        estimator="factor",
                        factor_returns=factor_returns,
                        asset_names=ordered_tickers,
                        fit_index=fit.index,
                    )
                    estimator_outputs.append(
                        (
                            "Factor",
                            float(forecast_factor),
                            float(realised_factor_raw),
                            float(realised_alias_raw),
                        )
                    )
                except Exception as exc:
                    if not factor_warned and estimator_mode == "factor":
                        print(
                            f"[equity-panel] observed factor baseline skipped: {exc}",
                            file=sys.stderr,
                        )
                        factor_warned = True
                try:
                    forecast_factor_obs, realised_factor_obs = oos_variance_forecast(
                        fit_matrix,
                        hold_matrix,
                        weights,
                        estimator="factor_obs",
                        factor_returns=factor_returns,
                        asset_names=ordered_tickers,
                        fit_index=fit.index,
                    )
                    estimator_outputs.append(
                        (
                            "Factor-Observed",
                            float(forecast_factor_obs),
                            float(realised_factor_obs),
                            float(realised_alias_raw),
                        )
                    )
                except Exception as exc:
                    if not factor_obs_warned:
                        print(
                            f"[equity-panel] factor_obs baseline skipped: {exc}",
                            file=sys.stderr,
                        )
                        factor_obs_warned = True

            try:
                forecast_poet, realised_poet_raw = oos_variance_forecast(
                    fit_matrix,
                    hold_matrix,
                    weights,
                    estimator="poet",
                    asset_names=ordered_tickers,
                    fit_index=fit.index,
                )
                estimator_outputs.append(
                    (
                        "POET-lite",
                        float(forecast_poet),
                        float(realised_poet_raw),
                        float(realised_alias_raw),
                    )
                )
            except Exception as exc:
                if not poet_warned:
                    print(
                        f"[equity-panel] POET baseline skipped: {exc}",
                        file=sys.stderr,
                    )
                    poet_warned = True

            hold_returns = hold_matrix @ weights
            for estimator_name, forecast_value, realised_raw, fallback_raw in estimator_outputs:
                base_value = float(realised_raw)
                if not np.isfinite(base_value) and fallback_raw is not None:
                    base_value = float(fallback_raw)
                combo_key = f"{strategy_label}::{estimator_name}"
                strategy_estimators[strategy_label].add(estimator_name)
                realized_by_combo_raw[combo_key][window_idx] = base_value
                if np.isfinite(base_value):
                    realised_adjusted = max(base_value - turnover_cost, 0.0)
                else:
                    realised_adjusted = base_value
                if np.isfinite(forecast_value) and np.isfinite(realised_adjusted):
                    error_value = float((forecast_value - realised_adjusted) ** 2)
                    qlike_val = float(
                        eval_qlike(
                            [float(forecast_value)],
                            [float(realised_adjusted)],
                        )[0]
                    )
                else:
                    error_value = float("nan")
                    qlike_val = float("nan")
                errors_by_combo[combo_key][window_idx] = error_value
                qlike_by_combo[combo_key][window_idx] = qlike_val
                var95 = -1.65 * np.sqrt(max(forecast_value, 0.0))
                var95_by_combo[combo_key].extend([var95] * hold_returns.size)
                realised_returns_by_combo[combo_key].extend(hold_returns.tolist())

                suffix = estimator_name.lower().replace("-", "").replace(" ", "_")
                window_record[f"{prefix}_{suffix}_forecast"] = float(forecast_value)
                window_record[f"{prefix}_{suffix}_realized"] = float(realised_adjusted)

            if strategy_label == baseline_name:
                var_forecasts_alias_baseline.append(float(forecast_alias))
                var_forecasts_de_baseline.append(float(forecast_dealias))
                var_forecasts_lw_baseline.append(float(forecast_lw))

        if len(window_record) > 5:
            records.append(window_record)

    if not records:
        raise ValueError("No rolling windows were evaluated after balancing.")

    for strategy_label in strategies:
        weight_seq = weights_history.get(strategy_label, [])
        if not weight_seq:
            continue
        alias_key = f"{strategy_label}::Aliased"
        raw_map = realized_by_combo_raw.get(alias_key, {})
        if not raw_map:
            continue
        indices = strategy_windows.get(strategy_label, [])
        if not indices:
            continue
        var_array = np.array([raw_map.get(idx, 0.0) for idx in indices], dtype=np.float64)
        _, cost_series = apply_turnover_cost(var_array, weight_seq, turnover_cost_bps)
        turnover_cost_history[strategy_label] = list(cost_series)

    coverage_errors: dict[str, float] = {}
    for combo_key, forecasts in var95_by_combo.items():
        realised = realised_returns_by_combo.get(combo_key, [])
        if forecasts and realised:
            metrics = risk_metrics(forecasts, realised)
            coverage_errors[combo_key] = metrics["var95_coverage_error"]
        else:
            coverage_errors[combo_key] = float("nan")

    errors_for_plot = {
        combo_key: np.array(
            [
                errors_by_combo[combo_key][idx]
                for idx in sorted(errors_by_combo[combo_key])
            ],
            dtype=np.float64,
        )
        for combo_key in errors_by_combo
        if errors_by_combo[combo_key]
    }

    if errors_for_plot:
        eval_plot_var_panel(errors_for_plot, output_dir / "E3_variance_mse")
        # Baseline-only E3 into experiments/<run>/figures
        try:
            baseline_name = "Equal Weight"
            baseline_map: dict[str, np.ndarray] = {}
            for est in (
                "Aliased",
                "De-aliased",
                "Ledoit-Wolf",
                "OAS",
                "Constant-Correlation",
                "Factor",
                "SCM",
            ):
                key = f"{baseline_name}::{est}"
                if key in errors_for_plot:
                    baseline_map[est] = errors_for_plot[key]
            if baseline_map:
                run_name = output_dir.parents[1].name if len(output_dir.parents) >= 2 else output_dir.parent.name
                e3_plot_var_mse(baseline_map, run=run_name)
        except Exception:
            pass
    if coverage_errors:
        eval_plot_cov_err(coverage_errors, output_dir / "E4_var95_coverage_error")
        # Baseline-only E4 into experiments/<run>/figures
        try:
            baseline_name = "Equal Weight"
            cov_baseline: dict[str, float] = {}
            for est in (
                "Aliased",
                "De-aliased",
                "Ledoit-Wolf",
                "OAS",
                "Constant-Correlation",
                "Factor",
                "SCM",
            ):
                key = f"{baseline_name}::{est}"
                if key in coverage_errors:
                    cov_baseline[est] = float(coverage_errors[key])
            if cov_baseline:
                run_name = output_dir.parents[1].name if len(output_dir.parents) >= 2 else output_dir.parent.name
                e4_plot_var_coverage(cov_baseline, run=run_name)
        except Exception:
            pass

    for key in ("edge_buffer", "off_component_ratio", "stability_fail", "energy_floor", "neg_mu", "other"):
        rejection_totals.setdefault(key, 0)

    total_records = len(records)
    detection_count = int(detection_windows)
    detection_rate = float(detection_count / total_records) if total_records else 0.0
    substitution_fraction = (
        float(substituted_windows) / float(total_records) if total_records else float("nan")
    )
    no_iso_count = int(gating_skip_reasons.get("no_isolated_spike", 0))
    skip_no_iso_share = (
        float(no_iso_count) / float(total_records) if total_records else float("nan")
    )
    if edge_margin_values:
        edge_array = np.asarray(edge_margin_values, dtype=np.float64)
        edge_median = float(np.median(edge_array))
        q1, q3 = np.percentile(edge_array, [25.0, 75.0])
        edge_iqr = float(q3 - q1)
    else:
        edge_median = None
        edge_iqr = None

    baseline_errors_map = errors_by_combo.get(baseline_alias_key, {})
    baseline_keys = set(baseline_errors_map.keys())

    # Build metrics summary, with paired tests and CIs for De-aliased vs LW/Aliased
    import os
    fast = os.environ.get("FAST_TESTS", "0") == "1"
    n_boot = 200 if fast else 1000
    metrics_summary = eval_build_metrics_summary(
        errors_by_combo=errors_by_combo,
        coverage_errors=coverage_errors,
        qlike_by_combo=qlike_by_combo,
        var_forecasts=var95_by_combo,
        realised_returns=realised_returns_by_combo,
        es_forecasts=None,
        label=label,
        block_len=12,
        n_boot=n_boot,
        alpha=0.05,
    )
    if not metrics_summary.empty:
        metrics_summary["substitution_fraction"] = substitution_fraction
        metrics_summary["skip_no_isolated_share"] = skip_no_iso_share
        metrics_summary["edge_margin_median"] = (
            float(edge_median) if edge_median is not None else float("nan")
        )
        metrics_summary["edge_margin_iqr"] = (
            float(edge_iqr) if edge_iqr is not None else float("nan")
        )
        metrics_summary["edge_margin_count"] = len(edge_margin_values)
        metrics_summary["edge_mode"] = edge_mode_cfg
        metrics_summary["gating_mode"] = gating_mode_value
        if delta_used_values:
            metrics_summary["delta_frac_used_min"] = float(min(delta_used_values))
            metrics_summary["delta_frac_used_max"] = float(max(delta_used_values))
        else:
            metrics_summary["delta_frac_used_min"] = float("nan")
            metrics_summary["delta_frac_used_max"] = float("nan")
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_dir / "rolling_results.csv", index=False)

    # Persist detection summary and spike timeseries (E2)
    if not results_df.empty and "top_lambda_hat" in results_df.columns:
        for col, default in {
            "skip_reason": "",
            "isolated_spikes": 0,
            "gate_discarded_count": 0,
            "gate_discarded": "[]",
            "angle_min_deg": float("nan"),
            "energy_mu": float("nan"),
        }.items():
            if col not in results_df.columns:
                results_df[col] = default
        det_summary = results_df[
            [
                "fit_start",
                "fit_end",
                "hold_start",
                "hold_end",
                "n_detections",
                "skip_reason",
                "isolated_spikes",
                "gate_discarded_count",
                "gate_discarded",
                "edge_mode",
                "gating_mode",
                "edge_scale",
                "edge_scm",
                "edge_tyler",
                "edge_selected",
                "edge_band_min",
                "edge_band_max",
                "delta_frac_used",
                "delta_frac_config",
                "delta_frac_calibrated",
                "stability_eta_deg",
                "off_component_cap",
                "top_lambda_hat",
                "top_mu_hat",
                "top_a0",
                "top_a1",
                "top_stability_margin",
                "top_edge_margin",
                "top_buffer_margin",
                "top_t_vector_abs",
                "top_admissible_root",
                "top_sigma1_eigval",
                "top_z_plus",
                "top_z_plus_scm",
                "top_edge_scale",
                "top_threshold_main",
                "top_off_component_ratio",
                "top_component_sigma1",
                "top_component_sigma2",
                "angle_min_deg",
                "energy_mu",
                "detections_detail",
            ]
        ].copy()
        det_summary.to_csv(output_dir / "detection_summary.csv", index=False)

        lambda_series = det_summary["top_lambda_hat"].to_numpy(dtype=float)
        mu_series = det_summary["top_mu_hat"].to_numpy(dtype=float)
        if np.isfinite(lambda_series).any() and np.isfinite(mu_series).any():
            x_axis = np.arange(lambda_series.shape[0])
            plot_spike_timeseries(
                x_axis,
                np.nan_to_num(lambda_series, nan=np.nan),
                np.nan_to_num(mu_series, nan=np.nan),
                out_path=output_dir / "spike_timeseries.png",
                title=f"{label.title()} - Aliased λ̂ vs De-aliased µ̂",
                xlabel="Window",
                ylabel="Spike magnitude",
            )
            try:
                run_name = output_dir.parents[1].name if len(output_dir.parents) >= 2 else output_dir.parent.name
                e2_plot_spike_timeseries(
                    x_axis,
                    np.nan_to_num(lambda_series, nan=np.nan),
                    np.nan_to_num(mu_series, nan=np.nan),
                    run=run_name,
                    title=f"{label.title()} - Aliased λ̂ vs De-aliased µ̂",
                    xlabel="Window",
                    ylabel="Spike magnitude",
                )
            except Exception:
                pass
        else:
            # E2-alt: plot top aliased Σ1 eigenvalue series when no detections
            if "top_sigma1_eigval" in det_summary.columns:
                top_alias = det_summary["top_sigma1_eigval"].to_numpy(dtype=float)
                if np.isfinite(top_alias).any():
                    x_axis = np.arange(top_alias.shape[0])
                    plot_spike_timeseries(
                        x_axis,
                        np.nan_to_num(top_alias, nan=np.nan),
                        np.nan_to_num(top_alias, nan=np.nan),
                        out_path=output_dir / "spike_timeseries.png",
                        title=f"{label.title()} - Top aliased Σ1 eigenvalue (E2 alt)",
                        xlabel="Window",
                        ylabel="Eigenvalue",
                    )
                    try:
                        run_name = output_dir.parents[1].name if len(output_dir.parents) >= 2 else output_dir.parent.name
                        e2_plot_spike_timeseries(
                            x_axis,
                            np.nan_to_num(top_alias, nan=np.nan),
                            np.nan_to_num(top_alias, nan=np.nan),
                            run=run_name,
                            title=f"{label.title()} - Top aliased Σ1 eigenvalue (E2 alt)",
                            xlabel="Window",
                            ylabel="Eigenvalue",
                        )
                    except Exception:
                        pass

    de_scoped_equity = False
    if design_mode == "nested" and records:
        skip_reasons_seq = [str(item.get("skip_reason", "")) for item in records]
        isolated_series = [int(item.get("isolated_spikes", 0) or 0) for item in records]
        if skip_reasons_seq and all(reason == "no_isolated_spike" for reason in skip_reasons_seq):
            de_scoped_equity = True
        elif isolated_series and all(count == 0 for count in isolated_series):
            de_scoped_equity = True

    summary_payload: dict[str, Any] = {
        "label": label,
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "balanced_weeks": int(weekly_balanced.shape[0]),
        "dropped_weeks": int(dropped_weeks),
        "imputed_weeks": int(balanced_panel.imputed_weeks),
        "partial_week_policy": partial_week_policy,
        "window_weeks": int(window_weeks),
        "horizon_weeks": int(horizon_weeks),
        "rolling_windows_evaluated": len(records),
        "detection_windows": detection_count,
        "detection_rate": detection_rate,
        "substitution_fraction": substitution_fraction,
        "skip_no_isolated_share": skip_no_iso_share,
        "replicates_per_week": int(replicates),
        "n_assets": int(weekly_balanced.shape[1]),
        "strategies": {name: bool(strategy_success[name]) for name in strategies},
        "edge_margin_stats": {
            "count": len(edge_margin_values),
            "median": edge_median,
            "iqr": edge_iqr,
        },
        "design": design_mode,
        "edge_mode": edge_mode_cfg,
        "edge_huber_c": float(edge_huber_c_val),
        "nested_replicates": int(nested_reps_value),
        "estimator": estimator_mode,
        "preprocess_flags": dict(preprocess_flags or {}),
        "solver_used": sorted(solver_usage),
        "alignment_top_p": int(alignment_top_p),
        "gating_mode": gating_mode_value,
    }
    if delta_used_values:
        summary_payload["delta_frac_used_min"] = float(min(delta_used_values))
        summary_payload["delta_frac_used_max"] = float(max(delta_used_values))
    if de_scoped_equity:
        nested_note: dict[str, Any] = {"de_scoped_equity": True}
        if design_mode == "nested" and records:
            total_windows_nested = len(records)
            if total_windows_nested > 0 and all(int(item.get("isolated_spikes", 0) or 0) == 0 for item in records):
                nested_note["reason"] = f"no isolated spikes across {total_windows_nested} nested windows"
        summary_payload["nested_scope"] = nested_note
    if crisis_label is not None:
        summary_payload["crisis_label"] = str(crisis_label)
    if design_mode == "nested":
        skipped_total = int(sum(nested_skip_reasons.values()))
        summary_payload["nested_windows_skipped"] = skipped_total
        if nested_skip_reasons:
            summary_payload["nested_skip_reasons"] = [
                {"reason": reason, "count": int(count)}
                for reason, count in sorted(
                    nested_skip_reasons.items(), key=lambda item: (-item[1], item[0])
                )
            ]
        if nested_skip_detail_map:
            summary_payload["nested_skip_details"] = [
                {
                    "exit_reason": detail.get("exit_reason", reason_key),
                    "windows": int(detail.get("windows", 0)),
                    "years_kept": int(detail.get("years_kept", 0)),
                    "weeks_common": int(detail.get("weeks_common", 0)),
                    "replicates": int(detail.get("replicates", nested_reps_value)),
                    **(
                        {"replicates_observed": int(detail.get("replicates_observed", 0))}
                        if "replicates_observed" in detail
                        else {}
                    ),
                }
                for reason_key, detail in sorted(
                    nested_skip_detail_map.items(),
                    key=lambda item: (-int(item[1].get("windows", 0)), item[0]),
                )
            ]
    summary_payload["rejection_stats"] = rejection_totals
    gating_summary: dict[str, Any] = {
        "enabled": bool(gating_enabled),
        "q_max": int(gating_q_max),
        "require_isolated": bool(gating_require_isolated),
        "windows_substituted": int(substituted_windows),
        "mode": gating_mode_value,
    }
    if delta_frac_config is not None:
        gating_summary["delta_frac_config"] = float(delta_frac_config)
    if delta_used_values:
        gating_summary["delta_frac_used_min"] = float(min(delta_used_values))
        gating_summary["delta_frac_used_max"] = float(max(delta_used_values))
    if delta_usage_records:
        gating_summary["delta_frac_windows"] = delta_usage_records
    if gating_mode_value == "calibrated":
        gating_summary["calibration_path"] = str(calibration_path)
        if calibration_misses:
            gating_summary["calibration_missing"] = [
                {
                    "edge_mode": mode,
                    "p": int(p_val),
                    "t": int(t_val),
                }
                for mode, p_val, t_val in sorted(calibration_misses)
            ]
    if gating_skip_reasons:
        gating_summary["skip_reasons"] = [
            {"reason": reason, "count": int(count)}
            for reason, count in sorted(
                gating_skip_reasons.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        gating_summary["skip_windows"] = int(sum(gating_skip_reasons.values()))
    if gating_discard_log:
        total_discarded = sum(len(entry.get("discarded", [])) for entry in gating_discard_log)
        gating_summary["discarded_total"] = int(total_discarded)
    summary_payload["gating"] = gating_summary
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    de_series_baseline = var_forecasts_de_baseline
    if (
        var_forecasts_alias_baseline
        and de_series_baseline
        and len(var_forecasts_alias_baseline) == len(de_series_baseline)
    ):
        x_axis = np.arange(len(var_forecasts_alias_baseline))
        plot_spike_timeseries(
            x_axis,
            var_forecasts_alias_baseline,
            de_series_baseline,
            out_path=output_dir / "variance_forecasts.png",
            title=f"{label.title()} - Forecast variance comparison",
            xlabel="Window",
            ylabel="Variance",
        )

    baseline_var95 = var95_by_combo.get(baseline_alias_key, [])
    de_var95 = var95_by_combo.get(f"{baseline_name}::De-aliased", [])
    if baseline_var95 and de_var95 and len(baseline_var95) == len(de_var95):
        plot_spike_timeseries(
            np.arange(len(baseline_var95)),
            baseline_var95,
            de_var95,
            out_path=output_dir / "var95_forecasts.png",
            title=f"{label.title()} - 95% VaR comparison",
            xlabel="Hold observation",
            ylabel="VaR",
        )

    if sigma_ablation:
        _run_sigma_ablation(
            daily_subset,
            output_dir,
            cs_drop_top_frac,
            delta,
            delta_frac_config,
            eps,
            stability_eta,
            signed_a,
            target_component,
        )

    # Lightweight consistency check: if detections occurred, ensure de-aliased
    # forecasts differ from aliased ones for at least one strategy in each such
    # window.
    try:
        results_csv = output_dir / "rolling_results.csv"
        if results_csv.exists():
            check_dealiased_applied(pd.read_csv(results_csv))
    except Exception:
        # Do not fail the entire experiment on diagnostic issues
        pass


def _run_sigma_ablation(
    daily_returns: pd.DataFrame,
    output_dir: Path,
    cs_drop_top_frac: float,
    delta: float,
    delta_frac: float | None,
    eps: float,
    stability_eta: float,
    signed_a: bool,
    target_component: int,
) -> None:
    """Evaluate Cs perturbations and persist sensitivity diagnostics."""

    try:
        balanced_obs, groups, _ = balance_weeks(daily_returns)
    except ValueError as exc:
        pd.DataFrame(
            [
                {
                    "scale": np.nan,
                    "mp_edge": np.nan,
                    "n_detections": 0,
                    "top_lambda": np.nan,
                    "top_mu": np.nan,
                    "error": str(exc),
                }
            ]
        ).to_csv(output_dir / "sigma_ablation.csv", index=False)
        return

    stats = mean_squares(balanced_obs, groups)
    ms1 = stats["MS1"].astype(np.float64)
    ms2 = stats["MS2"].astype(np.float64)
    p_dim = ms1.shape[0]
    d_vec = np.array(
        [float(stats["I"] - 1), float(stats["n"] - stats["I"])],
        dtype=np.float64,
    )
    c_vec = np.array([float(stats["J"]), 1.0], dtype=np.float64)
    drop_frac = float(max(cs_drop_top_frac, 0.0))
    drop_top = min(p_dim - 1, max(1, int(round(p_dim * drop_frac))))
    cs_base = estimate_Cs_from_MS([ms1, ms2], d_vec, c_vec, drop_top=drop_top)
    design_c = np.ones_like(cs_base, dtype=np.float64)
    base_a = np.zeros_like(cs_base, dtype=np.float64)
    if base_a.size:
        base_a[0] = 1.0
    n_val = float(stats["J"])

    records: list[dict[str, Any]] = []
    for scale in (0.9, 1.0, 1.1):
        cs_scaled = cs_base * scale
        try:
            detections = dealias_search(
                balanced_obs,
                groups,
                target_r=target_component,
                Cs=cs_scaled,
                delta=delta,
                delta_frac=delta_frac,
                eps=eps,
                stability_eta_deg=stability_eta,
                use_tvector=True,
                nonnegative_a=not signed_a,
            )
            error_msg = ""
        except Exception as exc:  # pragma: no cover - defensive
            detections = []
            error_msg = str(exc)

        edge_val = mp_edge(base_a, design_c, d_vec, n_val, Cs=cs_scaled)

        record: dict[str, Any] = {
            "scale": float(scale),
            "mp_edge": float(edge_val),
            "n_detections": int(len(detections)),
            "top_lambda": float(detections[0]["lambda_hat"]) if detections else np.nan,
            "top_mu": float(detections[0]["mu_hat"]) if detections else np.nan,
            "error": error_msg,
        }
        for idx, value in enumerate(cs_scaled):
            record[f"Cs_{idx}"] = float(value)
        records.append(record)

    pd.DataFrame(records).to_csv(output_dir / "sigma_ablation.csv", index=False)


def run_experiment(
    config_path: Path | str | None = None,
    *,
    sigma_ablation: bool = False,
    crisis: str | None = None,
    delta_frac_override: float | None = None,
    signed_a_override: bool | None = None,
    target_component_override: int | None = None,
    design_override: str | None = None,
    nested_replicates_override: int | None = None,
    oneway_a_solver_override: str | None = None,
    cs_drop_top_frac_override: float | None = None,
    progress_override: bool | None = None,
    eps_override: float | None = None,
    a_grid_override: int | None = None,
    ablations: bool | None = None,
    eta_override: float | None = None,
    window_weeks_override: int | None = None,
    horizon_weeks_override: int | None = None,
    energy_min_abs_override: float | None = None,
    partial_week_policy: str | None = None,
    precompute_panel: bool = False,
    cache_dir_override: str | None = None,
    resume_cache: bool = False,
    estimator_override: str | None = None,
    winsorize_q_override: float | None = None,
    huber_c_override: float | None = None,
    factor_csv_override: str | None = None,
    minvar_ridge_override: float | None = None,
    minvar_box_override: str | None = None,
    turnover_cost_override: float | None = None,
    minvar_condition_cap_override: float | None = None,
    edge_mode_override: str | None = None,
    edge_huber_c_override: float | None = None,
    gating_mode_override: str | None = None,
    gating_calibration_override: str | None = None,
    exec_mode: str | None = None,
) -> None:
    """Execute the rolling equity forecasting experiment."""

    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("config.yaml")
    )
    config = load_config(path)
    if delta_frac_override is not None:
        config["dealias_delta_frac"] = float(delta_frac_override)
    if signed_a_override is not None:
        config["signed_a"] = bool(signed_a_override)
    if target_component_override is not None:
        config["target_component"] = int(target_component_override)
    if design_override is not None:
        config["design"] = str(design_override)
    if nested_replicates_override is not None:
        config["nested_replicates"] = int(nested_replicates_override)
    if oneway_a_solver_override is not None:
        config["oneway_a_solver"] = str(oneway_a_solver_override)
    if cs_drop_top_frac_override is not None:
        config["cs_drop_top_frac"] = float(cs_drop_top_frac_override)
    if eps_override is not None:
        config["dealias_eps"] = float(eps_override)
    if a_grid_override is not None:
        config["a_grid"] = int(a_grid_override)
    if eta_override is not None:
        config["stability_eta_deg"] = float(eta_override)
    if window_weeks_override is not None:
        config["window_weeks"] = int(window_weeks_override)
    if horizon_weeks_override is not None:
        config["horizon_weeks"] = int(horizon_weeks_override)
    if energy_min_abs_override is not None:
        config["energy_min_abs"] = float(energy_min_abs_override)
    if estimator_override is not None:
        config["estimator"] = str(estimator_override)
    if winsorize_q_override is not None:
        config["winsorize_q"] = float(winsorize_q_override)
    if huber_c_override is not None:
        config["huber_c"] = float(huber_c_override)
    if factor_csv_override is not None:
        config["factor_csv"] = factor_csv_override
    if minvar_ridge_override is not None:
        config["minvar_ridge"] = float(minvar_ridge_override)
    if minvar_box_override is not None:
        config["minvar_box"] = minvar_box_override
    if turnover_cost_override is not None:
        config["turnover_cost_bps"] = float(turnover_cost_override)
    if minvar_condition_cap_override is not None:
        config["minvar_condition_cap"] = float(minvar_condition_cap_override)
    if edge_mode_override is not None:
        config["edge_mode"] = str(edge_mode_override)
    if edge_huber_c_override is not None:
        config["edge_huber_c"] = float(edge_huber_c_override)
    if exec_mode is not None:
        config["exec_mode"] = str(exec_mode)
    gating_cfg_overrides = dict(config.get("gating", {}) or {})
    if gating_mode_override is not None:
        gating_cfg_overrides["mode"] = str(gating_mode_override)
    if gating_calibration_override is not None:
        gating_cfg_overrides["calibration_path"] = str(gating_calibration_override)
    if gating_cfg_overrides:
        config["gating"] = gating_cfg_overrides
    panel_policy = str(
        partial_week_policy
        if partial_week_policy is not None
        else config.get("partial_week_policy", "drop")
    )
    if panel_policy not in {"drop", "impute"}:
        raise ValueError("partial_week_policy must be 'drop' or 'impute'.")
    config["partial_week_policy"] = panel_policy

    design_value = str(config.get("design", "oneway")).lower()
    if design_value not in {"oneway", "nested", "dow", "vol"}:
        raise ValueError("design must be one of {'oneway', 'nested', 'dow', 'vol'}.")
    config["design"] = design_value
    nested_reps_cfg = int(config.get("nested_replicates", 5))
    if nested_reps_cfg <= 0:
        nested_reps_cfg = 5
    config["nested_replicates"] = nested_reps_cfg
    if design_value == "nested" and int(config.get("target_component", 0)) >= 3:
        config["target_component"] = 0
    solver_value = str(config.get("oneway_a_solver", "auto")).lower()
    if solver_value not in {"auto", "rootfind", "grid"}:
        raise ValueError("oneway_a_solver must be 'auto', 'rootfind', or 'grid'.")
    config["oneway_a_solver"] = solver_value
    minvar_lo, minvar_hi = _parse_box_bounds(config.get("minvar_box"))
    config["minvar_box"] = [float(minvar_lo), float(minvar_hi)]
    minvar_ridge_val = float(config.get("minvar_ridge", 1e-4))
    if minvar_ridge_val < 0.0:
        raise ValueError("minvar_ridge must be non-negative.")
    config["minvar_ridge"] = minvar_ridge_val
    turnover_cost_bps = float(config.get("turnover_cost_bps", 5.0))
    if turnover_cost_bps < 0.0:
        raise ValueError("turnover_cost_bps must be non-negative.")
    config["turnover_cost_bps"] = turnover_cost_bps
    condition_cap_val = float(config.get("minvar_condition_cap", 1e9))
    if condition_cap_val <= 0.0:
        raise ValueError("minvar_condition_cap must be positive.")
    config["minvar_condition_cap"] = condition_cap_val
    winsorize_q_cfg = config.get("winsorize_q")
    huber_c_cfg = config.get("huber_c")
    if winsorize_q_cfg is not None and huber_c_cfg is not None:
        raise ValueError("winsorize_q and huber_c preprocessing are mutually exclusive.")
    winsorize_q_val = None
    if winsorize_q_cfg is not None:
        winsorize_q_val = float(winsorize_q_cfg)
        if not 0.0 < winsorize_q_val < 0.5:
            raise ValueError("winsorize_q must be between 0 and 0.5.")
        config["winsorize_q"] = winsorize_q_val
    huber_c_val = None
    if huber_c_cfg is not None:
        huber_c_val = float(huber_c_cfg)
        if huber_c_val <= 0.0:
            raise ValueError("huber_c must be positive.")
        config["huber_c"] = huber_c_val
    estimator_value = str(config.get("estimator", "dealias")).lower()
    allowed_estimators = {
        "aliased",
        "dealias",
        "lw",
        "oas",
        "cc",
        "factor",
        "tyler_shrink",
        "factor_obs",
        "poet",
    }
    if estimator_value not in allowed_estimators:
        raise ValueError(
            f"Unsupported estimator '{estimator_value}'. "
            f"Valid options: {', '.join(sorted(allowed_estimators))}."
        )
    config["estimator"] = estimator_value

    edge_mode_value = str(config.get("edge_mode", "scm")).lower()
    if edge_mode_value not in {"scm", "tyler", "huber"}:
        raise ValueError("edge_mode must be one of {'scm', 'tyler', 'huber'}.")
    config["edge_mode"] = edge_mode_value
    edge_huber_c_val = float(config.get("edge_huber_c", 1.5))
    if edge_mode_value == "huber":
        if edge_huber_c_val <= 0.0:
            raise ValueError("edge_huber_c must be positive when edge_mode='huber'.")
    else:
        edge_huber_c_val = max(edge_huber_c_val, 1.0)
    config["edge_huber_c"] = float(edge_huber_c_val)

    factor_returns: pd.DataFrame | None = None
    factor_csv_cfg = config.get("factor_csv")
    if factor_csv_cfg:
        factor_path = Path(str(factor_csv_cfg)).expanduser()
        if not factor_path.exists():
            raise FileNotFoundError(
                f"Factor CSV not found at '{factor_path}'."
            )
        factor_df = pd.read_csv(factor_path)
        if factor_df.empty or factor_df.shape[1] < 2:
            raise ValueError("Factor CSV must contain a date column and at least one factor column.")
        date_col_candidates = [
            col
            for col in factor_df.columns
            if str(col).lower() in {"date", "timestamp", "time", "week", "period"}
        ]
        date_col = date_col_candidates[0] if date_col_candidates else factor_df.columns[0]
        factor_df[date_col] = pd.to_datetime(factor_df[date_col])
        factor_df = factor_df.set_index(date_col).sort_index()
        factor_df = factor_df.apply(pd.to_numeric, errors="coerce")
        factor_df = factor_df.dropna(how="all")
        factor_df = factor_df.loc[~factor_df.index.duplicated(keep="last")]
        if factor_df.empty:
            raise ValueError("Factor CSV contains no usable numeric data after cleaning.")
        factor_returns = factor_df

    if factor_returns is None and estimator_value in {"factor", "factor_obs"}:
        print(
            "[equity-panel] factor baseline requested but no factors.csv provided; skipping.",
            file=sys.stderr,
        )

    try:
        alignment_top_p_cfg = int(config.get("alignment_top_p", 3))
    except (TypeError, ValueError):
        alignment_top_p_cfg = 3
    if alignment_top_p_cfg <= 0:
        alignment_top_p_cfg = 3

    cache_dir_path: Path | None
    if cache_dir_override is not None:
        cache_dir_path = Path(cache_dir_override).expanduser()
        cache_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir_path = None
    # Values from YAML remain if overrides not provided
    raw_daily_returns = _prepare_data(config)
    daily_returns, preprocess_flags = _apply_preprocessing(
        raw_daily_returns,
        winsorize_q=winsorize_q_val,
        huber_c=huber_c_val,
    )
    config["preprocess_flags"] = dict(preprocess_flags)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist resolved configuration for reproducibility
    try:
        with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=True)
    except Exception:
        pass

    runs: list[dict[str, Any]] = [
        {
            "label": "full",
            "start": config["start_date"],
            "end": config["end_date"],
            "sigma_ablation": sigma_ablation,
            "crisis_label": None,
        }
    ]

    estimator_value = str(config.get("estimator", "dealias")).strip().lower()
    preprocess_tag = (
        "none"
        if not preprocess_flags
        else "-".join(f"{k}{preprocess_flags[k]}" for k in sorted(preprocess_flags))
    )
    run_suffix = (
        f"{design_value}_J{nested_reps_cfg}_solver-{solver_value}_est-{estimator_value}_prep-{preprocess_tag}"
    ).replace(" ", "-")
    ablation_only = bool(config.get("ablation_only", False))

    if crisis:
        try:
            crisis_start_str, crisis_end_str = crisis.split(":")
        except ValueError as exc:  # pragma: no cover - input validation
            raise ValueError(
                "Crisis window must be specified as 'YYYY-MM-DD:YYYY-MM-DD'."
            ) from exc

        crisis_start = pd.to_datetime(crisis_start_str.strip())
        crisis_end = pd.to_datetime(crisis_end_str.strip())
        if crisis_start > crisis_end:
            raise ValueError("Crisis start date must be on or before the end date.")

        crisis_label = (
            f"crisis_{crisis_start.strftime('%Y%m%d')}_{crisis_end.strftime('%Y%m%d')}"
        )
        runs.append(
            {
                "label": crisis_label,
                "start": crisis_start,
                "end": crisis_end,
                "sigma_ablation": False,
                "crisis_label": crisis_label,
            }
        )

    if ablation_only and bool(ablations):
        primary_dir = output_dir / run_suffix if run_suffix else output_dir
        primary_dir.mkdir(parents=True, exist_ok=True)
    else:
        for run_cfg in runs:
            crisis_tag = run_cfg.get("crisis_label")
            if run_suffix:
                dir_suffix = run_suffix if not crisis_tag else f"{run_suffix}__{crisis_tag}"
                run_output_dir = output_dir / dir_suffix
            else:
                dir_suffix = crisis_tag or ""
                run_output_dir = output_dir / dir_suffix if dir_suffix else output_dir
            run_output_dir.mkdir(parents=True, exist_ok=True)
            label_with_suffix = (
                f"{run_cfg['label']}_{run_suffix}" if run_suffix else str(run_cfg["label"])
            )
            _run_single_period(
                daily_returns,
                start=run_cfg["start"],
                end=run_cfg["end"],
                output_dir=run_output_dir,
                window_weeks=int(config["window_weeks"]),
                horizon_weeks=int(config["horizon_weeks"]),
                delta=float(config.get("dealias_delta", 0.0)),
                delta_frac=cast(float | None, config.get("dealias_delta_frac")),
                eps=float(config.get("dealias_eps", 0.03)),
                stability_eta=float(config.get("stability_eta_deg", 1.0)),
                signed_a=bool(config.get("signed_a", True)),
                target_component=int(config.get("target_component", 0)),
                partial_week_policy=panel_policy,
                precompute_panel=precompute_panel,
                cache_dir=cache_dir_path,
                resume_cache=resume_cache,
                cs_drop_top_frac=float(config.get("cs_drop_top_frac", 0.05)),
                cs_sensitivity_frac=float(config.get("cs_sensitivity_frac", 0.0)),
                off_component_leak_cap=cast(float | None, config.get("off_component_leak_cap")),
                sigma_ablation=bool(run_cfg["sigma_ablation"]),
                label=str(label_with_suffix),
                crisis_label=str(crisis_tag) if crisis_tag else None,
                design_mode=str(config["design"]),
                nested_replicates=int(config["nested_replicates"]),
                oneway_a_solver=str(config["oneway_a_solver"]),
                estimator=str(config["estimator"]),
                progress=(True if progress_override is None else bool(progress_override)),
                a_grid=int(config.get("a_grid", 180)),
                energy_min_abs=cast(float | None, config.get("energy_min_abs")),
                factor_returns=factor_returns,
                minvar_ridge=minvar_ridge_val,
                minvar_box=(float(minvar_lo), float(minvar_hi)),
                turnover_cost_bps=turnover_cost_bps,
                minvar_condition_cap=condition_cap_val,
                preprocess_flags=preprocess_flags,
                gating=cast(Mapping[str, Any] | None, config.get("gating")),
                alignment_top_p=alignment_top_p_cfg,
                edge_mode=str(config.get("edge_mode", "scm")),
                edge_huber_c=float(config.get("edge_huber_c", 1.5)),
            )

            try:
                write_run_meta(
                    run_output_dir,
                    config=config,
                    delta=float(config.get("dealias_delta", 0.3)),
                    delta_frac=cast(float | None, config.get("dealias_delta_frac")),
                    a_grid=int(config.get("a_grid", 120)),
                    signed_a=bool(config.get("signed_a", True)),
                    sigma2_plugin=(
                        f"Cs_from_MS_drop_top_frac={float(config.get('cs_drop_top_frac', 0.1))}"
                    ),
                    code_signature_hash=CODE_SIGNATURE,
                    estimator=str(config.get("estimator", "dealias")),
                    design=design_value,
                    nested_replicates=nested_reps_cfg,
                    preprocess_flags=preprocess_flags,
                    label=label_with_suffix,
                    crisis_label=str(crisis_tag) if crisis_tag else None,
                    edge_mode=edge_mode_value,
                    exec_mode=config.get("exec_mode"),
                )
            except Exception:
                # Best effort; do not fail the entire run
                pass

    # Parameter ablations (E5)
    if bool(ablations):
        _run_param_ablation(
            daily_returns,
            output_dir / run_suffix,
            partial_week_policy=panel_policy,
            target_component=int(config.get("target_component", 0)),
            base_delta=float(config.get("dealias_delta", 0.0)),
            base_delta_frac=cast(float | None, config.get("dealias_delta_frac")),
            base_eps=float(config.get("dealias_eps", 0.03)),
            base_eta=float(config.get("stability_eta_deg", 0.4)),
            signed_a=bool(config.get("signed_a", True)),
            off_component_leak_cap=cast(float | None, config.get("off_component_leak_cap")),
            energy_min_abs=cast(float | None, config.get("energy_min_abs")),
            oneway_a_solver=str(config["oneway_a_solver"]),
            preprocess_flags=preprocess_flags,
            grid_overrides=cast(Mapping[str, Iterable[Any]] | None, config.get("ablation_grid")),
        )


def main() -> None:
    """Entry point for CLI execution."""

    parser = argparse.ArgumentParser(description="Equity panel forecasting experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--design",
        type=str,
        choices=["oneway", "nested", "dow", "vol"],
        default=None,
        help="Balanced design structure (default: oneway).",
    )
    parser.add_argument(
        "--nested-replicates",
        type=int,
        default=None,
        help="Replicates per (year, week) cell for nested design (default 5).",
    )
    parser.add_argument(
        "--oneway-a-solver",
        type=str,
        choices=["auto", "rootfind", "grid"],
        default=None,
        help="Refinement mode for one-way a-grid search (default auto).",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["aliased", "dealias", "lw", "oas", "cc", "factor", "tyler_shrink"],
        default=None,
        help="Primary covariance estimator to emphasise (default dealias).",
    )
    parser.add_argument(
        "--factor-csv",
        type=str,
        default=None,
        help="Optional CSV of factor returns (date-indexed) for observed-factor covariance.",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        choices=["scm", "tyler", "huber"],
        default=None,
        help="Edge estimation mode for detection margins (default: scm).",
    )
    parser.add_argument(
        "--edge-huber-c",
        type=float,
        default=None,
        help="Huber threshold used when --edge-mode=huber (default: 1.5).",
    )
    parser.add_argument(
        "--gating-mode",
        type=str,
        choices=["fixed", "calibrated"],
        default=None,
        help="Gating mode override (fixed or calibrated).",
    )
    parser.add_argument(
        "--exec-mode",
        type=str,
        choices=["deterministic", "throughput"],
        default="deterministic",
        help="Execution profile controlling BLAS threads (default: deterministic).",
    )
    parser.add_argument(
        "--gating-calibration",
        type=str,
        default=None,
        help="Path to calibrated delta thresholds JSON (used when gating mode is calibrated).",
    )
    parser.add_argument(
        "--minvar-ridge",
        type=float,
        default=None,
        help="Ridge parameter for box-constrained min-variance weights (lambda^2).",
    )
    parser.add_argument(
        "--minvar-box",
        type=str,
        default=None,
        help="Per-asset weight bounds for min-var as 'lo,hi' (default 0,0.1).",
    )
    parser.add_argument(
        "--minvar-condition-cap",
        type=float,
        default=None,
        help="Maximum acceptable condition number for the penalised covariance (default 1e9).",
    )
    parser.add_argument(
        "--turnover-cost",
        type=float,
        default=None,
        help="One-way turnover cost in basis points applied per rebalance.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Placeholder for backward compatibility; ignored.",
    )
    parser.add_argument(
        "--assets-top",
        type=int,
        default=None,
        help="Placeholder for backward compatibility; ignored.",
    )
    parser.add_argument(
        "--stride-windows",
        type=int,
        default=None,
        help="Placeholder for backward compatibility; ignored.",
    )
    parser.add_argument(
        "--sigma-ablation",
        action="store_true",
        help="Run ±10%% Cs perturbation ablation and persist diagnostics.",
    )
    parser.add_argument(
        "--crisis",
        type=str,
        default=None,
        help="Optional crisis window as 'YYYY-MM-DD:YYYY-MM-DD' for a focused rerun.",
    )
    parser.add_argument(
        "--delta-frac",
        type=float,
        default=None,
        help="Relative delta buffer (fraction of MP edge)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="t-vector acceptance threshold (epsilon)",
    )
    parser.add_argument(
        "--a-grid",
        type=int,
        default=None,
        help="Number of angular grid points for a (S^1)",
    )
    parser.add_argument(
        "--off-leak",
        type=float,
        default=None,
        help="Off-component leakage cap (relative to target t-value)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Stability perturbation in degrees",
    )
    parser.add_argument(
        "--energy-min-abs",
        type=float,
        default=None,
        help="Absolute component-energy gate for Σ̂ target component (optional)",
    )
    parser.add_argument(
        "--window-weeks",
        type=int,
        default=None,
        help="Override rolling window length in weeks",
    )
    parser.add_argument(
        "--horizon-weeks",
        type=int,
        default=None,
        help="Override holdout horizon length in weeks",
    )
    parser.add_argument(
        "--signed-a",
        dest="signed_a_override",
        action="store_const",
        const=True,
        default=None,
        help="Enable signed a-grid search (default true)",
    )
    parser.add_argument(
        "--nonnegative-a",
        dest="signed_a_override",
        action="store_const",
        const=False,
        help="Restrict a-grid to nonnegative combinations",
    )
    parser.add_argument(
        "--target-component",
        type=int,
        default=None,
        help="Target component index (0 or 1)",
    )
    parser.add_argument(
        "--cs-drop-top-frac",
        type=float,
        default=None,
        help="Fraction of top eigenvalues dropped when estimating Cs",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Run parameter ablations and emit E5 outputs",
    )
    parser.add_argument(
        "--precompute-panel",
        action="store_true",
        help="Cache the balanced Week×Day panel and write its manifest for reuse.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for per-window cache artifacts (JSON/NPZ).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached per-window statistics when available.",
    )
    preprocess_group = parser.add_mutually_exclusive_group()
    preprocess_group.add_argument(
        "--winsorize",
        type=float,
        default=None,
        help="Clip returns to [q, 1-q] quantiles column-wise before balancing.",
    )
    preprocess_group.add_argument(
        "--huber",
        type=float,
        default=None,
        help="Huber clip returns at median ± c·MAD column-wise before balancing.",
    )
    partial_group = parser.add_mutually_exclusive_group()
    partial_group.add_argument(
        "--drop-partial-weeks",
        dest="partial_week_policy",
        action="store_const",
        const="drop",
        default=None,
        help="Drop weeks with fewer than the required business days (default).",
    )
    partial_group.add_argument(
        "--impute-partial-weeks",
        dest="partial_week_policy",
        action="store_const",
        const="impute",
        help="Impute missing business days within a week before balancing.",
    )
    args = parser.parse_args()
    exec_settings = runtime.configure_exec_mode(args.exec_mode)

    run_experiment(
        args.config,
        sigma_ablation=args.sigma_ablation,
        crisis=args.crisis,
        delta_frac_override=args.delta_frac,
        signed_a_override=args.signed_a_override,
        target_component_override=args.target_component,
        design_override=args.design,
        nested_replicates_override=args.nested_replicates,
        oneway_a_solver_override=args.oneway_a_solver,
        cs_drop_top_frac_override=args.cs_drop_top_frac,
        progress_override=(not args.no_progress),
        eps_override=args.eps,
        a_grid_override=args.a_grid,
        ablations=args.ablations,
        eta_override=args.eta,
        window_weeks_override=args.window_weeks,
        horizon_weeks_override=args.horizon_weeks,
        energy_min_abs_override=args.energy_min_abs,
        partial_week_policy=args.partial_week_policy,
        precompute_panel=args.precompute_panel,
        cache_dir_override=args.cache_dir,
        resume_cache=args.resume,
        estimator_override=args.estimator,
        winsorize_q_override=args.winsorize,
        huber_c_override=args.huber,
        factor_csv_override=args.factor_csv,
        minvar_ridge_override=args.minvar_ridge,
        minvar_box_override=args.minvar_box,
        turnover_cost_override=args.turnover_cost,
        minvar_condition_cap_override=args.minvar_condition_cap,
        edge_mode_override=args.edge_mode,
        edge_huber_c_override=args.edge_huber_c,
        gating_mode_override=args.gating_mode,
        gating_calibration_override=args.gating_calibration,
        exec_mode=exec_settings.mode,
    )


if __name__ == "__main__":
    main()
