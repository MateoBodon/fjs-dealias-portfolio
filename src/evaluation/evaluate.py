from __future__ import annotations

"""
Evaluation utilities for portfolio forecast accuracy and coverage.

This module provides:
- Var-MSE and VaR(95%) coverage error summaries for EW, MinVar(box),
  and MinVar(long-only) across rolling windows.
- Paired sign-test p-values for window-level ΔMSE comparisons
  (De-aliased minus Ledoit–Wolf, and De-aliased minus Aliased).
- Moving block bootstrap confidence intervals for the median ΔMSE
  (default block length = 12 weeks).

It is designed to operate on window-level artifacts produced by the equity
rolling evaluation, but can be used with any mapping from method -> per-window
errors and per-observation VaR forecasts/realised returns.
"""

from dataclasses import dataclass
from math import erfc, log, sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from .dm import dm_test

# --- Basic statistics -------------------------------------------------------


def iqr(values: Iterable[float] | np.ndarray) -> float:
    """Interquartile range (75th - 25th percentile).

    Parameters
    ----------
    values
        Sequence or array of values.

    Returns
    -------
    float
        IQR or NaN when the input is empty.
    """

    arr = np.asarray(list(values), dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan")
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    return float(q75 - q25)


def sign_test_pvalue(differences: Iterable[float] | np.ndarray) -> float:
    """Two-sided sign test p-value for paired differences.

    Small p-values indicate the median of differences is unlikely to be zero.

    Parameters
    ----------
    differences
        Paired differences (e.g., ΔMSE per window).

    Returns
    -------
    float
        Two-sided p-value, or NaN when no finite differences are provided.
    """

    arr = np.asarray(list(differences), dtype=np.float64).ravel()
    mask = np.isfinite(arr) & (np.abs(arr) > 1e-12)
    arr = arr[mask]
    n = int(arr.size)
    if n == 0:
        return float("nan")
    positives = int(np.count_nonzero(arr > 0))
    tail = min(positives, n - positives)
    # Compute 2 * sum_{k <= tail} Binomial(n, 0.5)
    # Use log-sum-exp stability when n is large
    from math import comb

    cumulative = sum(comb(n, k) for k in range(0, tail + 1)) * (0.5**n)
    p_value = min(1.0, 2.0 * cumulative)
    return float(p_value)


def qlike(
    forecasts: Iterable[float] | np.ndarray,
    realised: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Quasi-likelihood (QLIKE) loss for variance forecasts."""

    f_arr = np.asarray(list(forecasts), dtype=np.float64).ravel()
    r_arr = np.asarray(list(realised), dtype=np.float64).ravel()
    if f_arr.size != r_arr.size:
        raise ValueError("Forecast and realised arrays must have matching lengths.")
    eps = 1e-12
    f_safe = np.clip(f_arr, eps, None)
    r_safe = np.clip(r_arr, 0.0, None)
    return np.log(f_safe) + r_safe / f_safe


def block_bootstrap_ci_median(
    series: Iterable[float] | np.ndarray,
    *,
    block_len: int = 12,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Moving block bootstrap CI for the median of a time series.

    Parameters
    ----------
    series
        Ordered time series (e.g., ΔMSE per rolling window).
    block_len
        Block length (in weeks). Defaults to 12.
    n_boot
        Number of bootstrap replications. Defaults to 1000.
    alpha
        Tail probability for the two-sided CI (0.05 → 95% CI).
    rng
        Optional NumPy Generator for reproducibility.

    Returns
    -------
    (float, float)
        Lower and upper CI bounds.
    """

    arr = np.asarray(list(series), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng()

    L = max(1, int(block_len))
    B = max(1, int(n_boot))
    k = int(np.ceil(n / L))  # number of blocks per replicate

    # Circular moving block bootstrap: sample start indices, wrap as needed
    medians = np.empty(B, dtype=np.float64)
    for b in range(B):
        starts = rng.integers(0, n, size=k)
        pieces = []
        for s in starts:
            end = s + L
            if end <= n:
                pieces.append(arr[s:end])
            else:
                wrap = end - n
                pieces.append(np.concatenate([arr[s:n], arr[0:wrap]]))
        boot = np.concatenate(pieces)[:n]
        medians[b] = float(np.median(boot))

    lo = float(np.quantile(medians, alpha / 2.0))
    hi = float(np.quantile(medians, 1.0 - alpha / 2.0))
    return lo, hi


def _clip_prob(value: float, eps: float = 1e-8) -> float:
    return float(min(max(value, eps), 1.0 - eps))


def kupiec_pof_test(violations: np.ndarray, alpha: float = 0.05) -> float:
    """Return the Kupiec proportion-of-failures p-value."""

    indicators = np.asarray(violations, dtype=bool).ravel()
    n = int(indicators.size)
    if n == 0:
        return float("nan")
    hits = int(indicators.sum())
    pi_hat = (hits + 0.5) / (n + 1.0)
    pi_hat = _clip_prob(pi_hat)
    alpha = _clip_prob(float(alpha))
    log_l0 = hits * log(alpha) + (n - hits) * log(1.0 - alpha)
    log_l1 = hits * log(pi_hat) + (n - hits) * log(1.0 - pi_hat)
    lr = max(0.0, -2.0 * (log_l0 - log_l1))
    return float(erfc(sqrt(lr / 2.0)))


def christoffersen_independence_test(violations: np.ndarray) -> float:
    """Return the Christoffersen independence test p-value."""

    indicators = np.asarray(violations, dtype=bool).ravel()
    if indicators.size <= 1:
        return float("nan")
    prev = indicators[:-1]
    curr = indicators[1:]
    N00 = int(np.sum((~prev) & (~curr)))
    N01 = int(np.sum((~prev) & curr))
    N10 = int(np.sum(prev & (~curr)))
    N11 = int(np.sum(prev & curr))
    denom0 = N00 + N01
    denom1 = N10 + N11
    if denom0 == 0 or denom1 == 0:
        return float("nan")
    pi01 = _clip_prob(N01 / denom0)
    pi11 = _clip_prob(N11 / denom1)
    pi_hat = _clip_prob((N01 + N11) / (denom0 + denom1))
    log_l0 = (denom0 * log(1.0 - pi_hat)) + (denom1 * log(pi_hat))
    log_l1 = (
        N00 * log(1.0 - pi01)
        + N01 * log(pi01)
        + N10 * log(1.0 - pi11)
        + N11 * log(pi11)
    )
    lr = max(0.0, -2.0 * (log_l0 - log_l1))
    return float(erfc(sqrt(lr / 2.0)))


def expected_shortfall_test(
    losses: np.ndarray,
    es_forecasts: np.ndarray,
    violations: np.ndarray,
) -> float:
    """Approximate two-sided t-test comparing realised losses and ES forecasts."""

    losses = np.asarray(losses, dtype=np.float64).ravel()
    es_forecasts = np.asarray(es_forecasts, dtype=np.float64).ravel()
    mask = np.asarray(violations, dtype=bool).ravel()
    if losses.shape != es_forecasts.shape or losses.shape != mask.shape:
        raise ValueError("losses, es_forecasts, and violations must share shape.")
    hit_mask = mask & np.isfinite(losses) & np.isfinite(es_forecasts)
    m = int(hit_mask.sum())
    if m < 2:
        return float("nan")
    diff = losses[hit_mask] - es_forecasts[hit_mask]
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff <= 0.0:
        return float(1.0 if abs(mean_diff) < 1e-8 else 0.0)
    t_stat = mean_diff / (std_diff / sqrt(float(m)))
    return float(erfc(abs(t_stat) / sqrt(2.0)))


def alignment_diagnostics(
    covariance: np.ndarray,
    direction: np.ndarray,
    *,
    top_p: int = 3,
) -> tuple[float, float]:
    """Return (angle_deg, energy_mu) between detection direction and PCA subspace."""

    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    vec = np.asarray(direction, dtype=np.float64).reshape(-1)
    if cov.shape[0] != vec.size:
        raise ValueError("Direction vector dimension mismatch.")
    if top_p <= 0:
        raise ValueError("top_p must be positive.")

    # Ensure orthonormal eigenbasis
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[: min(int(top_p), cov.shape[0])]]
    vec_norm = np.linalg.norm(vec)
    if vec_norm <= 0.0:
        raise ValueError("Direction vector must have non-zero norm.")
    unit_vec = vec / vec_norm
    projection = basis.T @ unit_vec
    proj_norm = np.clip(np.linalg.norm(projection), -1.0, 1.0)
    angle_rad = np.arccos(proj_norm)
    energy_mu = float(unit_vec.T @ cov @ unit_vec)
    return float(np.degrees(angle_rad)), energy_mu


# --- Plotting ---------------------------------------------------------------


def plot_variance_error_panel(
    errors: Mapping[str, Iterable[float] | np.ndarray], base_path: Path
) -> None:
    """Plot E3: variance MSE mean and distribution by method.

    Parameters
    ----------
    errors
        Mapping from method label to sequence of squared errors across windows.
    base_path
        Output path base (without extension). PNG and PDF will be written.
    """

    import matplotlib.pyplot as plt  # local import to avoid test dependency

    filtered: dict[str, np.ndarray] = {}
    for key, values in errors.items():
        arr = np.asarray(list(values), dtype=np.float64).ravel()
        mask = np.isfinite(arr)
        if mask.any():
            filtered[key] = arr[mask]
    if not filtered:
        return

    methods = list(filtered.keys())
    means = [float(np.mean(filtered[m])) for m in methods]
    violin_data = [filtered[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = [f"C{i}" for i in range(len(methods))]
    axes[0].bar(methods, means, color=colors)
    axes[0].set_ylabel("Squared error")
    axes[0].set_title("Mean variance MSE")
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")

    parts = axes[1].violinplot(violin_data, showmeans=True, showmedians=False, widths=0.7)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx])
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    axes[1].set_xticks(np.arange(1, len(methods) + 1))
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("Squared error")
    axes[1].set_title("Distribution across windows")

    fig.suptitle("E3: Variance forecast errors", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_error(coverage_errors: Mapping[str, float], base_path: Path) -> None:
    """Plot E4: VaR(95%) coverage errors by method.

    Parameters
    ----------
    coverage_errors
        Mapping from method label to coverage error (p_empirical - 0.05).
    base_path
        Output path base (without extension). PNG and PDF will be written.
    """

    import matplotlib.pyplot as plt  # local import to avoid test dependency

    if not coverage_errors:
        return
    methods = list(coverage_errors.keys())
    values = [coverage_errors[m] for m in methods]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(methods, values, color=[f"C{i}" for i in range(len(methods))])
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Coverage error")
    ax.set_title("E4: 95% VaR coverage error")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --- Aggregation ------------------------------------------------------------


@dataclass
class DeltaSummary:
    median: float
    iqr: float
    ci_lo: float
    ci_hi: float
    p_value: float


def summarize_deltas(
    deltas: Iterable[float] | np.ndarray,
    *,
    block_len: int = 12,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> DeltaSummary:
    """Return robust summary statistics and CI for paired deltas.

    Parameters
    ----------
    deltas
        Array of paired differences across windows.
    block_len, n_boot, alpha, rng
        Block bootstrap parameters.
    """

    arr = np.asarray(list(deltas), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return DeltaSummary(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    med = float(np.median(arr))
    ci_lo, ci_hi = block_bootstrap_ci_median(arr, block_len=block_len, n_boot=n_boot, alpha=alpha, rng=rng)
    return DeltaSummary(median=med, iqr=iqr(arr), ci_lo=ci_lo, ci_hi=ci_hi, p_value=sign_test_pvalue(arr))


def build_metrics_summary(
    *,
    errors_by_combo: Mapping[str, Mapping[int, float]],
    coverage_errors: Mapping[str, float],
    qlike_by_combo: Mapping[str, Mapping[int, float]] | None = None,
    var_forecasts: Mapping[str, Iterable[float]] | None = None,
    realised_returns: Mapping[str, Iterable[float]] | None = None,
    es_forecasts: Mapping[str, Iterable[float]] | None = None,
    label: str,
    block_len: int = 12,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate window-level errors into a metrics summary DataFrame.

    The input follows the convention used by the equity runner:
    keys are "{strategy}::{estimator}" and values map window index -> squared error.
    For De-aliased rows, the function augments with paired ΔMSE vs LW and Aliased.
    Optional VaR/ES sequences supply backtest diagnostics when available.
    """

    rows: list[dict[str, Any]] = []
    # Canonical estimator names used in the codebase
    EST_DE = "De-aliased"
    EST_LW = "Ledoit-Wolf"
    EST_AL = "Aliased"

    # Build per-combo summaries
    for combo_key, error_map in errors_by_combo.items():
        if not error_map:
            continue
        errors_array = np.array([error_map[idx] for idx in sorted(error_map.keys())], dtype=np.float64)
        strat, est = combo_key.split("::", maxsplit=1)
        entry: dict[str, Any] = {
            "label": label,
            "strategy": strat,
            "estimator": est,
            "n_windows": int(len(error_map)),
            "mean_mse": float(np.mean(errors_array)),
            "median_mse": float(np.median(errors_array)),
            "iqr_mse": iqr(errors_array),
            "coverage_error": float(coverage_errors.get(combo_key, float("nan"))),
            "mean_qlike": float("nan"),
            "median_qlike": float("nan"),
            "sign_test_p_de_vs_lw": float("nan"),
            "sign_test_p_de_vs_alias": float("nan"),
            "delta_median_de_minus_lw": float("nan"),
            "delta_median_de_minus_alias": float("nan"),
            "ci_lo_de_minus_lw": float("nan"),
            "ci_hi_de_minus_lw": float("nan"),
            "ci_lo_de_minus_alias": float("nan"),
            "ci_hi_de_minus_alias": float("nan"),
            "dm_stat_de_vs_lw": float("nan"),
            "dm_p_de_vs_lw": float("nan"),
            "dm_stat_de_vs_oas": float("nan"),
            "dm_p_de_vs_oas": float("nan"),
            "dm_stat_de_vs_cc": float("nan"),
            "dm_p_de_vs_cc": float("nan"),
            "dm_stat_de_vs_factor": float("nan"),
            "dm_p_de_vs_factor": float("nan"),
            "dm_stat_de_vs_lw_qlike": float("nan"),
            "dm_p_de_vs_lw_qlike": float("nan"),
            "dm_stat_de_vs_oas_qlike": float("nan"),
            "dm_p_de_vs_oas_qlike": float("nan"),
            "var_kupiec_p": float("nan"),
            "var_independence_p": float("nan"),
            "es_shortfall_p": float("nan"),
        }

        qlike_map = qlike_by_combo.get(combo_key, {}) if qlike_by_combo is not None else {}
        if qlike_map:
            qlike_array = np.array([qlike_map[idx] for idx in sorted(qlike_map.keys())], dtype=np.float64)
            entry["mean_qlike"] = float(np.mean(qlike_array))
            entry["median_qlike"] = float(np.median(qlike_array))

        if var_forecasts is not None and realised_returns is not None:
            var_seq = var_forecasts.get(combo_key, [])
            realised_seq = realised_returns.get(combo_key, [])
            var_arr = np.array(list(var_seq), dtype=np.float64)
            realised_arr = np.array(list(realised_seq), dtype=np.float64)
            if var_arr.size == realised_arr.size and var_arr.size > 0:
                mask = np.isfinite(var_arr) & np.isfinite(realised_arr)
                if mask.any():
                    violations = realised_arr[mask] < var_arr[mask]
                    entry["var_kupiec_p"] = kupiec_pof_test(violations, alpha=0.05)
                    entry["var_independence_p"] = christoffersen_independence_test(violations)
                    if es_forecasts is not None and combo_key in es_forecasts:
                        es_seq = es_forecasts.get(combo_key, [])
                        es_arr = np.array(list(es_seq), dtype=np.float64)
                        if es_arr.size == var_arr.size:
                            entry["es_shortfall_p"] = expected_shortfall_test(
                                losses=realised_arr[mask],
                                es_forecasts=es_arr[mask],
                                violations=violations,
                            )

        # Only compute Δ summaries for De-aliased rows where both comparators are present
        if est == EST_DE:
            base_lw_key = f"{strat}::{EST_LW}"
            base_al_key = f"{strat}::{EST_AL}"
            dm_comparators = {
                EST_LW: "lw",
                "OAS": "oas",
                "Constant-Correlation": "cc",
                "Factor": "factor",
                "Factor-Observed": "factor_obs",
                "POET-lite": "poet",
                "Tyler-Shrink": "tyler",
            }
            lw_map = errors_by_combo.get(base_lw_key, {})
            al_map = errors_by_combo.get(base_al_key, {})
            # Common windows
            common_lw = sorted(set(error_map.keys()) & set(lw_map.keys()))
            common_al = sorted(set(error_map.keys()) & set(al_map.keys()))
            if common_lw:
                de = np.array([error_map[i] for i in common_lw], dtype=np.float64)
                lw = np.array([lw_map[i] for i in common_lw], dtype=np.float64)
                deltas = de - lw
                ds = summarize_deltas(deltas, block_len=block_len, n_boot=n_boot, alpha=alpha)
                entry["sign_test_p_de_vs_lw"] = ds.p_value
                entry["delta_median_de_minus_lw"] = ds.median
                entry["ci_lo_de_minus_lw"] = ds.ci_lo
                entry["ci_hi_de_minus_lw"] = ds.ci_hi
            if common_al:
                de = np.array([error_map[i] for i in common_al], dtype=np.float64)
                al = np.array([al_map[i] for i in common_al], dtype=np.float64)
                deltas = de - al
                ds = summarize_deltas(deltas, block_len=block_len, n_boot=n_boot, alpha=alpha)
                entry["sign_test_p_de_vs_alias"] = ds.p_value
                entry["delta_median_de_minus_alias"] = ds.median
                entry["ci_lo_de_minus_alias"] = ds.ci_lo
                entry["ci_hi_de_minus_alias"] = ds.ci_hi

            for est_name, suffix in dm_comparators.items():
                comp_key = f"{strat}::{est_name}"
                comp_map = errors_by_combo.get(comp_key, {})
                if not comp_map:
                    continue
                common = sorted(set(error_map.keys()) & set(comp_map.keys()))
                if not common:
                    continue
                de_vals = np.array([error_map[i] for i in common], dtype=np.float64)
                comp_vals = np.array([comp_map[i] for i in common], dtype=np.float64)
                dm_stat, dm_p = dm_test(de_vals, comp_vals)
                entry[f"dm_stat_de_vs_{suffix}"] = dm_stat
                entry[f"dm_p_de_vs_{suffix}"] = dm_p

            if qlike_by_combo is not None and qlike_map:
                de_qlike_map = qlike_by_combo.get(combo_key, {})
                lw_qlike_map = qlike_by_combo.get(f"{strat}::{EST_LW}", {})
                oas_qlike_map = qlike_by_combo.get(f"{strat}::OAS", {})
                if de_qlike_map and lw_qlike_map:
                    common = sorted(set(de_qlike_map.keys()) & set(lw_qlike_map.keys()))
                    if common:
                        de_vals = np.array([de_qlike_map[i] for i in common], dtype=np.float64)
                        lw_vals = np.array([lw_qlike_map[i] for i in common], dtype=np.float64)
                        dm_stat, dm_p = dm_test(de_vals, lw_vals)
                        entry["dm_stat_de_vs_lw_qlike"] = dm_stat
                        entry["dm_p_de_vs_lw_qlike"] = dm_p
                if de_qlike_map and oas_qlike_map:
                    common = sorted(set(de_qlike_map.keys()) & set(oas_qlike_map.keys()))
                    if common:
                        de_vals = np.array([de_qlike_map[i] for i in common], dtype=np.float64)
                        oas_vals = np.array([oas_qlike_map[i] for i in common], dtype=np.float64)
                        dm_stat, dm_p = dm_test(de_vals, oas_vals)
                        entry["dm_stat_de_vs_oas_qlike"] = dm_stat
                        entry["dm_p_de_vs_oas_qlike"] = dm_p

        rows.append(entry)

    return pd.DataFrame(rows)
