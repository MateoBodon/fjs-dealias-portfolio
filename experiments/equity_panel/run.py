# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
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
from finance.io import load_prices_csv, to_daily_returns
from finance.portfolios import equal_weight, min_variance_box, minimum_variance
from finance.returns import balance_weeks, weekly_panel
from fjs.balanced import mean_squares
from fjs.dealias import dealias_search
from fjs.mp import estimate_Cs_from_MS, mp_edge
from fjs.spectra import plot_spectrum_with_edges, plot_spike_timeseries
from meta.run_meta import write_run_meta
from evaluation import check_dealiased_applied

DEFAULT_CONFIG = {
    "data_path": "data/prices_sample.csv",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "window_weeks": 156,
    "horizon_weeks": 4,
    "output_dir": "experiments/equity_panel/outputs",
    "dealias_delta_frac": None,
    "signed_a": True,
    "cs_drop_top_frac": 0.1,
    "target_component": 0,
    "a_grid": 120,
    "dealias_eps": 0.05,
}


def load_config(path: Path | str) -> dict[str, Any]:
    """Load experiment configuration, falling back to defaults."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping.")
    merged = DEFAULT_CONFIG | data
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
    """Load prices (or synthesise) and return daily log returns."""

    data_path = Path(config["data_path"])
    if not data_path.exists():
        _generate_synthetic_prices(data_path)

    prices = load_prices_csv(str(data_path))
    return to_daily_returns(prices)


def _run_param_ablation(
    daily_returns: pd.DataFrame,
    output_dir: Path,
    *,
    target_component: int,
    base_delta: float,
    base_delta_frac: float | None,
    base_eps: float,
    base_eta: float,
    signed_a: bool,
) -> None:
    """Grid sweep over detection parameters; emit CSV and heatmaps (E5).

    This routine uses a shorter rolling setup for speed and reproducibility.
    """

    weekly_balanced, week_map, replicates, _ = _balanced_weekly_panel(daily_returns)
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

    delta_fracs = [0.02, 0.03, 0.05]
    eps_vals = [0.02, 0.03, 0.05]
    a_grids = [72, 120, 144]
    etas = [0.4, 1.0]

    records: list[dict[str, Any]] = []
    for df in delta_fracs:
        for eps in eps_vals:
            for ag in a_grids:
                for eta in etas:
                    det_count = 0
                    mse_alias_list: list[float] = []
                    mse_de_list: list[float] = []
                    for fit, hold in windows:
                        fit_blocks = [
                            week_map[idx] for idx in fit.index if idx in week_map
                        ]
                        hold_blocks = [
                            week_map[idx] for idx in hold.index if idx in week_map
                        ]
                        if len(fit_blocks) != len(fit.index) or len(hold_blocks) != len(
                            hold.index
                        ):
                            continue
                        y_fit_daily = np.vstack(fit_blocks)
                        y_hold_daily = np.vstack(hold_blocks)
                        groups_fit = np.repeat(np.arange(len(fit_blocks)), replicates)
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


def _iqr(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    return float(q75 - q25)


def _sign_test_pvalue(differences: list[float] | np.ndarray) -> float:
    arr = np.asarray(differences, dtype=np.float64)
    mask = np.abs(arr) > 1e-12
    arr = arr[mask]
    n = arr.size
    if n == 0:
        return float("nan")
    positives = int(np.count_nonzero(arr > 0))
    tail = min(positives, n - positives)
    cumulative = sum(comb(n, k) for k in range(0, tail + 1)) * (0.5**n)
    p_value = min(1.0, 2.0 * cumulative)
    return float(p_value)


def _plot_variance_error_panel(
    errors: dict[str, list[float] | np.ndarray], base_path: Path
) -> None:
    if not errors:
        return
    filtered: dict[str, np.ndarray] = {}
    for key, values in errors.items():
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 0:
            continue
        mask = np.isfinite(arr)
        if not mask.any():
            continue
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
    # Reduce label overlap for long method names
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")

    parts = axes[1].violinplot(
        violin_data, showmeans=True, showmedians=False, widths=0.7
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx])
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    axes[1].set_xticks(np.arange(1, len(methods) + 1))
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("Squared error")
    axes[1].set_title("Distribution across windows")

    fig.suptitle("E3: Variance forecast errors", fontsize=12)
    # Add a bit more bottom margin for rotated tick labels
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_error(coverage_errors: dict[str, float], base_path: Path) -> None:
    if not coverage_errors:
        return
    methods = list(coverage_errors.keys())
    values = [coverage_errors[m] for m in methods]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(methods, values, color=[f"C{i}" for i in range(len(methods))])
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Coverage error")
    ax.set_title("E4: 95% VaR coverage error")
    # Rotate long method labels, right-align, and increase bottom margin
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _balanced_weekly_panel(
    daily_returns: pd.DataFrame,
    *,
    replicates: int = 5,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, np.ndarray], int, list[str]]:
    """Build a balanced weekly panel with consistent tickers and daily slices."""

    if daily_returns.index.inferred_type != "datetime64":
        raise ValueError("daily_returns must use a DatetimeIndex.")

    panel = daily_returns.sort_index()
    grouped = panel.groupby(panel.index.to_period("W-MON"))

    week_frames: list[pd.DataFrame] = []
    week_labels: list[pd.Timestamp] = []

    for period, frame in grouped:
        cleaned = frame.dropna(axis=1, how="all")
        cleaned = cleaned.dropna(axis=0, how="any")
        cleaned = cleaned.sort_index()
        if cleaned.shape[0] < replicates:
            continue
        trimmed = cleaned.iloc[:replicates]
        if trimmed.isna().any().any():
            continue
        week_frames.append(trimmed)
        week_labels.append(period.start_time)

    if not week_frames:
        raise ValueError("No balanced weeks available for evaluation.")

    common_tickers = set(week_frames[0].columns)
    for frame in week_frames[1:]:
        common_tickers &= set(frame.columns)
    if not common_tickers:
        raise ValueError("No common tickers across balanced weeks.")
    ordered_tickers = sorted(common_tickers)

    week_arrays = [
        frame.loc[:, ordered_tickers].to_numpy(dtype=np.float64)
        for frame in week_frames
    ]
    replicate_count = week_arrays[0].shape[0]
    if any(arr.shape[0] != replicate_count for arr in week_arrays):
        raise ValueError("Replicate count varies across balanced weeks.")

    week_map = {week_labels[idx]: week_arrays[idx] for idx in range(len(week_labels))}
    weekly_data = np.stack([arr.sum(axis=0) for arr in week_arrays], axis=0)
    weekly_df = pd.DataFrame(
        weekly_data,
        index=pd.Index(week_labels, name="week_start"),
        columns=ordered_tickers,
    )
    return weekly_df, week_map, replicate_count, ordered_tickers


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
    cs_drop_top_frac: float,
    sigma_ablation: bool,
    label: str,
    progress: bool = True,
    a_grid: int = 120,
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

    _, dropped_weeks = weekly_panel(daily_subset, start_ts, end_ts)
    weekly_balanced, week_map, replicates, tickers = _balanced_weekly_panel(
        daily_subset
    )
    if weekly_balanced.shape[0] < window_weeks + horizon_weeks:
        raise ValueError(
            "Not enough balanced weeks for the requested rolling evaluation window."
        )

    p_assets = len(tickers)
    equal_weights = equal_weight(p_assets)

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

    def _equal_weight_weights(_: np.ndarray) -> np.ndarray:
        return equal_weights.copy()

    def _min_var_weights(covariance: np.ndarray) -> np.ndarray:
        result = min_variance_box(covariance, lb=-0.02, ub=0.02)
        return np.asarray(result.weights, dtype=np.float64)

    def _min_var_longonly_weights(covariance: np.ndarray) -> np.ndarray:
        try:
            result = minimum_variance(covariance, allow_short=False)
            return np.asarray(result.weights, dtype=np.float64)
        except ImportError:
            raise

    strategies: dict[str, dict[str, Any]] = {
        "Equal Weight": {
            "prefix": "eq",
            "get_weights": _equal_weight_weights,
            "available": True,
        },
        "Min-Variance (box)": {
            "prefix": "mv",
            "get_weights": _min_var_weights,
            "available": True,
        },
        "Min-Variance (long-only)": {
            "prefix": "mvlo",
            "get_weights": _min_var_longonly_weights,
            "available": True,
        },
    }

    errors_by_combo: dict[str, dict[int, float]] = defaultdict(dict)
    var95_by_combo: dict[str, list[float]] = defaultdict(list)
    realised_returns_by_combo: dict[str, list[float]] = defaultdict(list)
    strategy_success: dict[str, bool] = {name: False for name in strategies}
    records: list[dict[str, Any]] = []

    baseline_name = "Equal Weight"
    baseline_alias_key = f"{baseline_name}::Aliased"
    var_forecasts_alias_baseline: list[float] = []
    var_forecasts_de_baseline: list[float] = []
    var_forecasts_lw_baseline: list[float] = []

    total_windows = weekly_balanced.shape[0] - (window_weeks + horizon_weeks) + 1
    window_iter = rolling_windows(weekly_balanced, window_weeks, horizon_weeks)
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

        fit_blocks = [week_map[idx] for idx in fit.index if idx in week_map]
        hold_blocks = [week_map[idx] for idx in hold.index if idx in week_map]
        if len(fit_blocks) != len(fit.index) or len(hold_blocks) != len(hold.index):
            continue

        y_fit_daily = np.vstack(fit_blocks)
        y_hold_daily = np.vstack(hold_blocks)
        groups_fit = np.repeat(np.arange(len(fit_blocks)), replicates)

        detections = dealias_search(
            y_fit_daily,
            groups_fit,
            target_r=target_component,
            delta=delta,
            delta_frac=delta_frac,
            eps=eps,
            stability_eta_deg=stability_eta,
            use_tvector=True,
            nonnegative_a=not signed_a,
            a_grid=int(a_grid),
            cs_drop_top_frac=float(cs_drop_top_frac),
        )

        fit_matrix = fit.to_numpy(dtype=np.float64)
        hold_matrix = hold.to_numpy(dtype=np.float64)
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
        }

        # Log top detection (by lambda_hat) for diagnostics/time series
        if detections:
            det_sorted = sorted(
                detections, key=lambda d: float(d["lambda_hat"]), reverse=True
            )
            top = det_sorted[0]
            window_record["top_lambda_hat"] = float(top["lambda_hat"])
            window_record["top_mu_hat"] = float(top["mu_hat"])
            window_record["top_a0"] = float(top["a"][0])
            window_record["top_a1"] = float(top["a"][1])
            window_record["top_stability_margin"] = float(top["stability_margin"])
            # Optional diagnostics populated by dealias_search
            window_record["top_z_plus"] = (
                float(top.get("z_plus", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
            window_record["top_threshold_main"] = (
                float(top.get("threshold_main", np.nan))
                if isinstance(top, dict)
                else float("nan")
            )
        else:
            window_record["top_lambda_hat"] = float("nan")
            window_record["top_mu_hat"] = float("nan")
            window_record["top_a0"] = float("nan")
            window_record["top_a1"] = float("nan")
            window_record["top_stability_margin"] = float("nan")
            window_record["top_z_plus"] = float("nan")
            window_record["top_threshold_main"] = float("nan")

        # Always record the top aliased Σ1 eigenvalue (for E2-alt)
        try:
            stats_local = mean_squares(y_fit_daily, groups_fit)
            sigma1_local = stats_local["Sigma1_hat"].astype(np.float64)
            window_record["top_sigma1_eigval"] = float(
                np.linalg.eigvalsh(sigma1_local)[-1]
            )
        except Exception:
            window_record["top_sigma1_eigval"] = float("nan")

        for strategy_label, cfg in strategies.items():
            if not cfg.get("available", True):
                continue
            try:
                weights = np.asarray(
                    cfg["get_weights"](cov_fit), dtype=np.float64
                ).reshape(-1)
            except ImportError:
                cfg["available"] = False
                continue
            except Exception:
                continue

            if weights.size != p_assets or not np.all(np.isfinite(weights)):
                continue
            weight_sum = float(weights.sum())
            if not np.isfinite(weight_sum) or abs(weight_sum) < 1e-12:
                continue
            if not np.isclose(weight_sum, 1.0):
                weights = weights / weight_sum

            strategy_success[strategy_label] = True
            prefix = cfg["prefix"]

            forecast_alias, realised_var_alias = variance_forecast_from_components(
                y_fit_daily,
                y_hold_daily,
                replicates,
                weights,
            )
            forecast_dealias, realised_var_de = variance_forecast_from_components(
                y_fit_daily,
                y_hold_daily,
                replicates,
                weights,
                detections=detections,
            )
            realised_var = (
                realised_var_de if np.isfinite(realised_var_de) else realised_var_alias
            )
            forecast_lw, realised_var_lw = oos_variance_forecast(
                fit_matrix,
                hold_matrix,
                weights,
                estimator="lw",
            )

            hold_returns = hold_matrix @ weights

            alias_error = float((forecast_alias - realised_var) ** 2)
            dealias_error = float((forecast_dealias - realised_var) ** 2)
            lw_error = float((forecast_lw - realised_var_lw) ** 2)

            combos = [
                ("Aliased", forecast_alias, realised_var, alias_error),
                ("De-aliased", forecast_dealias, realised_var, dealias_error),
                ("Ledoit-Wolf", forecast_lw, realised_var_lw, lw_error),
                (
                    "SCM",
                    oos_variance_forecast(
                        fit_matrix,
                        hold_matrix,
                        weights,
                        estimator="scm",
                    )[0],
                    oos_variance_forecast(
                        fit_matrix,
                        hold_matrix,
                        weights,
                        estimator="scm",
                    )[1],
                    float(
                        (
                            oos_variance_forecast(
                                fit_matrix,
                                hold_matrix,
                                weights,
                                estimator="scm",
                            )[0]
                            - oos_variance_forecast(
                                fit_matrix,
                                hold_matrix,
                                weights,
                                estimator="scm",
                            )[1]
                        )
                        ** 2
                    ),
                ),
            ]

            var95_values = {
                "Aliased": -1.65 * np.sqrt(max(forecast_alias, 0.0)),
                "De-aliased": -1.65 * np.sqrt(max(forecast_dealias, 0.0)),
                "Ledoit-Wolf": -1.65 * np.sqrt(max(forecast_lw, 0.0)),
                "SCM": -1.65
                * np.sqrt(
                    max(
                        oos_variance_forecast(
                            fit_matrix,
                            hold_matrix,
                            weights,
                            estimator="scm",
                        )[0],
                        0.0,
                    )
                ),
            }

            for estimator_name, forecast_value, realised_value, error_value in combos:
                combo_key = f"{strategy_label}::{estimator_name}"
                errors_by_combo[combo_key][window_idx] = error_value
                var95_by_combo[combo_key].extend(
                    [var95_values[estimator_name]] * hold_returns.size
                )
                realised_returns_by_combo[combo_key].extend(hold_returns.tolist())

                suffix = estimator_name.lower().replace("-", "").replace(" ", "_")
                window_record[f"{prefix}_{suffix}_forecast"] = float(forecast_value)
                window_record[f"{prefix}_{suffix}_realized"] = float(realised_value)

            if strategy_label == baseline_name:
                var_forecasts_alias_baseline.append(float(forecast_alias))
                var_forecasts_de_baseline.append(float(forecast_dealias))
                var_forecasts_lw_baseline.append(float(forecast_lw))

        if len(window_record) > 5:
            records.append(window_record)

    if not records:
        raise ValueError("No rolling windows were evaluated after balancing.")

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
        _plot_variance_error_panel(errors_for_plot, output_dir / "E3_variance_mse")
    if coverage_errors:
        _plot_coverage_error(coverage_errors, output_dir / "E4_var95_coverage_error")

    baseline_errors_map = errors_by_combo.get(baseline_alias_key, {})
    baseline_keys = set(baseline_errors_map.keys())

    metrics_rows: list[dict[str, Any]] = []
    for combo_key, error_map in errors_by_combo.items():
        if not error_map:
            continue
        errors_array = np.array(
            [error_map[idx] for idx in sorted(error_map.keys())], dtype=np.float64
        )
        strategy_label, estimator_label = combo_key.split("::", maxsplit=1)
        metrics_entry: dict[str, Any] = {
            "label": label,
            "strategy": strategy_label,
            "estimator": estimator_label,
            "n_windows": int(len(error_map)),
            "mean_mse": float(np.mean(errors_array)),
            "median_mse": float(np.median(errors_array)),
            "iqr_mse": _iqr(errors_array),
            "coverage_error": coverage_errors.get(combo_key, float("nan")),
        }
        if combo_key == baseline_alias_key or not baseline_errors_map:
            metrics_entry["sign_test_p"] = float("nan")
        else:
            common_keys = sorted(baseline_keys & set(error_map.keys()))
            if not common_keys:
                metrics_entry["sign_test_p"] = float("nan")
            else:
                diffs = [
                    baseline_errors_map[idx] - error_map[idx] for idx in common_keys
                ]
                metrics_entry["sign_test_p"] = _sign_test_pvalue(diffs)
        metrics_rows.append(metrics_entry)

    metrics_summary = pd.DataFrame(metrics_rows)
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_dir / "rolling_results.csv", index=False)

    # Persist detection summary and spike timeseries (E2)
    if not results_df.empty and "top_lambda_hat" in results_df.columns:
        det_summary = results_df[
            [
                "fit_start",
                "fit_end",
                "hold_start",
                "hold_end",
                "n_detections",
                "top_lambda_hat",
                "top_mu_hat",
                "top_a0",
                "top_a1",
                "top_stability_margin",
                "top_sigma1_eigval",
                "top_z_plus",
                "top_threshold_main",
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

    summary_payload: dict[str, Any] = {
        "label": label,
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "balanced_weeks": int(weekly_balanced.shape[0]),
        "dropped_weeks": int(dropped_weeks),
        "window_weeks": int(window_weeks),
        "horizon_weeks": int(horizon_weeks),
        "rolling_windows_evaluated": len(records),
        "replicates_per_week": int(replicates),
        "n_assets": int(weekly_balanced.shape[1]),
        "strategies": {name: bool(strategy_success[name]) for name in strategies},
    }
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
            delta_frac,
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
    cs_drop_top_frac_override: float | None = None,
    progress_override: bool | None = None,
    eps_override: float | None = None,
    a_grid_override: int | None = None,
    ablations: bool | None = None,
    eta_override: float | None = None,
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
    if cs_drop_top_frac_override is not None:
        config["cs_drop_top_frac"] = float(cs_drop_top_frac_override)
    if eps_override is not None:
        config["dealias_eps"] = float(eps_override)
    if a_grid_override is not None:
        config["a_grid"] = int(a_grid_override)
    if eta_override is not None:
        config["stability_eta_deg"] = float(eta_override)
    # Values from YAML remain if overrides not provided
    daily_returns = _prepare_data(config)

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
            "output_dir": output_dir,
            "sigma_ablation": sigma_ablation,
        }
    ]

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
                "output_dir": output_dir / crisis_label,
                "sigma_ablation": False,
            }
        )

    for run_cfg in runs:
        run_output_dir = Path(run_cfg["output_dir"])
        run_output_dir.mkdir(parents=True, exist_ok=True)
        _run_single_period(
            daily_returns,
            start=run_cfg["start"],
            end=run_cfg["end"],
            output_dir=run_output_dir,
            window_weeks=int(config["window_weeks"]),
            horizon_weeks=int(config["horizon_weeks"]),
            delta=float(config.get("dealias_delta", 0.3)),
            delta_frac=cast(float | None, config.get("dealias_delta_frac")),
            eps=float(config.get("dealias_eps", 0.05)),
            stability_eta=float(config.get("stability_eta_deg", 1.0)),
            signed_a=bool(config.get("signed_a", True)),
            target_component=int(config.get("target_component", 0)),
            cs_drop_top_frac=float(config.get("cs_drop_top_frac", 0.1)),
            sigma_ablation=bool(run_cfg["sigma_ablation"]),
            label=str(run_cfg["label"]),
            progress=(True if progress_override is None else bool(progress_override)),
            a_grid=int(config.get("a_grid", 120)),
        )

        # Persist meta information for this run
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
            )
        except Exception:
            # Best effort; do not fail the entire run
            pass

    # Parameter ablations (E5)
    if bool(ablations):
        _run_param_ablation(
            daily_returns,
            output_dir,
            target_component=int(config.get("target_component", 0)),
            base_delta=float(config.get("dealias_delta", 0.0)),
            base_delta_frac=cast(float | None, config.get("dealias_delta_frac")),
            base_eps=float(config.get("dealias_eps", 0.03)),
            base_eta=float(config.get("stability_eta_deg", 0.4)),
            signed_a=bool(config.get("signed_a", True)),
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
        "--sigma-ablation",
        action="store_true",
        help="Run ±10% Cs perturbation ablation and persist diagnostics.",
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
        "--eta",
        type=float,
        default=None,
        help="Stability perturbation in degrees",
    )
    parser.add_argument(
        "--signed-a",
        action="store_true",
        help="Enable signed a-grid search (default true)",
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
    args = parser.parse_args()

    run_experiment(
        args.config,
        sigma_ablation=args.sigma_ablation,
        crisis=args.crisis,
        delta_frac_override=args.delta_frac,
        signed_a_override=args.signed_a,
        target_component_override=args.target_component,
        cs_drop_top_frac_override=args.cs_drop_top_frac,
        progress_override=(not args.no_progress),
        eps_override=args.eps,
        a_grid_override=args.a_grid,
        ablations=args.ablations,
        eta_override=args.eta,
    )


if __name__ == "__main__":
    main()
