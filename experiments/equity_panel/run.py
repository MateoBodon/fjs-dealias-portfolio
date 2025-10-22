# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from finance.eval import (
    oos_variance_forecast,
    risk_metrics,
    rolling_windows,
    variance_forecast_from_components,
)
from finance.io import load_prices_csv, to_daily_returns
from finance.portfolios import equal_weight, min_variance_box
from finance.returns import balance_weeks, weekly_panel
from fjs.balanced import mean_squares
from fjs.dealias import dealias_search
from fjs.mp import estimate_Cs_from_MS, mp_edge
from fjs.spectra import plot_spectrum_with_edges, plot_spike_timeseries

DEFAULT_CONFIG = {
    "data_path": "data/prices_sample.csv",
    "start_date": "2015-01-01",
    "end_date": "2024-12-31",
    "window_weeks": 156,
    "horizon_weeks": 4,
    "output_dir": "experiments/equity_panel/outputs",
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


def _plot_variance_error_panel(errors: dict[str, list[float]], base_path: Path) -> None:
    if not errors:
        return
    filtered = {k: np.asarray(v, dtype=np.float64) for k, v in errors.items() if v}
    if not filtered:
        return

    methods = list(filtered.keys())
    means = [float(np.mean(filtered[m])) for m in methods]
    violin_data = [filtered[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = [f"C{i}" for i in range(len(methods))]

    axes[0].bar(methods, means, color=colors)
    axes[0].set_ylabel("Squared error")
    axes[0].set_title("Mean variance MSE")

    parts = axes[1].violinplot(
        violin_data, showmeans=True, showmedians=False, widths=0.7
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx])
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    axes[1].set_xticks(np.arange(1, len(methods) + 1))
    axes[1].set_xticklabels(methods, rotation=15)
    axes[1].set_ylabel("Squared error")
    axes[1].set_title("Distribution across windows")

    fig.suptitle("E3: Variance forecast errors", fontsize=12)
    fig.tight_layout()
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_error(coverage_errors: dict[str, float], base_path: Path) -> None:
    if not coverage_errors:
        return
    methods = list(coverage_errors.keys())
    values = [coverage_errors[m] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, values, color=[f"C{i}" for i in range(len(methods))])
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Coverage error")
    ax.set_title("E4: 95% VaR coverage error")
    fig.tight_layout()
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
    eps: float,
    stability_eta: float,
    sigma_ablation: bool,
    label: str,
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
    )
    plot_spectrum_with_edges(
        eigenvalues,
        edges=edges,
        out_path=output_dir / "spectrum.pdf",
        title=plot_title,
    )

    def _equal_weight_weights(_: np.ndarray) -> np.ndarray:
        return equal_weights.copy()

    def _min_var_weights(covariance: np.ndarray) -> np.ndarray:
        result = min_variance_box(covariance, lb=-0.02, ub=0.02)
        return np.asarray(result.weights, dtype=np.float64)

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

    for window_idx, (fit, hold) in enumerate(
        rolling_windows(weekly_balanced, window_weeks, horizon_weeks)
    ):
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
            target_r=0,
            delta=delta,
            eps=eps,
            stability_eta_deg=stability_eta,
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
            ]

            var95_values = {
                "Aliased": -1.65 * np.sqrt(max(forecast_alias, 0.0)),
                "De-aliased": -1.65 * np.sqrt(max(forecast_dealias, 0.0)),
                "Ledoit-Wolf": -1.65 * np.sqrt(max(forecast_lw, 0.0)),
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
        _run_sigma_ablation(daily_subset, output_dir)


def _run_sigma_ablation(daily_returns: pd.DataFrame, output_dir: Path) -> None:
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
    drop_top = min(5, max(1, p_dim // 20))
    d_vec = np.array(
        [float(stats["I"] - 1), float(stats["n"] - stats["I"])],
        dtype=np.float64,
    )
    c_vec = np.array([float(stats["J"]), 1.0], dtype=np.float64)
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
                target_r=0,
                Cs=cs_scaled,
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
) -> None:
    """Execute the rolling equity forecasting experiment."""

    path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).with_name("config.yaml")
    )
    config = load_config(path)
    daily_returns = _prepare_data(config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

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
            eps=float(config.get("dealias_eps", 0.05)),
            stability_eta=float(config.get("stability_eta_deg", 1.0)),
            sigma_ablation=bool(run_cfg["sigma_ablation"]),
            label=str(run_cfg["label"]),
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
    args = parser.parse_args()

    run_experiment(
        args.config,
        sigma_ablation=args.sigma_ablation,
        crisis=args.crisis,
    )


if __name__ == "__main__":
    main()
