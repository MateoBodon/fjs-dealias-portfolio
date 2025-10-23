from __future__ import annotations

from inspect import isclass, isfunction
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.equity_panel import run as equity_run
from experiments.synthetic_oneway import run as synthetic_run

from finance import (
    OptimizationResult,
    build_design_matrix,
    compute_log_returns,
    equal_weight,
    evaluate_portfolio,
    ledoit_wolf_shrinkage,
    load_market_data,
    oos_variance_forecast,
    optimize_portfolio,
    risk_metrics,
    rolling_windows,
    to_daily_returns,
    variance_forecast_from_components,
    weekly_cov_from_components,
    weekly_panel,
)
from fjs import (
    BalancedConfig,
    DealiasingResult,
    MarchenkoPasturModel,
    compute_balanced_weights,
    dealias_covariance,
    estimate_spectrum,
    marchenko_pastur_edges,
    marchenko_pastur_pdf,
    mean_squares,
)
from fjs.dealias import dealias_search


def test_core_types_are_accessible() -> None:
    assert isclass(BalancedConfig)
    assert isclass(DealiasingResult)
    assert isclass(MarchenkoPasturModel)
    assert isclass(OptimizationResult)


def test_core_functions_are_callable() -> None:
    for candidate in [
        compute_balanced_weights,
        dealias_covariance,
        estimate_spectrum,
        marchenko_pastur_edges,
        marchenko_pastur_pdf,
        load_market_data,
        compute_log_returns,
        build_design_matrix,
        ledoit_wolf_shrinkage,
        optimize_portfolio,
        evaluate_portfolio,
        equal_weight,
        oos_variance_forecast,
        risk_metrics,
        rolling_windows,
        weekly_panel,
    ]:
        assert isfunction(candidate)


def test_experiment_entry_points_are_callable() -> None:
    assert isfunction(synthetic_run.run_experiment)
    assert isfunction(equity_run.run_experiment)


def test_experiment_configs_load() -> None:
    synth_config = synthetic_run.load_config(
        Path("experiments/synthetic_oneway/config.yaml")
    )
    equity_config = equity_run.load_config(Path("experiments/equity_panel/config.yaml"))
    assert isinstance(synth_config, dict)
    assert isinstance(equity_config, dict)


def test_single_equity_window_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-06", periods=15, freq="B")
    tickers = ["A", "B", "C", "D"]

    records = []
    for ticker in tickers:
        returns = rng.normal(scale=0.01, size=len(dates))
        prices = 100.0 * np.exp(np.cumsum(returns))
        records.append(
            pd.DataFrame({"date": dates, "ticker": ticker, "price_close": prices})
        )

    price_frame = pd.concat(records, ignore_index=True)
    csv_path = tmp_path / "prices.csv"
    price_frame.to_csv(csv_path, index=False)

    prices = load_market_data(csv_path)
    daily_returns = to_daily_returns(prices)
    weekly_returns, dropped_weeks = weekly_panel(
        daily_returns, start=dates[0], end=dates[-1]
    )
    assert dropped_weeks == 0
    windows = list(rolling_windows(weekly_returns, window_weeks=2, horizon_weeks=1))
    assert windows

    fit, hold = windows[0]
    weights = equal_weight(weekly_returns.shape[1])

    forecast_de, realized_de = oos_variance_forecast(
        fit.to_numpy(dtype=np.float64),
        hold.to_numpy(dtype=np.float64),
        weights,
        estimator="dealias",
    )
    forecast_lw, realized_lw = oos_variance_forecast(
        fit.to_numpy(dtype=np.float64),
        hold.to_numpy(dtype=np.float64),
        weights,
        estimator="lw",
    )

    metrics = risk_metrics([forecast_lw], [realized_lw])
    assert np.isfinite(forecast_de)
    assert np.isfinite(forecast_lw)
    assert np.isfinite(metrics["mse"])


def test_weekly_components_identity_and_detection_gain() -> None:
    rng = np.random.default_rng(7)
    p = 3
    days_per_week = 5
    total_weeks = 48
    fit_weeks = 28
    hold_weeks = total_weeks - fit_weeks

    def random_psd(scale: float) -> np.ndarray:
        mat = rng.normal(scale=scale, size=(p, p))
        return mat @ mat.T + 0.1 * np.eye(p)

    sigma1_true = random_psd(0.25)
    sigma2_true = random_psd(0.08)

    group_effects = rng.multivariate_normal(np.zeros(p), sigma1_true, size=total_weeks)
    residuals = rng.multivariate_normal(
        np.zeros(p), sigma2_true, size=(total_weeks, days_per_week)
    )
    observations = group_effects[:, None, :] + residuals
    panel = observations.reshape(total_weeks * days_per_week, p)
    groups = np.repeat(np.arange(total_weeks), days_per_week)

    stats = mean_squares(panel, groups)
    weekly_cov_est = weekly_cov_from_components(
        stats["MS1"], stats["MS2"], days_per_week
    )
    weekly_sums = panel.reshape(total_weeks, days_per_week, p).sum(axis=1)
    weekly_cov_emp = np.cov(weekly_sums, rowvar=False, ddof=1)

    np.testing.assert_allclose(
        weekly_cov_est,
        weekly_cov_emp,
        rtol=0.12,
        atol=0.05,
    )

    fit_panel = panel[: fit_weeks * days_per_week]
    hold_panel = panel[
        fit_weeks * days_per_week : (fit_weeks + hold_weeks) * days_per_week
    ]
    weights = np.full(p, 1.0 / p, dtype=np.float64)

    forecast_alias, realized_var = variance_forecast_from_components(
        fit_panel, hold_panel, days_per_week, weights
    )

    eigvals_true, eigvecs_true = np.linalg.eigh(sigma1_true)
    order = np.argsort(eigvals_true)[::-1]
    mu_vals = eigvals_true[order][:2]
    vecs = eigvecs_true[:, order][:, :2]
    detections = [
        {
            "mu_hat": float(mu_vals[idx]),
            "eigvec": vecs[:, idx],
        }
        for idx in range(mu_vals.size)
    ]

    forecast_detected, realized_var_check = variance_forecast_from_components(
        fit_panel, hold_panel, days_per_week, weights, detections=detections
    )
    np.testing.assert_allclose(realized_var, realized_var_check, rtol=0.0, atol=1e-12)

    mse_alias = (forecast_alias - realized_var) ** 2
    mse_detected = (forecast_detected - realized_var) ** 2
    assert mse_detected <= mse_alias * 0.9


def test_relative_delta_and_signed_a_integration() -> None:
    # Small synthetic weekly panel; smoke that signed-a and delta_frac wire through
    rng = np.random.default_rng(17)
    p = 5
    days_per_week = 3
    total_weeks = 24
    group_effects = rng.normal(scale=0.4, size=(total_weeks, p))
    residuals = rng.normal(scale=0.2, size=(total_weeks, days_per_week, p))
    panel = (group_effects[:, None, :] + residuals).reshape(
        total_weeks * days_per_week, p
    )
    groups = np.repeat(np.arange(total_weeks), days_per_week)

    stats = mean_squares(panel, groups)
    assert stats["MS1"].shape == (p, p)

    detections = dealias_search(
        panel,
        groups,
        target_r=0,
        a_grid=36,
        delta=0.0,
        delta_frac=0.05,
        nonnegative_a=False,
    )
    # It is OK if no detection occurs; this is an integration smoke test
    assert isinstance(detections, list)


def test_rolling_synthetic_oos_gain() -> None:
    rng = np.random.default_rng(1234)
    p = 8
    replicates = 3
    total_weeks = 36
    fit_weeks = 20
    hold_weeks = 4

    # Single spiked between-group component with strong signal
    v = rng.standard_normal(p)
    v /= np.linalg.norm(v)
    mu = 6.0
    sigma_within = 0.25

    group_scores = rng.normal(scale=np.sqrt(mu), size=total_weeks)
    between = np.outer(group_scores, v)
    residuals = rng.normal(scale=sigma_within, size=(total_weeks, replicates, p))
    observations = between[:, None, :] + residuals

    y_matrix = observations.reshape(total_weeks * replicates, p)
    w = np.full(p, 1.0 / p, dtype=np.float64)

    errs_alias: list[float] = []
    errs_de: list[float] = []
    for start in range(0, total_weeks - fit_weeks - hold_weeks + 1):
        fit_range = slice(start * replicates, (start + fit_weeks) * replicates)
        hold_range = slice(
            (start + fit_weeks) * replicates,
            (start + fit_weeks + hold_weeks) * replicates,
        )
        y_fit = y_matrix[fit_range]
        y_hold = y_matrix[hold_range]
        groups_fit = np.repeat(np.arange(fit_weeks), replicates)

        detections = dealias_search(
            y_fit,
            groups_fit,
            target_r=0,
            a_grid=72,
            delta=0.3,
            eps=0.05,
            stability_eta_deg=0.4,
        )
        f_alias, r_alias = variance_forecast_from_components(
            y_fit, y_hold, replicates, w
        )
        f_de, r_de = variance_forecast_from_components(
            y_fit, y_hold, replicates, w, detections=detections
        )
        realized = r_de if np.isfinite(r_de) else r_alias
        errs_alias.append(float((f_alias - realized) ** 2))
        errs_de.append(float((f_de - realized) ** 2))

    mse_alias = float(np.mean(errs_alias))
    mse_de = float(np.mean(errs_de))
    # Non-inferiority: de-aliased should not be worse than aliased
    assert mse_de <= mse_alias
