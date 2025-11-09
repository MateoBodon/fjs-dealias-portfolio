from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.eval.config import resolve_eval_config
from experiments.eval.diagnostics import DiagnosticReason
from experiments.eval.run import (
    DailyLoaderConfig,
    EvalConfig,
    load_daily_panel,
    run_evaluation,
    _aligned_dm_stat,
    _min_variance_weights,
    _vol_thresholds,
)


def _make_returns_csv(tmp_path: pytest.TempPathFactory) -> str:
    dates = pd.date_range("2024-01-02", periods=80, freq="B")
    rng = np.random.default_rng(2025)
    returns = rng.normal(scale=0.01, size=(len(dates), 6))
    frame = pd.DataFrame(returns, index=dates, columns=[f"A{i}" for i in range(6)])
    path = tmp_path.mktemp("data") / "returns.csv"
    frame.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)
    return str(path)


@pytest.mark.slow
def test_run_evaluation_emits_artifacts(tmp_path_factory: pytest.TempPathFactory) -> None:
    returns_csv = _make_returns_csv(tmp_path_factory)
    out_dir = tmp_path_factory.mktemp("outputs")
    config = EvalConfig(
            returns_csv=Path(returns_csv),
            factors_csv=None,
            window=20,
            horizon=5,
            out_dir=Path(out_dir),
            shrinker="rie",
            seed=123,
            use_factor_prewhiten=False,
            mv_box_hi=1.0,
        )
    outputs = run_evaluation(config)

    resolved_path = Path(out_dir) / "resolved_config.json"
    assert resolved_path.exists()
    resolved_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved_payload["returns_csv"].endswith("returns.csv")
    assert resolved_payload["prewhiten_mode_effective"]

    full_metrics = outputs.metrics["full"]
    assert full_metrics.exists()
    metrics_df = pd.read_csv(full_metrics)
    assert {
        "estimator",
        "portfolio",
        "delta_mse_vs_baseline",
        "delta_mse_ci_lower",
        "delta_mse_ci_upper",
    }.issubset(metrics_df.columns)
    estimators = set(metrics_df["estimator"].unique())
    expected_estimators = {
        "overlay",
        "baseline",
        "sample",
        "scm",
        "rie",
        "lw",
        "oas",
        "cc",
        "quest",
        "ewma",
        "factor",
        "poet",
    }
    assert expected_estimators <= estimators

    full_risk = outputs.risk["full"]
    assert full_risk.exists()
    risk_df = pd.read_csv(full_risk)
    assert {"vaR95", "es95"}.issubset(risk_df.columns)

    full_dm = outputs.dm["full"]
    assert full_dm.exists()
    dm_df = pd.read_csv(full_dm)
    assert {"dm_stat", "p_value", "n_effective"}.issubset(dm_df.columns)

    full_diag = outputs.diagnostics["full"]
    assert full_diag.exists()
    diag_df = pd.read_csv(full_diag)
    assert {
        "detection_rate",
        "reason_code",
        "resolved_config_path",
        "calm_threshold",
        "crisis_threshold",
        "vol_signal",
        "group_design",
        "group_count",
        "group_replicates",
        "prewhiten_r2_mean",
        "prewhiten_r2_median",
        "prewhiten_mode_requested",
        "prewhiten_mode_effective",
        "prewhiten_factor_count",
        "prewhiten_beta_abs_mean",
        "prewhiten_beta_abs_std",
        "prewhiten_beta_abs_median",
        "prewhiten_factors",
        "residual_energy_mean",
        "acceptance_delta",
        "group_label_counts",
        "group_observations",
        "vol_state_label",
        "factor_present",
        "changed_flag",
    }.issubset(diag_df.columns)
    assert "reps_by_label" in diag_df.columns
    if not diag_df.empty:
        reason_values = set(diag_df["reason_code"].dropna().unique())
        allowed = {reason.value for reason in DiagnosticReason} | {""}
        assert reason_values <= allowed
        assert diag_df["resolved_config_path"].str.endswith("resolved_config.json").all()
        assert diag_df["calm_threshold"].notna().any()
        assert diag_df["crisis_threshold"].notna().any()
        assert diag_df["reps_by_label"].astype(str).str.len().gt(0).any()
        assert diag_df["vol_signal"].notna().any()
        assert diag_df["group_label_counts"].notna().all()
        assert diag_df["vol_state_label"].notna().all()
        assert diag_df["prewhiten_mode_effective"].notna().all()
        assert diag_df["prewhiten_beta_abs_mean"].notna().all()
        assert diag_df["residual_energy_mean"].notna().all()

    detail_diag = outputs.diagnostics_detail["full"]
    assert detail_diag.exists()
    assert outputs.diagnostics_detail["all"].exists()
    detail_df = pd.read_csv(detail_diag)
    assert "reps_by_label" in detail_df.columns
    if not detail_df.empty:
        assert detail_df["reps_by_label"].astype(str).str.len().gt(0).any()
    run_json = Path(out_dir) / "run.json"
    assert run_json.exists()
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert run_payload["config"]["use_factor_prewhiten"] is False
    prewhiten_diag = Path(out_dir) / "prewhiten_diagnostics.csv"
    prewhiten_summary = Path(out_dir) / "prewhiten_summary.json"
    overlay_toggle = Path(out_dir) / "overlay_toggle.md"
    assert prewhiten_diag.exists()
    assert prewhiten_summary.exists()
    assert overlay_toggle.exists()
    summary_payload = json.loads(prewhiten_summary.read_text(encoding="utf-8"))
    assert summary_payload["mode_effective"] in {"ff5mom", "ff5", "mkt", "off"}
    assert "beta_abs_mean" in summary_payload


def test_run_evaluation_prewhiten_off(tmp_path_factory: pytest.TempPathFactory) -> None:
    returns_csv = _make_returns_csv(tmp_path_factory)
    out_dir = tmp_path_factory.mktemp("outputs_off")
    config = EvalConfig(
        returns_csv=Path(returns_csv),
        factors_csv=None,
        window=20,
        horizon=5,
        out_dir=Path(out_dir),
        shrinker="rie",
        seed=321,
        prewhiten="off",
        use_factor_prewhiten=False,
    )
    outputs = run_evaluation(config)
    summary_payload = json.loads((Path(out_dir) / "prewhiten_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["mode_effective"] == "off"
    diag_df = pd.read_csv(outputs.diagnostics["full"])
    assert not diag_df.empty
    assert (diag_df["prewhiten_mode_effective"] == "off").all()
    assert np.isclose(diag_df["prewhiten_r2_mean"], 0.0).all()


def test_run_evaluation_respects_assets_top(tmp_path_factory: pytest.TempPathFactory) -> None:
    returns_csv = _make_returns_csv(tmp_path_factory)
    out_dir = tmp_path_factory.mktemp("outputs_capped")
    config = EvalConfig(
        returns_csv=Path(returns_csv),
        factors_csv=None,
        window=20,
        horizon=5,
        out_dir=Path(out_dir),
        shrinker="rie",
        seed=4321,
        assets_top=3,
        use_factor_prewhiten=False,
    )
    outputs = run_evaluation(config)
    summary_payload = json.loads((Path(out_dir) / "prewhiten_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["asset_count"] == 3
    resolved_payload = json.loads((Path(out_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved_payload["assets_top"] == 3
def test_run_evaluation_vol_design_logs_state(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("vol_design")
    dates = pd.date_range("2024-02-01", periods=90, freq="B")
    rng = np.random.default_rng(2042)
    scales = np.array([0.004, 0.012, 0.025], dtype=np.float64)
    data = np.vstack([rng.normal(scale=scales[idx % 3], size=6) for idx in range(len(dates))])
    frame = pd.DataFrame(data, index=dates, columns=[f"S{col}" for col in range(6)])
    returns_csv = tmp_dir / "returns.csv"
    frame.reset_index().rename(columns={"index": "date"}).to_csv(returns_csv, index=False)
    out_dir = tmp_dir / "outputs"
    config = EvalConfig(
        returns_csv=returns_csv,
        factors_csv=None,
        window=21,
        horizon=5,
        out_dir=out_dir,
        shrinker="rie",
        seed=17,
        group_design="vol",
        group_min_count=3,
        group_min_replicates=2,
        vol_ewma_span=5,
        use_factor_prewhiten=False,
    )
    outputs = run_evaluation(config)
    detail_path = outputs.diagnostics_detail["all"]
    detail = pd.read_csv(detail_path)
    non_empty = detail[detail["group_label_counts"].astype(str) != ""]
    assert not non_empty.empty
    assert non_empty["group_label_counts"].str.contains("calm:").any()
    assert non_empty["group_label_counts"].str.contains("mid:").any()
    assert non_empty["group_label_counts"].str.contains("crisis:").any()
    assert set(non_empty["vol_state_label"].unique()) <= {"calm", "mid", "crisis", "unknown"}
    assert non_empty["prewhiten_mode_effective"].notna().all()


def test_resolve_eval_config_precedence(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("config_layers")
    thresholds_path = tmp_dir / "thresholds.json"
    thresholds_payload = {
        "window": 40,
        "horizon": 10,
        "shrinker": "oas",
        "seed": 777,
        "calm_quantile": 0.15,
        "crisis_quantile": 0.85,
        "overlay_a_grid": 72,
        "mv_gamma": 0.002,
        "mv_tau": 0.05,
        "assets_top": 120,
    }
    thresholds_path.write_text(json.dumps(thresholds_payload), encoding="utf-8")

    yaml_path = tmp_dir / "config.yaml"
    yaml_path.write_text(
        "\n".join(
        [
            "window: 35",
            "shrinker: lw",
            "seed: 555",
            "calm_quantile: 0.2",
            "assets_top: 90",
            "overlay_a_grid: 120",
            "mv_gamma: 0.0015",
        ]
        ),
        encoding="utf-8",
    )

    cli_args = {
        "returns_csv": tmp_dir / "returns.csv",
        "factors_csv": None,
        "window": 30,
        "horizon": None,
        "start": None,
        "end": None,
        "out": None,
        "shrinker": "rie",
        "seed": None,
        "config": yaml_path,
        "thresholds": thresholds_path,
        "echo_config": False,
        "calm_quantile": None,
        "crisis_quantile": None,
        "vol_ewma_span": None,
        "workers": None,
        "reason_codes": True,
        "mv_tau": 0.03,
        "assets_top": 45,
    }

    resolved = resolve_eval_config(cli_args)
    config = resolved.config

    assert config.window == 30  # CLI override wins
    assert config.horizon == 10  # thresholds layer when CLI absent
    assert config.shrinker == "rie"
    assert config.seed == 555  # YAML overrides thresholds for seed
    assert config.calm_quantile == pytest.approx(0.2)
    assert config.crisis_quantile == pytest.approx(0.85)
    assert resolved.resolved["window"] == 30
    assert resolved.resolved["horizon"] == 10
    assert config.overlay_a_grid == 120
    assert resolved.resolved["overlay_a_grid"] == 120
    assert config.mv_gamma == pytest.approx(0.0015)
    assert config.mv_tau == pytest.approx(0.03)
    assert resolved.resolved["mv_gamma"] == pytest.approx(0.0015)
    assert resolved.resolved["mv_tau"] == pytest.approx(0.03)
    assert resolved.resolved["bootstrap_samples"] == 0
    assert config.assets_top == 45
    assert resolved.resolved["assets_top"] == 45


def test_fpr_guard_for_calibrated_thresholds(tmp_path_factory: pytest.TempPathFactory) -> None:
    calibration_path = Path("calibration/edge_delta_thresholds.json")
    if not calibration_path.exists():
        pytest.skip("calibration thresholds not present")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", {})
    tyler = thresholds.get("tyler", {})
    assert tyler, "expected tyler thresholds in calibration payload"
    if "80x36" in tyler:
        entry = tyler["80x36"]
    else:
        entry = next(iter(tyler.values()))
    fpr = float(entry.get("fpr", 1.0))
    assert fpr <= 0.02 + 1e-6, f"FPR guard breached: {fpr}"


def test_resolve_eval_config_prewhiten_cli(tmp_path_factory: pytest.TempPathFactory) -> None:
    returns_path = Path(_make_returns_csv(tmp_path_factory))
    cli_args = {
        "returns_csv": returns_path,
        "factors_csv": None,
        "prewhiten": "off",
    }
    resolved = resolve_eval_config(cli_args)
    assert resolved.config.prewhiten == "off"
    assert resolved.resolved["prewhiten"] == "off"


@pytest.mark.parametrize("shrinker", ["rie", "lw", "oas", "cc", "quest", "ewma", "factor", "poet"])
def test_resolve_eval_config_shrinkers(tmp_path_factory: pytest.TempPathFactory, shrinker: str) -> None:
    returns_path = Path(_make_returns_csv(tmp_path_factory))
    cli_args = {
        "returns_csv": returns_path,
        "shrinker": shrinker,
    }
    resolved = resolve_eval_config(cli_args)
    assert resolved.config.shrinker == shrinker
    assert resolved.resolved["shrinker"] == shrinker


def test_load_daily_panel_from_parquet(tmp_path_factory: pytest.TempPathFactory) -> None:
    dates = pd.date_range("2024-01-02", periods=40, freq="B")
    rng = np.random.default_rng(7)
    rows = []
    tickers = ["A", "B", "C"]
    for date in dates:
        for ticker in tickers:
            rows.append({"date": date, "ticker": ticker, "ret": float(rng.normal(scale=0.01))})
    frame = pd.DataFrame(rows)
    path = Path(tmp_path_factory.mktemp("parquet")) / "returns.parquet"
    frame.to_parquet(path, index=False)

    panel = load_daily_panel(path, config=DailyLoaderConfig(min_history=20))
    assert panel.returns.shape[0] >= 20
    assert set(panel.returns.columns) == set(tickers)


def test_min_variance_weights_turnover_penalty() -> None:
    sigma = np.array(
        [
            [0.05, 0.02, 0.015],
            [0.02, 0.04, 0.018],
            [0.015, 0.018, 0.03],
        ],
        dtype=np.float64,
    )
    prev = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    base = _min_variance_weights(sigma, gamma=5e-4, tau=0.0, prev_weights=None)
    penalised = _min_variance_weights(sigma, gamma=5e-4, tau=0.5, prev_weights=prev)
    assert np.isclose(base.sum(), 1.0)
    assert np.isclose(penalised.sum(), 1.0)
    baseline_distance = float(np.linalg.norm(base - prev))
    penalised_distance = float(np.linalg.norm(penalised - prev))
    assert penalised_distance <= baseline_distance + 1e-9


def test_dm_alignment_uses_common_windows() -> None:
    metrics = pd.DataFrame(
        [
            {"window_id": 0, "regime": "full", "portfolio": "ew", "estimator": "overlay", "sq_error": 0.5},
            {"window_id": 0, "regime": "full", "portfolio": "ew", "estimator": "baseline", "sq_error": 0.7},
            {"window_id": 1, "regime": "full", "portfolio": "ew", "estimator": "overlay", "sq_error": 0.4},
            # Baseline missing for window 1
            {"window_id": 2, "regime": "full", "portfolio": "ew", "estimator": "overlay", "sq_error": 0.6},
            {"window_id": 2, "regime": "full", "portfolio": "ew", "estimator": "baseline", "sq_error": 0.55},
        ]
    )
    dm_stat, p_value, n_eff = _aligned_dm_stat(metrics, "full", "ew")
    assert n_eff == 2
    assert np.isfinite(dm_stat)
    assert np.isfinite(p_value)


def test_vol_thresholds_use_training_data(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("vol_thresholds")
    cfg = EvalConfig(
        returns_csv=tmp_dir / "returns.csv",
        factors_csv=None,
        window=5,
        horizon=2,
        out_dir=tmp_dir,
        shrinker="rie",
        seed=11,
        overlay_a_grid=80,
        use_factor_prewhiten=False,
    )
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    series = pd.Series([0.1] * 5 + [5.0] * 5, index=dates)
    calm, crisis = _vol_thresholds(series, dates[4], cfg)
    assert calm == pytest.approx(0.1)
    assert crisis == pytest.approx(0.1)
    calm_future, crisis_future = _vol_thresholds(series, dates[-1], cfg)
    assert calm_future == pytest.approx(calm)
    assert crisis_future > crisis


def test_run_evaluation_is_reproducible(tmp_path_factory: pytest.TempPathFactory) -> None:
    returns_csv = _make_returns_csv(tmp_path_factory)
    out_one = tmp_path_factory.mktemp("outputs_one")
    out_two = tmp_path_factory.mktemp("outputs_two")

    base_kwargs = dict(
        returns_csv=Path(returns_csv),
        factors_csv=None,
        window=18,
        horizon=4,
        shrinker="rie",
        seed=321,
        overlay_a_grid=90,
        use_factor_prewhiten=False,
    )

    cfg_one = EvalConfig(out_dir=Path(out_one), **base_kwargs)
    cfg_two = EvalConfig(out_dir=Path(out_two), **base_kwargs)
    out_workers = tmp_path_factory.mktemp("outputs_workers")
    cfg_workers = EvalConfig(out_dir=Path(out_workers), workers=2, **base_kwargs)

    outputs_one = run_evaluation(cfg_one)
    outputs_two = run_evaluation(cfg_two)
    outputs_workers = run_evaluation(cfg_workers)

    metrics_one = pd.read_csv(outputs_one.metrics["full"]).sort_index(axis=1)
    metrics_two = pd.read_csv(outputs_two.metrics["full"]).sort_index(axis=1)
    metrics_workers = pd.read_csv(outputs_workers.metrics["full"]).sort_index(axis=1)
    pd.testing.assert_frame_equal(metrics_one, metrics_two)
    pd.testing.assert_frame_equal(metrics_one, metrics_workers)

    risk_one = pd.read_csv(outputs_one.risk["full"]).sort_index(axis=1)
    risk_two = pd.read_csv(outputs_two.risk["full"]).sort_index(axis=1)
    risk_workers = pd.read_csv(outputs_workers.risk["full"]).sort_index(axis=1)
    pd.testing.assert_frame_equal(risk_one, risk_two)
    pd.testing.assert_frame_equal(risk_one, risk_workers)

    dm_one = pd.read_csv(outputs_one.dm["full"]).sort_index(axis=1)
    dm_two = pd.read_csv(outputs_two.dm["full"]).sort_index(axis=1)
    dm_workers = pd.read_csv(outputs_workers.dm["full"]).sort_index(axis=1)
    pd.testing.assert_frame_equal(dm_one, dm_two)
    pd.testing.assert_frame_equal(dm_one, dm_workers)

    diag_one = pd.read_csv(outputs_one.diagnostics["full"]).sort_index(axis=1)
    diag_two = pd.read_csv(outputs_two.diagnostics["full"]).sort_index(axis=1)
    diag_workers = pd.read_csv(outputs_workers.diagnostics["full"]).sort_index(axis=1)
    shared_cols = [col for col in diag_one.columns if col != "resolved_config_path"]
    pd.testing.assert_frame_equal(diag_one[shared_cols], diag_two[shared_cols])
    pd.testing.assert_frame_equal(diag_one[shared_cols], diag_workers[shared_cols])

    detail_one = pd.read_csv(outputs_one.diagnostics_detail["full"]).sort_index(axis=1)
    detail_two = pd.read_csv(outputs_two.diagnostics_detail["full"]).sort_index(axis=1)
    detail_workers = pd.read_csv(outputs_workers.diagnostics_detail["full"]).sort_index(axis=1)
    detail_cols = [col for col in detail_one.columns if col != "resolved_config_path"]
    pd.testing.assert_frame_equal(detail_one[detail_cols], detail_two[detail_cols])
    pd.testing.assert_frame_equal(detail_one[detail_cols], detail_workers[detail_cols])


def test_bootstrap_bands_populate_for_overlay(tmp_path_factory: pytest.TempPathFactory) -> None:
    returns_csv = _make_returns_csv(tmp_path_factory)
    out_dir = tmp_path_factory.mktemp("bootstrap_outputs")
    config = EvalConfig(
        returns_csv=Path(returns_csv),
        factors_csv=None,
        window=18,
        horizon=4,
        out_dir=Path(out_dir),
        shrinker="rie",
        seed=432,
        bootstrap_samples=20,
        use_factor_prewhiten=False,
    )
    outputs = run_evaluation(config)
    summary = pd.read_csv(outputs.metrics["full"])
    overlay_rows = summary[(summary["estimator"] == "overlay") & (summary["portfolio"] == "ew")]
    assert overlay_rows["delta_mse_ci_lower"].notna().any()
    assert overlay_rows["delta_mse_ci_upper"].notna().any()
