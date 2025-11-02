from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.eval.config import resolve_eval_config
from experiments.eval.diagnostics import DiagnosticReason
from experiments.eval.run import EvalConfig, run_evaluation, _vol_thresholds


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
    )
    outputs = run_evaluation(config)

    resolved_path = Path(out_dir) / "resolved_config.json"
    assert resolved_path.exists()
    resolved_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved_payload["returns_csv"].endswith("returns.csv")

    full_metrics = outputs.metrics["full"]
    assert full_metrics.exists()
    metrics_df = pd.read_csv(full_metrics)
    assert {"estimator", "portfolio", "delta_mse_vs_baseline"}.issubset(metrics_df.columns)

    full_risk = outputs.risk["full"]
    assert full_risk.exists()
    risk_df = pd.read_csv(full_risk)
    assert {"vaR95", "es95"}.issubset(risk_df.columns)

    full_dm = outputs.dm["full"]
    assert full_dm.exists()
    dm_df = pd.read_csv(full_dm)
    assert {"dm_stat", "p_value"}.issubset(dm_df.columns)

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
    }.issubset(diag_df.columns)
    if not diag_df.empty:
        reason_values = set(diag_df["reason_code"].dropna().unique())
        allowed = {reason.value for reason in DiagnosticReason} | {""}
        assert reason_values <= allowed
        assert diag_df["resolved_config_path"].str.endswith("resolved_config.json").all()
        assert diag_df["calm_threshold"].notna().any()
        assert diag_df["crisis_threshold"].notna().any()
        assert diag_df["vol_signal"].notna().any()


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
                "overlay_a_grid: 120",
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
