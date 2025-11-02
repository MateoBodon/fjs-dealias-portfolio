from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.eval.config import resolve_eval_config
from experiments.eval.diagnostics import DiagnosticReason
from experiments.eval.run import EvalConfig, run_evaluation


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
    assert {"detection_rate", "reason_code", "resolved_config_path"}.issubset(diag_df.columns)
    if not diag_df.empty:
        reason_values = set(diag_df["reason_code"].dropna().unique())
        allowed = {reason.value for reason in DiagnosticReason} | {""}
        assert reason_values <= allowed
        assert diag_df["resolved_config_path"].str.endswith("resolved_config.json").all()


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
