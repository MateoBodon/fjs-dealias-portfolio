from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert "detection_rate" in diag_df.columns
