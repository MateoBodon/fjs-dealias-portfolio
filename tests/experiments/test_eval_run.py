from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.eval.run import EvalConfig, main, run_evaluation, save_results


def _synthetic_returns(days: int = 80, assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=days, freq="B")
    data = rng.normal(scale=0.01, size=(days, assets))
    cols = [f"asset_{i}" for i in range(assets)]
    return pd.DataFrame(data, index=dates, columns=cols)


def test_run_evaluation_generates_metrics(tmp_path) -> None:
    returns = _synthetic_returns(days=70, assets=3)
    config = EvalConfig(window=20, horizon=5, output_dir=tmp_path)
    result = run_evaluation(returns, config)
    assert {"estimator", "portfolio", "mse"}.issubset(result.metrics.columns)
    save_results(result, config)
    metrics_path = tmp_path / "metrics_full.csv"
    assert metrics_path.exists()


def test_main_writes_artifacts(tmp_path) -> None:
    returns = _synthetic_returns(days=90, assets=3)
    long_df = returns.stack().reset_index()
    long_df.columns = ["date", "ticker", "ret"]
    csv_path = tmp_path / "returns.csv"
    long_df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    main([
        "--returns-csv",
        str(csv_path),
        "--window",
        "20",
        "--horizon",
        "5",
        "--out",
        str(out_dir),
    ])

    assert (out_dir / "metrics_full.csv").exists()
    assert (out_dir / "dm_full.csv").exists()
    assert (out_dir / "var_full.csv").exists()
