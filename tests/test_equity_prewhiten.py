from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.equity_panel import run as equity_run


def _write_returns_csv(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-02", periods=35, freq="B")
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(11)
    records: list[dict[str, object]] = []
    for date in dates:
        for ticker in tickers:
            records.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "ret": float(rng.normal(scale=0.01)),
                }
            )
    frame = pd.DataFrame.from_records(records)
    path = tmp_path / "returns.csv"
    frame.to_csv(path, index=False)
    return path


def _write_factor_csv(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-02", periods=35, freq="B")
    rng = np.random.default_rng(17)
    data = rng.normal(scale=0.005, size=(len(dates), 6))
    columns = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    frame = pd.DataFrame(data, index=dates, columns=columns)
    frame.insert(0, "date", dates.strftime("%Y-%m-%d"))
    path = tmp_path / "factors.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.slow
def test_run_experiment_emits_prewhiten_columns(tmp_path: Path) -> None:
    returns_csv = _write_returns_csv(tmp_path)
    factors_csv = _write_factor_csv(tmp_path)
    outputs_dir = tmp_path / "outputs"
    cfg = {
        "data_path": str(returns_csv),
        "start_date": "2024-01-02",
        "end_date": "2024-02-16",
        "window_weeks": 4,
        "horizon_weeks": 1,
        "output_dir": str(outputs_dir),
        "design": "oneway",
        "nested_replicates": 5,
        "estimator": "dealias",
        "factor_csv": str(factors_csv),
        "prewhiten": "ff5mom",
        "use_factor_prewhiten": True,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    equity_run.run_experiment(
        config_path,
        progress_override=False,
        precompute_panel=False,
        cache_dir_override=str(tmp_path / "cache"),
    )

    run_dirs = list(outputs_dir.glob("*"))
    assert run_dirs, "expected at least one run directory"
    rolling_path = run_dirs[0] / "rolling_results.csv"
    assert rolling_path.exists()
    df = pd.read_csv(rolling_path)
    required_cols = {
        "prewhiten_mode_requested",
        "prewhiten_mode_effective",
        "prewhiten_r2_mean",
        "prewhiten_factor_count",
        "prewhiten_factors",
        "factor_present",
    }
    assert required_cols <= set(df.columns)
    assert df["prewhiten_mode_effective"].str.lower().eq("ff5mom").any()
    diag_path = run_dirs[0] / "prewhiten_diagnostics.csv"
    summary_path = run_dirs[0] / "prewhiten_summary.json"
    assert diag_path.exists()
    assert summary_path.exists()
    diag_df = pd.read_csv(diag_path)
    assert "r_squared" in diag_df.columns
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["mode_effective"] == "ff5mom"
