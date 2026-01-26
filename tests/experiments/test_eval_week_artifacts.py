from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.eval.run import EvalConfig, run_evaluation


def _make_week_returns_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    dates = dates[dates.weekday != 0]
    rng = np.random.default_rng(2026)
    returns = rng.normal(scale=0.01, size=(len(dates), 6))
    frame = pd.DataFrame(returns, index=dates, columns=[f"A{i}" for i in range(6)])
    path = tmp_path_factory.mktemp("week_returns") / "returns.csv"
    frame.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)
    return path


def test_week_early_exit_writes_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    returns_csv = _make_week_returns_csv(tmp_path_factory)
    out_dir = tmp_path_factory.mktemp("week_outputs")
    config = EvalConfig(
        returns_csv=Path(returns_csv),
        factors_csv=None,
        window=20,
        horizon=5,
        out_dir=Path(out_dir),
        shrinker="rie",
        seed=7,
        prewhiten="off",
        use_factor_prewhiten=False,
        group_design="week",
        group_min_count=1,
        group_min_replicates=1,
        mv_box_hi=1.0,
    )
    run_evaluation(config)

    resolved_path = Path(out_dir) / "resolved_config.json"
    run_path = Path(out_dir) / "run.json"
    run_log = Path(out_dir) / "run.log"
    assert resolved_path.exists()
    assert run_path.exists()
    assert run_log.exists()

    log_lines = run_log.read_text(encoding="utf-8").strip().splitlines()
    assert log_lines
    assert log_lines[0].startswith("START ")
    assert log_lines[-1].startswith("END ")

    payload = json.loads(run_path.read_text(encoding="utf-8"))
    assert payload.get("status") in {"ok", "no_windows", "error"}
    assert payload.get("status") == "no_windows"
    assert isinstance(payload.get("stage"), str)
    assert payload.get("stage")
    assert payload.get("resolved_config_path", "").endswith("resolved_config.json")
