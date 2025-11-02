from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.ablate.run import run_ablation


def _write_returns(path: Path, rows: int = 40, cols: int = 4) -> None:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(321)
    data = rng.normal(scale=0.01, size=(rows, cols))
    frame = pd.DataFrame(data, index=dates, columns=[f"T{i}" for i in range(cols)])
    out = frame.reset_index().rename(columns={"index": "date"})
    out.to_csv(path, index=False)


def _write_config(path: Path, returns_path: Path) -> None:
    payload = {
        "defaults": {
            "window": 12,
            "horizon": 3,
            "shrinker": "rie",
            "seed": 7,
            "overlay_a_grid": 36,
            "overlay_seed": 7,
            "require_isolated": True,
            "q_max": 1,
            "edge_mode": "tyler",
            "angle_min_cos": 0.9,
            "alignment_top_p": 2,
            "cs_drop_top_frac": 0.05,
            "prewhiten": True,
            "calm_quantile": 0.2,
            "crisis_quantile": 0.8,
            "vol_ewma_span": 8,
            "mv_gamma": 0.0005,
            "mv_tau": 0.0,
            "bootstrap_samples": 0,
        },
        "grid": {
            "panel": ["test"],
            "edge_mode": ["tyler"],
            "require_isolated": [True],
            "angle_min_cos": [0.9],
            "q_max": [1],
            "cs_drop_top_frac": [0.05],
            "shrinker": ["rie"],
            "prewhiten": [True, False],
        },
        "panels": {
            "test": {
                "returns_csv": str(returns_path),
            }
        },
        "io": {
            "cache_dir": str(path.parent / "cache"),
            "out_csv": str(path.parent / "ablation_matrix.csv"),
        },
        "summary": {
            "regime": "full",
            "portfolios": ["ew", "mv"],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_run_ablation_small_grid(tmp_path: Path) -> None:
    returns_path = tmp_path / "returns.csv"
    _write_returns(returns_path)

    config_path = tmp_path / "ablation.yaml"
    _write_config(config_path, returns_path)

    out_csv = run_ablation(config_path, force=True)

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert df.shape[0] == 2
    required_columns = {
        "panel",
        "prewhiten",
        "ew_delta_mse_vs_baseline",
        "ew_delta_mse_vs_baseline_vs_default",
        "detections_mean",
        "detections_mean_vs_default",
    }
    assert required_columns.issubset(df.columns)

    base_row = df[df["prewhiten"] == True].iloc[0]
    diff_val = base_row["ew_delta_mse_vs_baseline_vs_default"]
    assert math.isclose(diff_val, 0.0, abs_tol=1e-12) or np.isnan(diff_val)

    variant_row = df[df["prewhiten"] == False].iloc[0]
    assert "ew_delta_mse_vs_baseline_vs_default" in variant_row
    # Ensure cache metadata recorded
    run_path = Path(base_row["run_path"])
    assert run_path.exists()
