from __future__ import annotations

import numpy as np

from experiments.synthetic import power_null as power_null_mod
from experiments.synthetic.power_null import (
    DEFAULT_CONFIG,
    calibrate_delta_thresholds,
    run_trials,
    summarise_results,
)


def test_power_null_summary_behaves() -> None:
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 8,
        "n_groups": 8,
        "replicates": 2,
        "trials_null": 4,
        "trials_power": 4,
        "spike_grid": [3.5],
        "a_grid": 60,
    })
    rng = np.random.default_rng(99)
    original_gating = dict(power_null_mod.GATING_SETTINGS)
    power_null_mod.GATING_SETTINGS = {"default": original_gating["default"]}
    try:
        results = run_trials(
            config=config,
            edge_modes=["scm"],
            trials_null=int(config["trials_null"]),
            trials_power=int(config["trials_power"]),
            spike_grid=config["spike_grid"],
            two_spike=False,
            rng=rng,
        )
    finally:
        power_null_mod.GATING_SETTINGS = original_gating
    summary = summarise_results(results)
    assert not summary.empty

    null_rows = summary[summary["scenario"] == "null"]
    assert (null_rows["detection_rate"] < 0.5).all()

    power_rows = summary[summary["scenario"] == "power"]
    assert not power_rows.empty
    assert (power_rows["detection_rate"] > 0).any()
    assert power_rows["delta_mse_vs_lw"].notna().any()
    assert power_rows["delta_qlike_vs_lw"].notna().any()


def test_calibrate_delta_thresholds_returns_lookup() -> None:
    config = DEFAULT_CONFIG.copy()
    config.update({
        "n_assets": 6,
        "n_groups": 6,
        "replicates": 2,
        "trials_null": 6,
        "delta_frac": 0.01,
    })
    rng = np.random.default_rng(321)
    calibration = calibrate_delta_thresholds(
        config=config,
        edge_modes=["scm"],
        trials_null=int(config["trials_null"]),
        alpha=0.5,
        rng=rng,
        delta_grid=[0.0, 0.01, 0.02],
    )
    assert calibration.get("thresholds")
    scm_thresholds = calibration["thresholds"].get("scm")
    assert scm_thresholds, "Expected SCM entry in thresholds"
    key, entry = next(iter(scm_thresholds.items()))
    assert "x" in key  # key format "pxT"
    assert "delta_frac" in entry
    assert 0.0 <= entry["delta_frac"] <= 0.02
