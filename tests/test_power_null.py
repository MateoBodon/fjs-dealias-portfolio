from __future__ import annotations

import numpy as np

from experiments.synthetic import power_null as power_null_mod
from experiments.synthetic.power_null import DEFAULT_CONFIG, run_trials, summarise_results


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
