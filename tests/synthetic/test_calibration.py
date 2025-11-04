from __future__ import annotations

import json
from pathlib import Path

from synthetic.calibration import CalibrationConfig, calibrate_thresholds, write_thresholds


def test_calibrate_thresholds_controls_fpr(tmp_path: Path) -> None:
    config = CalibrationConfig(
        p_assets=6,
        n_groups=18,
        replicates=3,
        alpha=0.02,
        trials_null=16,
        trials_alt=16,
        delta_abs=0.35,
        delta_frac_grid=(0.01, 0.02),
        stability_grid=(0.3,),
        spike_strength=3.0,
        edge_modes=("scm",),
        seed=321,
    )
    result = calibrate_thresholds(config)
    assert result.grid, "expected grid statistics to be populated"
    entry = result.thresholds["scm"]
    assert entry.fpr <= config.alpha
    assert entry.trials_null == config.trials_null

    out_path = tmp_path / "thresholds.json"
    write_thresholds(result, out_path)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "scm" in payload["thresholds"]
    entry = payload["thresholds"]["scm"]
    assert entry["delta_frac"] >= 0.0
    assert "grid" in payload
