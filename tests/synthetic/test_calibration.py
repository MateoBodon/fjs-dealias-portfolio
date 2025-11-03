from __future__ import annotations

from pathlib import Path

from synthetic.calibration import CalibrationConfig, calibrate_thresholds, write_thresholds


def test_calibrate_thresholds_controls_fpr(tmp_path: Path) -> None:
    config = CalibrationConfig(
        p_assets=8,
        n_groups=30,
        replicates=3,
        alpha=0.02,
        trials_null=40,
        trials_alt=20,
        delta_frac_grid=(0.015, 0.03),
        stability_grid=(0.35,),
        spike_strength=2.5,
        edge_modes=("tyler",),
        seed=321,
    )
    result = calibrate_thresholds(config)
    entry = result.thresholds["tyler"]
    assert entry.fpr <= config.alpha
    assert entry.trials_null == config.trials_null

    out_path = tmp_path / "thresholds.json"
    write_thresholds(result, out_path)
    payload = out_path.read_text(encoding="utf-8")
    assert "tyler" in payload
    assert "delta_frac" in payload
