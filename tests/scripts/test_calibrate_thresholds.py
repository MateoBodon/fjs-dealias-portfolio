from __future__ import annotations

import json

from scripts.calibrate_thresholds import CalibrationConfig, run_calibration, save_results


def test_run_calibration_produces_threshold_artifacts(tmp_path) -> None:
    config = CalibrationConfig(
        p=6,
        n=30,
        mu=(4.0,),
        out=tmp_path,
        trials=5,
        seed=123,
        margin_grid=(0.05, 0.1),
    )
    result = run_calibration(config)
    save_results(result, config)

    data = json.loads((tmp_path / "thresholds.json").read_text())
    assert data["p"] == 6
    assert data["n"] == 30
    assert data["margin_grid"] == [0.05, 0.1]
    assert data["recommended"]["min_margin"] in data["margin_grid"]

    csv_path = tmp_path / "roc_mu4_00.csv"
    assert csv_path.exists()
    content = csv_path.read_text().strip().splitlines()
    # header + two rows
    assert len(content) == 3
