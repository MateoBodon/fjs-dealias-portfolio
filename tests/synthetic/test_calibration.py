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


def test_edge_delta_thresholds_real_run() -> None:
    path = Path("calibration/edge_delta_thresholds.json")
    assert path.exists(), "calibration/edge_delta_thresholds.json missing; run Step 3 calibration."
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", {})
    tyler = thresholds.get("tyler")
    assert isinstance(tyler, dict), "tyler thresholds missing"
    g36 = tyler.get("G36")
    assert isinstance(g36, dict), "expected G36 bucket"

    # Replicate bins 12-16 and 17-22 should exist with the 64-96 asset band.
    for r_label in ("12-16", "17-22"):
        r_bucket = g36.get(r_label)
        assert isinstance(r_bucket, dict), f"replicate bin {r_label} missing"
        entry = r_bucket.get("64-96")
        assert isinstance(entry, dict), f"asset bin 64-96 missing for {r_label}"
        fpr = float(entry["fpr"])
        assert fpr <= 0.02, f"tyler FPR too high for {r_label} ({fpr:.3f})"
        assert entry["replicates_bin"] == r_label
        assert entry["p_bin"] == "64-96"
