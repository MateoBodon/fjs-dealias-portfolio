from __future__ import annotations

from pathlib import Path
import json

import pytest

from fjs.gating import count_isolated_outliers, lookup_calibrated_delta, select_top_k


def test_count_isolated_outliers_zero_when_missing_isolated() -> None:
    detections = [
        {"lambda_hat": 2.1, "pre_outlier_count": 3, "stability_margin": 0.4},
        {"lambda_hat": 1.9, "pre_outlier_count": 0, "stability_margin": 0.2},
    ]
    assert count_isolated_outliers(detections, None, None) == 0


def test_select_top_k_prefers_high_score_and_edge_margin() -> None:
    detections = [
        {
            "lambda_hat": 2.5,
            "target_energy": 0.8,
            "stability_margin": 0.6,
            "edge_margin": 0.05,
        },
        {
            "lambda_hat": 2.3,
            "target_energy": 0.9,
            "stability_margin": 0.7,
            "edge_margin": 0.1,
        },
        {
            "lambda_hat": 2.4,
            "target_energy": 0.8,
            "stability_margin": 0.6,
            "edge_margin": 0.2,
        },
    ]
    selected, discarded = select_top_k(detections, 2)
    assert len(selected) == 2
    assert len(discarded) == 1
    # Highest score should be the second detection (0.9 * 0.7)
    assert selected[0]["lambda_hat"] == 2.3
    # Tie in score uses edge margin (0.2 > 0.05)
    assert selected[1]["lambda_hat"] == 2.4
    assert discarded[0]["lambda_hat"] == 2.5


def test_lookup_calibrated_delta_reads_json(tmp_path: Path) -> None:
    payload = {
        "thresholds": {
            "scm": {"40x120": {"delta_frac": 0.025, "fpr": 0.004, "trials": 200}}
        }
    }
    cal_path = tmp_path / "delta.json"
    cal_path.write_text(json.dumps(payload), encoding="utf-8")

    value = lookup_calibrated_delta("SCM", 40, 120, calibration_path=cal_path)
    assert value == pytest.approx(0.025)
    missing = lookup_calibrated_delta("tyler", 40, 120, calibration_path=cal_path)
    assert missing is None
