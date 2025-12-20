import json
from pathlib import Path

import pytest

from fjs.gating import lookup_calibrated_delta


def test_lookup_calibrated_delta_uses_design_file() -> None:
    path = Path("calibration/nested_edge_delta_thresholds.json")
    assert path.exists()
    delta = lookup_calibrated_delta(
        "tyler", 200, 70, calibration_path=path, design="nested"
    )
    assert delta is not None and delta > 0


def test_lookup_calibrated_delta_metadata_present() -> None:
    path = Path("calibration/nested_edge_delta_thresholds.json")
    payload = json.loads(path.read_text())
    meta = payload["metadata"]
    required_meta = [
        "run_name",
        "timestamp_utc",
        "git_sha",
        "config_hash",
        "target_fpr",
        "achieved_fpr",
        "null_trials",
        "operating_points",
    ]
    for key in required_meta:
        assert key in meta
    assert meta["operating_points"], "operating_points should list calibration points"

    entry = payload["thresholds"]["tyler"]["200x70"]
    for key in ("p_assets", "n_obs", "edge_mode", "weeks_common", "replicates"):
        assert key in entry
    delta = lookup_calibrated_delta("tyler", 200, 70, calibration_path=path, design="nested")
    assert delta == pytest.approx(entry["delta_frac"])


def test_lookup_calibrated_delta_design_thresholds_precedence(tmp_path: Path) -> None:
    file_path = tmp_path / "cal.json"
    file_path.write_text(
        json.dumps(
            {
                "design_thresholds": {
                    "nested": {"tyler": {"10x10": {"delta_frac": 0.123}}},
                },
                "thresholds": {"tyler": {"10x10": {"delta_frac": 0.9}}},
            }
        )
    )
    delta = lookup_calibrated_delta("tyler", 10, 10, calibration_path=file_path, design="nested")
    assert abs(delta - 0.123) < 1e-9


def test_lookup_calibrated_delta_requires_matching_design(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_design.json"
    file_path.write_text(
        json.dumps({"thresholds": {"tyler": {"10x10": {"delta_frac": 0.2}}}}),
        encoding="utf-8",
    )
    missing = lookup_calibrated_delta("tyler", 10, 10, calibration_path=file_path, design="nested")
    assert missing is None
    fallback = lookup_calibrated_delta("tyler", 10, 10, calibration_path=file_path, design=None)
    assert fallback == pytest.approx(0.2)
