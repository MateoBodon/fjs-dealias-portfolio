import json
from pathlib import Path

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
    entry = payload["thresholds"]["tyler"]["200x70"]
    assert "run_name" in entry and "git_sha" in entry
    delta = lookup_calibrated_delta("tyler", 200, 70, calibration_path=path, design="nested")
    assert delta == entry["delta_frac"]


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
