from __future__ import annotations

import json
from pathlib import Path

from src.fjs.config import (
    DetectionSettings,
    get_detection_settings,
    load_detection_settings,
    override_detection_settings,
)


def test_load_detection_settings_uses_thresholds(tmp_path: Path) -> None:
    thresholds = {
        "margin_grid": [0.05, 0.075, 0.1],
        "recommended": {
            "min_margin": 0.125,
            "fpr": 0.01,
            "tpr": {"6.00": 0.9},
        },
    }
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps(thresholds), encoding="utf-8")

    config_path = tmp_path / "detection.yaml"
    config_path.write_text(
        "\n".join(
            [
                "thresholds_path: " + str(threshold_path),
                "delta: 0.3",
                "a_grid_size: 64",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_detection_settings(config_path=config_path)
    assert settings.min_margin == 0.125
    assert settings.delta == 0.3
    assert settings.a_grid_size == 64


def test_override_detection_settings_roundtrip() -> None:
    custom = DetectionSettings(delta=0.2, min_margin=0.15, q_max=2)
    override_detection_settings(custom)
    try:
        loaded = get_detection_settings()
        assert loaded.delta == 0.2
        assert loaded.min_margin == 0.15
        assert loaded.q_max == 2
    finally:
        override_detection_settings(None)
