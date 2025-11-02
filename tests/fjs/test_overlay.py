from __future__ import annotations

import numpy as np
import pytest

from src.fjs.overlay import Detection, OverlayConfig, apply_overlay, detect_spikes


def test_detection_dataclass_fields() -> None:
    vec = np.ones(3)
    detection = Detection(index=0, eigenvalue=1.0, margin=0.0, score=0.0, direction=vec)
    assert detection.index == 0
    assert detection.eigenvalue == 1.0


def test_overlay_stubs_raise_not_implemented() -> None:
    cov = np.eye(3)
    config = OverlayConfig()
    with pytest.raises(NotImplementedError):
        detect_spikes(cov, config=config)
    detections = []
    with pytest.raises(NotImplementedError):
        apply_overlay(cov, np.eye(3), detections)
