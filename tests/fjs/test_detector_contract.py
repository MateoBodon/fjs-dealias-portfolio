from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from fjs.detector_contract import assess_power_curve
from fjs.overlay import OverlayConfig, detect_spikes

pytestmark = pytest.mark.unit

REFERENCE_CURVE = (
    Path(__file__).resolve().parents[2]
    / "docs/artifacts/detector-contract-reference/ticket24_week_full_fix/curve.csv"
)


def _reference_rows() -> list[dict[str, str]]:
    with REFERENCE_CURVE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_historical_week_curve_is_rejected_by_detector_stop_line() -> None:
    assessment = assess_power_curve(_reference_rows())
    assert not assessment.passed
    assert assessment.null_detection_rate == 0.0
    assert assessment.strong_detection_rate == 0.0
    assert assessment.strong_acceptance_rate == 0.0
    assert set(assessment.reasons) >= {
        "strong_signal_detection_below_minimum",
        "strong_signal_acceptance_below_minimum",
        "detection_gain_below_minimum",
    }


def test_historical_week_curve_cannot_support_between_component_power() -> None:
    with pytest.raises(ValueError, match="missing inject_mode provenance"):
        assess_power_curve(_reference_rows(), expected_inject_mode="between")


def test_power_curve_rejects_mismatched_injection_component() -> None:
    rows = [dict(row, inject_mode="total") for row in _reference_rows()]
    with pytest.raises(ValueError, match="does not match the target"):
        assess_power_curve(rows, expected_inject_mode="between")


def test_power_curve_rejects_acceptance_above_detection() -> None:
    rows = [
        {
            "mu": 0.0,
            "detection_rate": 0.05,
            "acceptance_rate": 0.04,
            "n_windows": 200,
        },
        {
            "mu": 1.5,
            "detection_rate": 0.85,
            "acceptance_rate": 0.90,
            "n_windows": 200,
        },
    ]

    assessment = assess_power_curve(rows)

    assert not assessment.passed
    assert assessment.reasons == ("acceptance_exceeds_detection",)


def test_planted_oneway_mechanism_is_labeled_fjs() -> None:
    rng = np.random.default_rng(123)
    n_groups, replicates, n_assets = 25, 2, 8
    direction = rng.normal(size=n_assets)
    direction /= np.linalg.norm(direction)
    between = np.outer(rng.normal(scale=3.0, size=n_groups), direction)
    within = rng.normal(scale=0.5, size=(n_groups, replicates, n_assets))
    observations = (between[:, None, :] + within).reshape(
        n_groups * replicates, n_assets
    )
    groups = np.repeat(np.arange(n_groups, dtype=np.intp), replicates)

    detections = detect_spikes(
        observations,
        groups,
        config=OverlayConfig(edge_mode="tyler", q_max=2, a_grid=90),
    )

    assert detections
    assert {detection["candidate_source"] for detection in detections} == {"fjs"}
