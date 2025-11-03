from __future__ import annotations

import numpy as np
import pytest

from baselines.covariance import ewma_covariance, quest_covariance
from fjs.dealias import Detection
from fjs.overlay import OverlayConfig, apply_overlay, detect_spikes


def _make_detection(mu: float, vec: np.ndarray, edge_margin: float = 0.2) -> Detection:
    vec = np.asarray(vec, dtype=np.float64)
    unit = vec / np.linalg.norm(vec)
    return Detection(
        mu_hat=float(mu),
        lambda_hat=float(mu),
        a=[1.0, 0.0],
        components=[float(mu), 0.0],
        eigvec=unit,
        stability_margin=0.1,
        edge_margin=float(edge_margin),
        buffer_margin=0.1,
        t_values=None,
        admissible_root=True,
        solver_used="grid",
        z_plus=None,
        threshold_main=None,
        z_plus_low=None,
        z_plus_high=None,
        threshold_low=None,
        threshold_high=None,
        stability_margin_low=None,
        stability_margin_high=None,
        sensitivity_low_accept=None,
        sensitivity_high_accept=None,
        target_energy=None,
        target_index=0,
        off_component_ratio=0.0,
        pre_outlier_count=0,
        edge_mode="tyler",
        edge_scale=1.0,
    )


def test_apply_overlay_substitutes_detected_eigenvalues() -> None:
    sample_cov = np.diag([5.0, 1.0, 0.5])
    baseline = np.diag([1.5, 1.0, 0.5])
    detection = _make_detection(3.0, np.array([1.0, 0.0, 0.0]))

    result = apply_overlay(sample_cov, [detection], baseline_covariance=baseline)
    assert pytest.approx(result[0, 0], rel=1e-6) == 3.0
    assert pytest.approx(result[1, 1], rel=1e-6) == 1.0
    assert pytest.approx(result[2, 2], rel=1e-6) == 0.5


def test_apply_overlay_respects_detection_cap() -> None:
    sample_cov = np.diag([4.0, 3.0, 1.0])
    baseline = np.diag([2.0, 2.0, 1.0])
    det1 = _make_detection(3.5, np.array([1.0, 0.0, 0.0]), edge_margin=0.3)
    det2 = _make_detection(2.5, np.array([0.0, 1.0, 0.0]), edge_margin=0.2)
    config = OverlayConfig(max_detections=1, q_max=2)

    result = apply_overlay(sample_cov, [det1, det2], baseline_covariance=baseline, config=config)
    assert pytest.approx(result[0, 0], rel=1e-6) == 3.5
    assert pytest.approx(result[1, 1], rel=1e-6) == 2.0


def test_detect_spikes_uses_tyler_edge_mode() -> None:
    rng = np.random.default_rng(123)
    groups = 25
    replicates = 2
    features = 8
    direction = rng.normal(size=features)
    direction /= np.linalg.norm(direction)
    group_scores = rng.normal(scale=3.0, size=groups)
    between = np.outer(group_scores, direction)
    within = rng.normal(scale=0.5, size=(groups, replicates, features))
    observations = between[:, None, :] + within
    y = observations.reshape(groups * replicates, features)
    labels = np.repeat(np.arange(groups, dtype=np.intp), replicates)

    detections = detect_spikes(y, labels, config=OverlayConfig(edge_mode="tyler", q_max=2, a_grid=90))
    assert detections
    assert all(det.get("edge_mode") == "tyler" for det in detections)
    assert len(detections) <= 2


def test_apply_overlay_with_ewma_shrinker_matches_baseline() -> None:
    rng = np.random.default_rng(99)
    observations = rng.normal(scale=0.3, size=(80, 4))
    sample_cov = np.cov(observations, rowvar=False, ddof=1)
    config = OverlayConfig(shrinker="ewma", ewma_halflife=5.0, require_isolated=False, q_max=1)
    result = apply_overlay(sample_cov, [], observations=observations, config=config)
    baseline = ewma_covariance(observations, halflife=5.0)
    assert np.allclose(result, baseline, atol=1e-8)


def test_apply_overlay_with_quest_shrinker_matches_baseline() -> None:
    rng = np.random.default_rng(101)
    observations = rng.normal(scale=0.4, size=(70, 5))
    sample_cov = np.cov(observations, rowvar=False, ddof=1)
    config = OverlayConfig(shrinker="quest", require_isolated=False, q_max=1)
    result = apply_overlay(sample_cov, [], observations=observations, config=config)
    baseline = quest_covariance(sample_cov, sample_count=observations.shape[0])
    assert np.allclose(result, baseline, atol=1e-8)
