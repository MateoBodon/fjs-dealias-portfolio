from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from baselines.covariance import (
    cc_covariance,
    ewma_covariance,
    lw_covariance,
    oas_covariance,
    quest_covariance,
)
from fjs.dealias import Detection
from fjs.overlay import OverlayConfig, apply_overlay, detect_spikes


def _make_detection(
    mu: float,
    vec: np.ndarray,
    *,
    edge_margin: float = 0.2,
    stability: float = 0.3,
    target_energy: float | None = None,
    alignment: float = 1.0,
    pre_outlier: int | None = 1,
) -> Detection:
    vec = np.asarray(vec, dtype=np.float64)
    unit = vec / np.linalg.norm(vec)
    det: Detection = Detection(
        mu_hat=float(mu),
        lambda_hat=float(mu),
        a=[1.0, 0.0],
        components=[float(mu), 0.0],
        eigvec=unit,
        stability_margin=float(stability),
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
        pre_outlier_count=pre_outlier,
        edge_mode="tyler",
        edge_scale=1.0,
    )
    det["target_energy"] = float(target_energy) if target_energy is not None else None
    det["alignment_cos"] = float(alignment)
    return det


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


def test_apply_overlay_with_lw_shrinker_matches_baseline() -> None:
    rng = np.random.default_rng(73)
    observations = rng.normal(scale=0.35, size=(90, 6))
    sample_cov = np.cov(observations, rowvar=False, ddof=1)
    config = OverlayConfig(shrinker="lw", require_isolated=False, q_max=1)
    result = apply_overlay(sample_cov, [], observations=observations, config=config)
    baseline = lw_covariance(observations)
    assert np.allclose(result, baseline, atol=1e-8)


def test_apply_overlay_with_oas_shrinker_matches_baseline() -> None:
    rng = np.random.default_rng(81)
    observations = rng.normal(scale=0.3, size=(85, 5))
    sample_cov = np.cov(observations, rowvar=False, ddof=1)
    config = OverlayConfig(shrinker="oas", require_isolated=False, q_max=1)
    result = apply_overlay(sample_cov, [], observations=observations, config=config)
    baseline = oas_covariance(observations)
    assert np.allclose(result, baseline, atol=1e-8)


def test_apply_overlay_with_cc_shrinker_matches_baseline() -> None:
    rng = np.random.default_rng(91)
    observations = rng.normal(scale=0.25, size=(75, 4))
    sample_cov = np.cov(observations, rowvar=False, ddof=1)
    config = OverlayConfig(shrinker="cc", require_isolated=False, q_max=1)
    result = apply_overlay(sample_cov, [], observations=observations, config=config)
    baseline = cc_covariance(observations)
    assert np.allclose(result, baseline, atol=1e-8)


def test_detect_spikes_strict_gate_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    det_bad = _make_detection(
        3.0,
        np.array([1.0, 0.0, 0.0]),
        edge_margin=0.25,
        stability=0.05,
        alignment=0.8,
        pre_outlier=1,
    )
    det_good = _make_detection(
        2.5,
        np.array([0.0, 1.0, 0.0]),
        edge_margin=0.4,
        stability=0.4,
        alignment=0.95,
        pre_outlier=1,
    )

    def fake_search(*args, **kwargs):
        return [det_bad, det_good]

    monkeypatch.setattr("fjs.overlay.dealias_search", fake_search)
    cfg = OverlayConfig(
        gate_mode="strict",
        gate_stability_min=0.2,
        gate_alignment_min=0.9,
        min_edge_margin=0.2,
        q_max=5,
    )
    obs = np.ones((10, 3))
    groups = np.repeat(np.arange(5), 2)
    stats: dict = {}
    kept = detect_spikes(obs, groups, config=cfg, stats=stats)
    assert len(kept) == 1
    assert kept[0]["mu_hat"] == pytest.approx(2.5)
    assert stats["gating"]["accepted"] == 1
    assert stats["gating"]["rejected"] == 1
    assert stats["gating"]["delta_frac_used"] is None


def test_detect_spikes_soft_gate_selects_top_score(monkeypatch: pytest.MonkeyPatch) -> None:
    det_low = _make_detection(
        2.0,
        np.array([1.0, 0.0, 0.0]),
        edge_margin=0.3,
        stability=0.2,
        target_energy=0.4,
        alignment=0.95,
        pre_outlier=1,
    )
    det_high = _make_detection(
        2.2,
        np.array([0.0, 1.0, 0.0]),
        edge_margin=0.35,
        stability=0.5,
        target_energy=1.5,
        alignment=0.98,
        pre_outlier=1,
    )

    def fake_search(*args, **kwargs):
        return [det_low, det_high]

    monkeypatch.setattr("fjs.overlay.dealias_search", fake_search)
    cfg = OverlayConfig(
        gate_mode="soft",
        gate_soft_max=1,
        min_edge_margin=0.2,
        gate_accept_nonisolated=True,
        q_max=3,
    )
    obs = np.ones((12, 3))
    groups = np.repeat(np.arange(6), 2)
    kept = detect_spikes(obs, groups, config=cfg)
    assert len(kept) == 1
    assert kept[0]["mu_hat"] == pytest.approx(2.2)


def test_detect_spikes_uses_calibrated_delta(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    det = _make_detection(
        3.0,
        np.array([1.0, 0.0, 0.0]),
        edge_margin=0.5,
        stability=0.6,
        alignment=0.97,
        pre_outlier=1,
    )

    def fake_search(*args, **kwargs):
        return [det]

    monkeypatch.setattr("fjs.overlay.dealias_search", fake_search)

    calib_path = tmp_path / "thresholds.json"
    payload = {
        "thresholds": {
            "tyler": {
                "3x5": {
                    "delta_frac": 0.03,
                }
            }
        }
    }
    calib_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = OverlayConfig(
        gate_mode="strict",
        gate_delta_calibration=str(calib_path),
        gate_delta_frac_min=0.02,
        gate_delta_frac_max=0.04,
        min_edge_margin=0.2,
        q_max=2,
    )
    obs = np.ones((10, 3))
    groups = np.repeat(np.arange(5), 2)
    stats: dict = {}
    kept = detect_spikes(obs, groups, config=cfg, stats=stats)
    assert kept
    assert kept[0]["delta_frac"] == pytest.approx(0.03)
    assert stats["gating"]["delta_frac_used"] == pytest.approx(0.03)


def test_detect_spikes_rejects_when_calibrated_delta_below_min(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    det = _make_detection(
        2.8,
        np.array([1.0, 0.0, 0.0]),
        edge_margin=0.4,
        stability=0.4,
        alignment=0.95,
        pre_outlier=1,
    )

    def fake_search(*args, **kwargs):
        return [det]

    monkeypatch.setattr("fjs.overlay.dealias_search", fake_search)

    calib_path = tmp_path / "thresholds.json"
    payload = {
        "thresholds": {
            "tyler": {
                "3x4": {
                    "delta_frac": 0.015,
                }
            }
        }
    }
    calib_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = OverlayConfig(
        gate_mode="strict",
        gate_delta_calibration=str(calib_path),
        gate_delta_frac_min=0.02,
        min_edge_margin=0.2,
        q_max=2,
    )
    obs = np.ones((8, 3))
    groups = np.repeat(np.arange(4), 2)
    stats: dict = {}
    kept = detect_spikes(obs, groups, config=cfg, stats=stats)
    assert kept == []
    assert stats["gating"]["accepted"] == 0
    assert stats["gating"]["rejected"] == 1
    assert stats["gating"]["delta_frac_used"] == pytest.approx(0.015)
