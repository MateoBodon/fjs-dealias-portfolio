from __future__ import annotations

import numpy as np
import pytest

from src.fjs.overlay import OverlayConfig, apply_overlay, detect_spikes


def _spiked_dataset(
    *,
    n: int = 800,
    p: int = 10,
    strength: float = 4.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, p))
    direction = rng.normal(size=p)
    direction /= np.linalg.norm(direction)
    factor = rng.normal(size=(n, 1))
    samples = noise + strength * factor @ direction[np.newaxis, :]
    covariance = np.cov(samples, rowvar=False, ddof=1)
    return samples, covariance


def test_detect_spikes_identifies_strong_direction() -> None:
    samples, covariance = _spiked_dataset()
    cfg = OverlayConfig(min_margin=0.05, min_isolation=0.02, max_detections=1, shrinkage=0.1)
    result = detect_spikes(covariance, samples=samples, config=cfg)

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.index == 0
    assert detection.margin > cfg.min_margin

    corrected = apply_overlay(covariance, result, config=cfg)
    orig_eigs = np.linalg.eigvalsh(covariance)
    corrected_eigs = np.linalg.eigvalsh(corrected)
    assert corrected_eigs[-1] < orig_eigs[-1]
    direction = result.detections[0].direction
    direction = direction / np.linalg.norm(direction)
    updated_value = float(direction @ corrected @ direction)
    assert updated_value == pytest.approx(result.detections[0].replacement, rel=0.1)
    assert np.allclose(corrected, corrected.T, atol=1e-10)


def test_detect_spikes_respects_null_case() -> None:
    rng = np.random.default_rng(321)
    samples = rng.normal(size=(600, 8))
    covariance = np.cov(samples, rowvar=False, ddof=1)
    cfg = OverlayConfig(min_margin=0.2, min_isolation=0.1, max_detections=2, shrinkage=0.0)
    result = detect_spikes(covariance, samples=samples, config=cfg)

    assert len(result.detections) == 0
    corrected = apply_overlay(covariance, result, config=cfg)
    assert np.allclose(corrected, covariance, atol=1e-6, rtol=1e-6)
