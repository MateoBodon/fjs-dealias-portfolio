from __future__ import annotations

import os

import numpy as np
import pytest

from src.fjs.config import get_detection_settings
from src.fjs.dealias import dealias_search
from tests.test_dealias import _simulate_one_way


@pytest.mark.slow
def test_dealias_null_false_positive_rate_calibrated() -> None:
    settings = get_detection_settings()
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    trials = 40 if fast else 90
    rng = np.random.default_rng(20251102)
    false_positives = 0
    for seed in rng.integers(0, 10_000, size=trials):
        sim_rng = np.random.default_rng(int(seed))
        y, groups = _simulate_one_way(
            sim_rng,
            p=48,
            n_groups=48,
            replicates=2,
            mu_sigma1=0.0,
            mu_sigma2=8.0,
        )
        detections = dealias_search(
            y,
            groups,
            target_r=0,
            a_grid=None,
            delta=None,
            delta_frac=None,
            settings=settings,
        )
        if detections:
            false_positives += 1
    fpr = false_positives / float(trials)
    assert fpr <= 0.02


@pytest.mark.slow
def test_dealias_power_mu6_calibrated() -> None:
    settings = get_detection_settings()
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    trials = 30 if fast else 60
    rng = np.random.default_rng(20251103)
    detections = 0
    for seed in rng.integers(0, 20_000, size=trials):
        sim_rng = np.random.default_rng(int(seed))
        y, groups = _simulate_one_way(
            sim_rng,
            p=60,
            n_groups=60,
            replicates=2,
            mu_sigma1=6.0,
            mu_sigma2=0.0,
        )
        result = dealias_search(
            y,
            groups,
            target_r=0,
            a_grid=None,
            delta=None,
            delta_frac=None,
            settings=settings,
        )
        if result:
            detections += 1
    power = detections / float(trials)
    assert power >= 0.8
