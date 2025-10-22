from __future__ import annotations

import numpy as np
import pytest

from fjs.dealias import DealiasingResult, dealias_covariance, dealias_search


def _simulate_one_way(
    rng: np.random.Generator,
    *,
    p: int,
    n_groups: int,
    replicates: int,
    mu_sigma1: float,
    mu_sigma2: float,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    group_labels = np.repeat(np.arange(n_groups), replicates)
    v1 = rng.normal(size=p)
    v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=p)
    v2 /= np.linalg.norm(v2)

    data = np.zeros((n_groups * replicates, p), dtype=np.float64)
    idx = 0
    for _ in range(n_groups):
        group_effect = np.zeros(p, dtype=np.float64)
        if mu_sigma1 > 0.0:
            group_effect += np.sqrt(mu_sigma1) * rng.normal() * v1
        for _ in range(replicates):
            noise = noise_scale * rng.normal(size=p)
            if mu_sigma2 > 0.0:
                noise += noise_scale * np.sqrt(mu_sigma2) * rng.normal() * v2
            data[idx] = group_effect + noise
            idx += 1
    return data, group_labels


def test_dealiasing_result_structure() -> None:
    fields = getattr(DealiasingResult, "__dataclass_fields__", None)
    assert fields is not None


def test_dealias_covariance_is_stub() -> None:
    covariance = np.eye(3)
    spectrum = np.ones(3)
    with pytest.raises(NotImplementedError):
        dealias_covariance(covariance, spectrum)


def test_dealias_search_detects_sigma1_spike() -> None:
    rng = np.random.default_rng(0)
    p, n_groups, replicates = 60, 60, 2
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=6.0,
        mu_sigma2=0.0,
        noise_scale=0.5,
    )
    detections = dealias_search(
        y, groups, target_r=0, a_grid=120, delta=0.3, eps=0.05
    )
    assert detections, "Expected a detection for Sigma1 spike."
    assert len(detections) == 1
    lambda_est = detections[0]["lambda_hat"]
    assert abs(lambda_est - 6.0) < 0.8


def test_dealias_search_limits_sigma2_false_positives() -> None:
    p, n_groups, replicates = 60, 60, 2
    trials = 20
    false_positives = 0
    for seed in range(trials):
        trial_rng = np.random.default_rng(seed)
        y, groups = _simulate_one_way(
            trial_rng,
            p=p,
            n_groups=n_groups,
            replicates=replicates,
            mu_sigma1=0.0,
            mu_sigma2=8.0,
        )
        detections = dealias_search(y, groups, target_r=0, a_grid=90, delta=0.5)
        if detections:
            false_positives += 1
    assert false_positives <= max(2, int(0.1 * trials))
