from __future__ import annotations

import numpy as np
import pytest

from fjs.balanced import mean_squares
from fjs.dealias import (
    DealiasingResult,
    _default_design,
    dealias_covariance,
    dealias_search,
)
from fjs.mp import t_vec


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
    detections = dealias_search(y, groups, target_r=0, a_grid=120, delta=0.3, eps=0.05)
    assert detections, "Expected a detection for Sigma1 spike."
    assert len(detections) == 1
    lambda_est = detections[0]["lambda_hat"]
    assert abs(lambda_est - 6.0) < 0.8
    assert detections[0]["stability_margin"] >= 0.0


def test_t_vector_acceptance_consistency_toy_spike() -> None:
    rng = np.random.default_rng(314159)
    p, n_groups, replicates = 24, 48, 3
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=6.0,
        mu_sigma2=0.0,
        noise_scale=0.6,
    )
    detections = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=100,
        delta=0.35,
        eps=0.04,
    )
    assert detections, "Expected at least one detection in the toy spike setting."

    stats = mean_squares(y, groups)
    design_params = _default_design(stats)
    c_weights = np.asarray(design_params["C"], dtype=np.float64).tolist()
    d_vec = np.asarray(design_params["d"], dtype=np.float64).tolist()
    n_total = float(design_params["N"])
    c_vec = np.asarray(design_params["c"], dtype=np.float64).tolist()
    order = design_params["order"]

    for det in detections:
        lam_val = float(det["lambda_hat"])
        mu_hat = float(det["mu_hat"])
        assert mu_hat > 0.0
        t_vals = t_vec(
            lam_val,
            det["a"],
            c_weights,
            d_vec,
            n_total,
            c_vec,
            order,
        )
        ratio = lam_val / mu_hat
        t_target = float(t_vals[0])
        assert ratio > 0.0
        assert np.isfinite(t_target)
        assert abs(ratio - t_target) < 0.1


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


def test_dealias_search_has_low_false_positive_rate() -> None:
    p, n_groups, replicates = 32, 48, 2
    trials = 100
    rng = np.random.default_rng(123)
    false_positives = 0
    for seed in rng.integers(0, 10_000, size=trials):
        trial_rng = np.random.default_rng(int(seed))
        y, groups = _simulate_one_way(
            trial_rng,
            p=p,
            n_groups=n_groups,
            replicates=replicates,
            mu_sigma1=0.0,
            mu_sigma2=4.0,
        )
        detections = dealias_search(
            y,
            groups,
            target_r=0,
            a_grid=90,
            delta=0.5,
            eps=0.05,
        )
        if detections:
            false_positives += 1
    assert false_positives <= max(1, int(0.01 * trials))


def test_dealias_search_isotropic_trials_under_one_percent() -> None:
    rng = np.random.default_rng(2024)
    p, n_groups, replicates = 12, 24, 2
    trials = 200
    false_positives = 0
    for _ in range(trials):
        y, groups = _simulate_one_way(
            rng,
            p=p,
            n_groups=n_groups,
            replicates=replicates,
            mu_sigma1=0.0,
            mu_sigma2=0.0,
            noise_scale=1.0,
        )
        detections = dealias_search(y, groups, target_r=0)
        if detections:
            false_positives += 1
    assert false_positives <= max(1, int(0.01 * trials))


def test_dealias_search_stability_consistent_across_eta() -> None:
    rng = np.random.default_rng(99)
    p, n_groups, replicates = 48, 48, 2
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=5.5,
        mu_sigma2=0.0,
        noise_scale=0.6,
    )
    base = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=120,
        delta=0.3,
        eps=0.05,
        stability_eta_deg=0.4,
    )
    assert base, "Expected detection at baseline eta."
    wider = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=120,
        delta=0.3,
        eps=0.05,
        stability_eta_deg=1.0,
    )
    assert wider, "Expected detection at wider eta."
    assert len(base) == len(wider)
    base_mu = sorted(det["mu_hat"] for det in base)
    wider_mu = sorted(det["mu_hat"] for det in wider)
    np.testing.assert_allclose(base_mu, wider_mu, rtol=0.0, atol=1e-6)
    assert all(det["stability_margin"] >= 0.0 for det in wider)
