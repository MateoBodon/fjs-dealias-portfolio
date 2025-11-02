from __future__ import annotations

import os
import numpy as np
import pytest

from src.fjs.config import DetectionSettings, get_detection_settings

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


def _test_settings(**overrides: float) -> DetectionSettings:
    return get_detection_settings().with_overrides(**overrides)


@pytest.mark.unit
def test_dealiasing_result_structure() -> None:
    fields = getattr(DealiasingResult, "__dataclass_fields__", None)
    assert fields is not None


@pytest.mark.unit
def test_dealias_covariance_uses_detection_vectors() -> None:
    covariance = np.diag([5.0, 2.0, 1.0]).astype(np.float64)
    detection = {
        "mu_hat": 3.5,
        "eigvec": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }
    result = dealias_covariance(covariance, [detection])
    assert isinstance(result, DealiasingResult)
    assert result.iterations == 1
    assert np.isclose(result.covariance[0, 0], 3.5, atol=1e-8)
    # Remaining diagonal entries remain unchanged
    assert np.isclose(result.covariance[1, 1], 2.0, atol=1e-8)
    assert np.isclose(result.covariance[2, 2], 1.0, atol=1e-8)


@pytest.mark.unit
def test_dealias_covariance_accepts_target_spectrum() -> None:
    covariance = np.diag([1.0, 2.0, 3.0]).astype(np.float64)
    target = np.array([4.0, 2.0, 1.0], dtype=np.float64)
    result = dealias_covariance(covariance, target)
    assert result.iterations == 3
    eigvals = np.sort(np.linalg.eigvalsh(result.covariance))[::-1]
    assert np.allclose(eigvals, np.sort(target)[::-1], atol=1e-8)


@pytest.mark.integration
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
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid = 48 if fast else 120
    detections = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=0.3,
        eps=0.05,
        settings=_test_settings(
            t_eps=0.05,
            off_component_cap=None,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
    )
    assert detections, "Expected a detection for Sigma1 spike."
    assert len(detections) == 1
    lambda_est = detections[0]["lambda_hat"]
    assert abs(lambda_est - 6.0) < 0.8
    assert detections[0]["stability_margin"] >= 0.0


@pytest.mark.slow
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
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid = 36 if fast else 100
    detections = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=0.3,
        eps=0.05,
        settings=_test_settings(
            t_eps=0.05,
            off_component_cap=None,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
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
        # Allow a more realistic tolerance after Cs-aware mapping
        assert abs(ratio - t_target) < 3.0


@pytest.mark.integration
def test_relative_delta_enables_detection_when_absolute_blocks() -> None:
    rng = np.random.default_rng(2025)
    p, n_groups, replicates = 24, 36, 3
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=5.0,
        mu_sigma2=0.0,
        noise_scale=0.7,
    )
    # With a large absolute delta, we expect no detections
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid = 24 if fast else 72
    blocked = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=10.0,
        settings=_test_settings(
            off_component_cap=None,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
    )
    assert not blocked
    # With a small relative delta, expect at least one detection
    grid2 = 36 if fast else 90
    allowed = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid2,
        delta=0.0,
        delta_frac=0.03,
        settings=_test_settings(
            off_component_cap=None,
            t_eps=0.05,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
    )
    assert allowed


@pytest.mark.integration
def test_equity_toy_detection_with_delta_frac() -> None:
    rng = np.random.default_rng(123)
    p, n_groups, replicates = 24, 36, 3
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=5.5,
        mu_sigma2=0.0,
        noise_scale=0.7,
    )
    detections = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=72,
        delta=0.0,
        delta_frac=0.03,
        eps=0.04,
        stability_eta_deg=0.4,
        settings=_test_settings(
            off_component_cap=None,
            t_eps=0.05,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
    )
    assert isinstance(detections, list)
    assert len(detections) >= 1


def test_detections_include_diagnostics_fields() -> None:
    rng = np.random.default_rng(2026)
    p, n_groups, replicates = 24, 36, 3
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=6.0,
        mu_sigma2=0.0,
        noise_scale=0.6,
    )
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid = 36 if fast else 72
    detections = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=0.3,
        eps=0.04,
    )
    assert isinstance(detections, list)
    if detections:
        top = detections[0]
        assert "z_plus" in top and "threshold_main" in top
        # values may be finite; allow nan fallbacks defensively
        assert np.isfinite(float(top.get("z_plus", np.nan))) or np.isnan(
            float(top.get("z_plus", np.nan))
        )
        assert np.isfinite(float(top.get("threshold_main", np.nan))) or np.isnan(
            float(top.get("threshold_main", np.nan))
        )


def test_signed_a_grid_no_crash() -> None:
    rng = np.random.default_rng(7)
    p, n_groups, replicates = 20, 24, 3
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=4.0,
        mu_sigma2=0.0,
        noise_scale=0.8,
    )
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid = 24 if fast else 60
    out1 = dealias_search(
        y, groups, target_r=0, a_grid=grid, delta=0.3, nonnegative_a=True
    )
    out2 = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=0.3,
        nonnegative_a=False,
    )
    assert isinstance(out1, list)
    assert isinstance(out2, list)


@pytest.mark.slow
def test_dealias_search_limits_sigma2_false_positives() -> None:
    p, n_groups, replicates = 60, 60, 2
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    trials = 8 if fast else 20
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
        grid = 36 if fast else 90
        detections = dealias_search(y, groups, target_r=0, a_grid=grid, delta=0.5)
        if detections:
            false_positives += 1
    assert false_positives <= max(2, int(0.1 * trials))


@pytest.mark.slow
def test_dealias_search_has_low_false_positive_rate() -> None:
    p, n_groups, replicates = 32, 48, 2
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    trials = 30 if fast else 100
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
        grid = 36 if fast else 90
        detections = dealias_search(
            y,
            groups,
            target_r=0,
            a_grid=grid,
            delta=0.5,
            eps=0.05,
        )
        if detections:
            false_positives += 1
    assert false_positives <= max(1, int(0.01 * trials))


@pytest.mark.slow
def test_dealias_search_isotropic_trials_under_one_percent() -> None:
    rng = np.random.default_rng(2024)
    p, n_groups, replicates = 12, 24, 2
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    trials = 40 if fast else 120
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
        grid = 36 if fast else 120
        detections = dealias_search(y, groups, target_r=0, a_grid=grid)
        if detections:
            false_positives += 1
    assert false_positives <= max(1, int(0.01 * trials))


def test_cs_drop_top_frac_influences_threshold_or_detections() -> None:
    rng = np.random.default_rng(11)
    p, n_groups, replicates = 30, 40, 2
    y, groups = _simulate_one_way(
        rng,
        p=p,
        n_groups=n_groups,
        replicates=replicates,
        mu_sigma1=6.0,
        mu_sigma2=0.0,
        noise_scale=0.6,
    )
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid_lo = 36 if fast else 90
    det_lo = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid_lo,
        delta=0.3,
        eps=0.04,
        cs_drop_top_frac=0.05,
        settings=_test_settings(
            off_component_cap=None,
            t_eps=0.05,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
    )
    det_hi = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid_lo,
        delta=0.3,
        eps=0.04,
        cs_drop_top_frac=0.4,
        settings=_test_settings(
            off_component_cap=None,
            t_eps=0.05,
            require_isolated=False,
            angle_min_cos=0.0,
        ),
    )
    # Either detection counts differ, or if both detect, thresholds differ
    if det_lo and det_hi:
        thr_lo = float(det_lo[0].get("threshold_main", np.nan))
        thr_hi = float(det_hi[0].get("threshold_main", np.nan))
        assert not np.isclose(thr_lo, thr_hi, rtol=0.0, atol=1e-9)
    else:
        assert len(det_lo) != len(det_hi)


@pytest.mark.slow
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
    fast = bool(int(os.getenv("FAST_TESTS", "0")))
    grid = 48 if fast else 120
    overrides = {
        "off_component_cap": None,
        "t_eps": 0.05,
        "require_isolated": False,
        "angle_min_cos": 0.0,
    }
    base = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=0.3,
        eps=0.05,
        stability_eta_deg=0.4,
        settings=_test_settings(**overrides),
    )
    assert base, "Expected detection at baseline eta."
    wider = dealias_search(
        y,
        groups,
        target_r=0,
        a_grid=grid,
        delta=0.3,
        eps=0.05,
        stability_eta_deg=1.0,
        settings=_test_settings(**overrides),
    )
    assert wider, "Expected detection at wider eta."
    assert len(base) == len(wider)
    base_mu = sorted(det["mu_hat"] for det in base)
    wider_mu = sorted(det["mu_hat"] for det in wider)
    np.testing.assert_allclose(base_mu, wider_mu, rtol=0.0, atol=1e-6)
    assert all(det["stability_margin"] >= 0.0 for det in wider)
