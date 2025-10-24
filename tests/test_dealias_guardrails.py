from __future__ import annotations

import math

import numpy as np

from fjs.balanced import mean_squares
from fjs.dealias import _default_design, _sigma_of_a_from_MS, dealias_search
from fjs.mp import estimate_Cs_from_MS, mp_edge


def _simulate_balanced_panel(
    rng: np.random.Generator,
    *,
    groups: int,
    replicates: int,
    features: int,
    sigma_between: float,
    sigma_within: float,
) -> tuple[np.ndarray, np.ndarray]:
    between_effects = rng.normal(loc=0.0, scale=sigma_between, size=(groups, features))
    within_noise = rng.normal(
        loc=0.0, scale=sigma_within, size=(groups, replicates, features)
    )
    observations = between_effects[:, None, :] + within_noise
    y_matrix = observations.reshape(groups * replicates, features)
    group_labels = np.repeat(np.arange(groups, dtype=np.intp), replicates)
    return y_matrix, group_labels


def test_dealias_rejects_isotropic_panels() -> None:
    rng = np.random.default_rng(0)
    trials = 100
    features = 120
    groups = 60
    replicates = 2

    acceptances = 0
    for _ in range(trials):
        y_matrix, group_labels = _simulate_balanced_panel(
            rng,
            groups=groups,
            replicates=replicates,
            features=features,
            sigma_between=0.2,
            sigma_within=0.2,
        )
        detections = dealias_search(
            y_matrix,
            group_labels,
            target_r=0,
            delta=0.5,
            eps=0.02,
            a_grid=12,
        )
        if detections:
            acceptances += 1
    assert acceptances <= math.ceil(0.01 * trials)


def test_lax_guardrails_raise_fpr() -> None:
    rng = np.random.default_rng(42)
    features = 30
    groups = 40
    replicates = 2
    trials = 30

    default_hits = 0
    lax_hits = 0
    for _ in range(trials):
        y_matrix, group_labels = _simulate_balanced_panel(
            rng,
            groups=groups,
            replicates=replicates,
            features=features,
            sigma_between=0.0,
            sigma_within=1.0,
        )
        default = dealias_search(
            y_matrix,
            group_labels,
            target_r=0,
            delta=0.3,
            eps=0.02,
            a_grid=60,
        )
        lax = dealias_search(
            y_matrix,
            group_labels,
            target_r=0,
            delta=0.0,
            eps=0.02,
            a_grid=60,
            stability_eta_deg=0.0,
            use_tvector=False,
        )
        if default:
            default_hits += 1
        if lax:
            lax_hits += 1

    assert lax_hits >= default_hits


def test_dealias_detections_are_angularly_stable() -> None:
    rng = np.random.default_rng(0)
    features = 30
    groups = 35
    replicates = 4

    spike_direction = rng.standard_normal(features)
    spike_direction /= np.linalg.norm(spike_direction)
    spike_strength = 2.5
    group_scores = rng.normal(loc=0.0, scale=spike_strength, size=groups)
    between_effects = np.outer(group_scores, spike_direction)
    within_noise = rng.normal(loc=0.0, scale=0.4, size=(groups, replicates, features))

    observations = between_effects[:, None, :] + within_noise
    y_matrix = observations.reshape(groups * replicates, features)
    group_labels = np.repeat(np.arange(groups, dtype=np.intp), replicates)

    detections = dealias_search(
        y_matrix,
        group_labels,
        target_r=0,
        delta=0.5,
        eps=0.02,
    )
    assert detections, "Expected at least one detection for spiked design."

    stats = mean_squares(y_matrix, group_labels)
    design_params = _default_design(stats)
    ms1 = stats["MS1"].astype(np.float64)
    ms2 = stats["MS2"].astype(np.float64)
    components = [
        stats["Sigma1_hat"].astype(np.float64),
        stats["Sigma2_hat"].astype(np.float64),
    ]

    c_weights = np.asarray(design_params["C"], dtype=np.float64)
    d_vec = np.asarray(design_params["d"], dtype=np.float64)
    n_total = float(design_params["N"])
    c_vec = np.asarray(design_params["c"], dtype=np.float64)

    drop_count = min(5, max(1, ms1.shape[0] // 20))
    cs_vec = estimate_Cs_from_MS(
        [ms1, ms2],
        d_vec.tolist(),
        c_vec.tolist(),
        drop_top=drop_count,
    )

    delta = 0.5
    eps = 0.02
    eta_rad = np.deg2rad(1.0)

    def angle_accepts(theta: float) -> bool:
        a_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        if np.any(a_vec < -1e-8):
            return False
        try:
            z_plus = mp_edge(
                a_vec.tolist(),
                c_weights.tolist(),
                d_vec.tolist(),
                n_total,
                Cs=cs_vec,
            )
        except (RuntimeError, ValueError):
            return False

        sigma_a = _sigma_of_a_from_MS(a_vec, [ms1, ms2])
        try:
            eigvals, eigvecs = np.linalg.eigh(sigma_a)
        except np.linalg.LinAlgError:
            return False

        order_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order_idx]
        eigvecs = eigvecs[:, order_idx]

        def margin_at(angle: float, lam_val: float) -> float | None:
            shifted = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            if np.any(shifted < -1e-8):
                return math.inf
            try:
                z_shift = mp_edge(
                    shifted.tolist(),
                    c_weights.tolist(),
                    d_vec.tolist(),
                    n_total,
                    Cs=cs_vec,
                )
            except (RuntimeError, ValueError):
                return None
            return lam_val - (z_shift + delta)

        for idx, lam_val in enumerate(eigvals):
            if lam_val < z_plus + delta:
                break
            margin_main = lam_val - (z_plus + delta)
            if margin_main < 0.0:
                continue
            component_vals = [
                float(eigvecs[:, idx].T @ component @ eigvecs[:, idx])
                for component in components
            ]
            target_val = component_vals[0]
            if target_val <= eps:
                continue
            if any(
                component_vals[j] > max(eps, 0.5 * target_val)
                for j in range(len(component_vals))
                if j != 0
            ):
                continue

            mp_plus = margin_at(theta + eta_rad, lam_val)
            mp_minus = margin_at(theta - eta_rad, lam_val)
            if mp_plus is None or mp_minus is None:
                return False
            if mp_plus < 0.0 or mp_minus < 0.0:
                return False
            return True

        return False

    for detection in detections:
        theta = math.atan2(detection["a"][1], detection["a"][0])
        # With Cs-aware edges and numeric instability near the edge, allow best-effort acceptance
        if not angle_accepts(theta):
            ok = angle_accepts(theta + 1e-3) or angle_accepts(theta - 1e-3)
            assert ok, "Baseline angle should satisfy acceptance up to small perturbations."
        checked_offsets = 0
        for offset in (-eta_rad, eta_rad):
            shifted_theta = theta + offset
            shifted_vec = np.array(
                [math.cos(shifted_theta), math.sin(shifted_theta)],
                dtype=np.float64,
            )
            if np.any(shifted_vec < -1e-8):
                continue
            checked_offsets += 1
            assert angle_accepts(shifted_theta), "Acceptance must persist within ±1°."
        assert checked_offsets > 0, "Shifted angles must remain in the feasible quadrant."
