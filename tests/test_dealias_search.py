from __future__ import annotations

import math

import numpy as np

from fjs.balanced import mean_squares
from fjs.dealias import _default_design, dealias_search
from fjs.mp import estimate_Cs_from_MS, mp_edge
from finance.io import load_returns_csv
from experiments.equity_panel.run import _balanced_weekly_panel


def _make_single_spike_panel(
    rng: np.random.Generator,
    *,
    groups: int,
    replicates: int,
    features: int,
    mu: float,
    sigma_within: float,
) -> tuple[np.ndarray, np.ndarray]:
    v = rng.normal(size=features)
    v /= np.linalg.norm(v)
    group_scores = rng.normal(scale=np.sqrt(mu), size=groups)
    between = np.outer(group_scores, v)
    residuals = rng.normal(scale=sigma_within, size=(groups, replicates, features))
    observations = between[:, None, :] + residuals
    y = observations.reshape(groups * replicates, features)
    labels = np.repeat(np.arange(groups, dtype=np.intp), replicates)
    return y, labels


def test_single_spike_mu_hat_within_standard_error() -> None:
    rng = np.random.default_rng(123)
    p, groups, replicates = 10, 60, 3
    mu_true = 4.0
    y, labels = _make_single_spike_panel(
        rng,
        groups=groups,
        replicates=replicates,
        features=p,
        mu=mu_true,
        sigma_within=0.3,
    )
    detections = dealias_search(
        y,
        labels,
        target_r=0,
        a_grid=60,
        delta=0.3,
        eps=0.03,
        stability_eta_deg=0.4,
    )
    assert detections, "Expected a detection in the single-spike setting."
    mu_hat = float(detections[0]["mu_hat"])
    # Approximate SE of sample variance under Normal(0, mu): sqrt(2/(G-1)) * mu
    se = mu_true * math.sqrt(2.0 / max(groups - 1, 1))
    assert abs(mu_hat - mu_true) <= 2.5 * se


def test_permuted_group_labels_yield_no_detections() -> None:
    rng = np.random.default_rng(321)
    p, groups, replicates = 12, 48, 3
    y, labels = _make_single_spike_panel(
        rng,
        groups=groups,
        replicates=replicates,
        features=p,
        mu=5.0,
        sigma_within=0.35,
    )
    # Shuffle labels to break the group structure while preserving balance
    permuted = rng.permutation(labels)
    detections = dealias_search(
        y,
        permuted,
        target_r=0,
        a_grid=72,
        delta=0.4,
        eps=0.03,
        stability_eta_deg=0.4,
        off_component_leak_cap=5.0,
        energy_min_abs=1e-6,
    )
    assert not detections


def test_decisions_stable_within_eta_band() -> None:
    rng = np.random.default_rng(7)
    p, groups, replicates = 20, 40, 3
    y, labels = _make_single_spike_panel(
        rng,
        groups=groups,
        replicates=replicates,
        features=p,
        mu=5.5,
        sigma_within=0.25,
    )
    eta = 0.6  # degrees
    detections = dealias_search(
        y,
        labels,
        target_r=0,
        a_grid=90,
        delta=0.35,
        eps=0.03,
        stability_eta_deg=eta,
    )
    assert detections, "Expected at least one detection."

    # Verify acceptance persists under ±eta angular perturbations
    stats = mean_squares(y, labels)
    design = _default_design(stats)
    ms1 = stats["MS1"].astype(np.float64)
    ms2 = stats["MS2"].astype(np.float64)
    c_weights = np.asarray(design["C"], dtype=np.float64)
    d_vec = np.asarray(design["d"], dtype=np.float64)
    n_total = float(design["N"])
    c_vec = np.asarray(design["c"], dtype=np.float64)

    drop_count = min(5, max(1, p // 20))
    cs_vec = estimate_Cs_from_MS([ms1, ms2], d_vec.tolist(), c_vec.tolist(), drop_top=drop_count)

    eta_rad = float(np.deg2rad(eta))

    def margin_at(theta: float, lam_val: float) -> float | None:
        a_vec = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        try:
            z = mp_edge(a_vec.tolist(), c_weights.tolist(), d_vec.tolist(), n_total, Cs=cs_vec)
        except (RuntimeError, ValueError):
            return None
        thr = z + max(0.35, 0.0 * z)
        return float(lam_val - thr)

    for det in detections:
        theta0 = math.atan2(float(det["a"][1]), float(det["a"][0]))
        lam = float(det["lambda_hat"])
        base_m = margin_at(theta0, lam)
        if base_m is None or base_m < 0.0:
            continue
        for off in (-eta_rad, eta_rad):
            m = margin_at(theta0 + off, lam)
            assert m is not None and m >= 0.0


def test_wrds_window_yields_detection_with_relaxed_leakage() -> None:
    returns = load_returns_csv("data/returns_daily.csv")
    mask = (returns.index >= "2015-01-01") & (returns.index <= "2024-12-31")
    panel = _balanced_weekly_panel(returns.loc[mask])
    weekly_df = panel.weekly
    week_map = panel.week_map
    replicates = panel.replicates
    # Use the first 156 balanced weeks to mirror the production rolling window
    fit = weekly_df.iloc[:156]
    fit_blocks = [week_map[idx] for idx in fit.index]
    ordered = sorted(set.intersection(*[set(df.columns) for df in fit_blocks]))
    fit_blocks = [df.loc[:, ordered].to_numpy(dtype=np.float64) for df in fit_blocks]
    y_fit_daily = np.vstack(fit_blocks)
    groups_fit = np.repeat(np.arange(len(fit_blocks)), int(replicates))
    detections = dealias_search(
        y_fit_daily,
        groups_fit,
        target_r=0,
        delta=0.0,
        delta_frac=0.02,
        eps=0.03,
        stability_eta_deg=0.4,
        use_tvector=True,
        nonnegative_a=False,
        a_grid=72,
        cs_drop_top_frac=0.05,
        scan_basis="sigma",
        off_component_leak_cap=None,
        energy_min_abs=1e-6,
    )
    assert detections, "Expected at least one detection on the WRDS weekly panel."
    ratios = [
        float(det.get("off_component_ratio", 0.0)) for det in detections
    ]
    assert any(r > 1.0 for r in ratios if np.isfinite(r))
