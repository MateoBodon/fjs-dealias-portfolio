from __future__ import annotations

import numpy as np

from experiments.synthetic.nested_killtest import (
    _edge_scale,
    mean_squares_nested,
    simulate_nested_panel,
)
from fjs.dealias import dealias_search
from fjs.gating import count_isolated_outliers


def _run_nested_trial(spike_strength: float) -> list[dict]:
    rng = np.random.default_rng(123)
    observations, year_labels, week_labels = simulate_nested_panel(
        rng,
        n_assets=30,
        years=2,
        weeks=4,
        replicates=3,
        spike_strength=spike_strength,
        signal_to_noise=0.35,
        noise_variance=1.0,
    )

    (ms1, ms2, ms3), meta = mean_squares_nested(
        observations, year_labels, week_labels, replicates=3
    )

    sigma1 = ((ms1 - ms2) / float(meta.J * meta.replicates)).astype(np.float64)
    sigma2 = ((ms2 - ms3) / float(meta.replicates)).astype(np.float64)
    sigma3 = ms3.astype(np.float64)

    stats_local = {
        "MS1": ms1.astype(np.float64),
        "MS2": ms2.astype(np.float64),
        "MS3": ms3.astype(np.float64),
        "Sigma1_hat": sigma1,
        "Sigma2_hat": sigma2,
        "Sigma3_hat": sigma3,
        "I": meta.I,
        "J": meta.J,
        "n": meta.n,
        "replicates": meta.replicates,
    }

    design_override = {
        "c": meta.c.astype(np.float64),
        "C": np.ones_like(meta.c, dtype=np.float64),
        "d": meta.d.astype(np.float64),
        "N": float(meta.N),
        "order": [[1, 2, 3], [2, 3], [3]],
    }

    edge_scale, _, _ = _edge_scale(
        observations,
        edge_mode="tyler",
        edge_huber_c=1.5,
    )

    detections = dealias_search(
        observations,
        np.arange(observations.shape[0], dtype=np.intp),
        target_r=0,
        delta=0.35,
        delta_frac=0.05,
        eps=1.0,
        energy_min_abs=2e-7,
        stability_eta_deg=0.3,
        use_tvector=True,
        nonnegative_a=False,
        a_grid=48,
        cs_drop_top_frac=0.1,
        cs_sensitivity_frac=0.0,
        scan_basis="sigma",
        off_component_leak_cap=0.3,
        diagnostics={},
        stats=stats_local,
        design=design_override,
        edge_scale=edge_scale,
        edge_mode="tyler",
    )

    detections = list(detections or [])
    if count_isolated_outliers(detections, None, None) == 0:
        return []

    return [
        det
        for det in detections
        if isinstance(det, dict)
        and int(det.get("pre_outlier_count", 0) or 0) == 1
    ]


def test_nested_null_is_rejected_by_tvector_guard() -> None:
    assert not _run_nested_trial(0.0)


def test_nested_spike_survives_isolation_guard() -> None:
    assert _run_nested_trial(6.0)
