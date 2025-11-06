from __future__ import annotations

import numpy as np

from synthetic.calibration import CalibrationConfig, _simulate_panel
from synthetic.threshold_eval import evaluate_threshold_grid
from fjs.overlay import OverlayConfig, detect_spikes
from fjs.balanced import mean_squares


def test_threshold_eval_matches_detect_spikes() -> None:
    config = CalibrationConfig(
        p_assets=12,
        n_groups=24,
        replicates=3,
        delta_abs=0.35,
        delta_frac_grid=(0.0, 0.02),
        stability_grid=(0.3, 0.4),
        edge_modes=("scm", "tyler"),
        trials_null=1,
        trials_alt=0,
        eps=0.02,
        q_max=2,
    )
    rng = np.random.default_rng(1234)
    observations, groups = _simulate_panel(config, rng, spike_strength=0.0)
    stats = mean_squares(observations, groups)

    evaluations = evaluate_threshold_grid(
        observations,
        groups,
        delta_abs=config.delta_abs,
        eps=config.eps,
        edge_modes=config.edge_modes,
        delta_frac_values=config.delta_frac_grid,
        stability_values=config.stability_grid,
        q_max=config.q_max,
        a_grid=120,
        require_isolated=True,
        alignment_min=0.0,
        stats=stats,
    )

    for mode in config.edge_modes:
        matrix = evaluations[str(mode)]
        for delta_idx, delta_frac in enumerate(config.delta_frac_grid):
            for stability_idx, stability_eta in enumerate(config.stability_grid):
                overlay_cfg = OverlayConfig(
                    shrinker="rie",
                    delta=config.delta_abs,
                    delta_frac=float(delta_frac),
                    eps=config.eps,
                    stability_eta_deg=float(stability_eta),
                    q_max=config.q_max,
                    max_detections=config.q_max,
                    edge_mode=str(mode),
                    seed=0,
                    require_isolated=True,
                    a_grid=120,
                )
                detections = detect_spikes(
                    observations,
                    groups,
                    config=overlay_cfg,
                    stats=stats,
                )
                assert bool(detections) == bool(matrix[delta_idx, stability_idx])
