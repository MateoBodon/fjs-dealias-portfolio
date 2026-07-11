from __future__ import annotations

import numpy as np
import pytest
from tools.generate_fjs_between_fixture import (
    DEFAULT_INPUT_SPEC,
    aggregate_curve,
    build_panel,
    draw_trial,
    load_input_spec,
    render_outputs,
)

pytestmark = pytest.mark.unit


def test_frozen_input_spec_matches_predeclaration() -> None:
    spec = load_input_spec(DEFAULT_INPUT_SPEC)

    assert spec["master_seed"] == 20260710
    assert spec["trial_count"] == 12
    assert spec["mu_grid"] == [0.0, 6.0]
    assert spec["inject_mode"] == "between"


def test_trial_draws_and_paired_panels_are_deterministic() -> None:
    child = np.random.SeedSequence(20260710).spawn(1)[0]
    first = draw_trial(child, groups=3, replicates=2, features=4)
    second = draw_trial(child, groups=3, replicates=2, features=4)

    for observed, expected in zip(first, second, strict=True):
        assert np.array_equal(observed, expected)
    np.testing.assert_allclose(
        np.linalg.norm(first[0]),
        1.0,
    )

    null_panel, labels = build_panel(
        direction=first[0],
        group_scores=first[1],
        residuals=first[2],
        mu=0.0,
        within_noise_scale=0.3,
    )
    strong_panel, strong_labels = build_panel(
        direction=first[0],
        group_scores=first[1],
        residuals=first[2],
        mu=6.0,
        within_noise_scale=0.3,
    )
    assert np.array_equal(labels, strong_labels)
    assert not np.array_equal(null_panel, strong_panel)


def test_curve_render_is_stable_and_acceptance_is_subset() -> None:
    trials = [
        {
            "trial_index": index,
            "child_spawn_key": str(index),
            "mu": mu,
            "inject_mode": "between",
            "detected": detected,
            "accepted": accepted,
            "pre_gate_candidate_count": detected,
            "accepted_candidate_count": accepted,
        }
        for index, mu, detected, accepted in (
            (0, 0.0, 0, 0),
            (1, 0.0, 0, 0),
            (0, 6.0, 1, 1),
            (1, 6.0, 1, 0),
        )
    ]

    curve = aggregate_curve(trials)
    curve_bytes, trial_bytes, rendered_curve = render_outputs(trials)

    assert curve == rendered_curve
    assert curve[0]["detection_rate"] == 0.0
    assert curve[1]["detection_rate"] == 1.0
    assert curve[1]["acceptance_rate"] == 0.5
    assert curve_bytes.startswith(b"mu,inject_mode,detection_rate")
    assert trial_bytes.startswith(b"trial_index,child_spawn_key,mu")
