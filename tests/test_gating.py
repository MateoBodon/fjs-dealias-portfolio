from __future__ import annotations

from fjs.gating import count_isolated_outliers, select_top_k


def test_count_isolated_outliers_zero_when_missing_isolated() -> None:
    detections = [
        {"lambda_hat": 2.1, "pre_outlier_count": 3, "stability_margin": 0.4},
        {"lambda_hat": 1.9, "pre_outlier_count": 0, "stability_margin": 0.2},
    ]
    assert count_isolated_outliers(detections, None, None) == 0


def test_select_top_k_prefers_high_score_and_edge_margin() -> None:
    detections = [
        {
            "lambda_hat": 2.5,
            "target_energy": 0.8,
            "stability_margin": 0.6,
            "edge_margin": 0.05,
        },
        {
            "lambda_hat": 2.3,
            "target_energy": 0.9,
            "stability_margin": 0.7,
            "edge_margin": 0.1,
        },
        {
            "lambda_hat": 2.4,
            "target_energy": 0.8,
            "stability_margin": 0.6,
            "edge_margin": 0.2,
        },
    ]
    selected, discarded = select_top_k(detections, 2)
    assert len(selected) == 2
    assert len(discarded) == 1
    # Highest score should be the second detection (0.9 * 0.7)
    assert selected[0]["lambda_hat"] == 2.3
    # Tie in score uses edge margin (0.2 > 0.05)
    assert selected[1]["lambda_hat"] == 2.4
    assert discarded[0]["lambda_hat"] == 2.5
