from experiments.equity_panel import run


def test_infer_skip_reason_calibration_missing() -> None:
    reason = run._infer_skip_reason(
        {"edge_buffer": 4, "stability_fail": 2},
        calibration_missing=True,
        isolated_spikes=0,
    )
    assert reason.primary == "calibration_missing_p_T"
    assert reason.detail == ""


def test_infer_skip_reason_prefers_stability() -> None:
    diag = {"edge_buffer": 1, "stability_fail": 3}
    reason = run._infer_skip_reason(
        diag, calibration_missing=False, isolated_spikes=0
    )
    assert reason.primary == "instability_in_a_neighborhood"
    assert "stability_fail" in reason.detail
