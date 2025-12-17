from fjs.gating import lookup_calibrated_delta


def test_lookup_calibrated_delta_nested_tyler() -> None:
    delta = lookup_calibrated_delta(
        "tyler", 188, 70, calibration_path="calibration/edge_delta_thresholds.json"
    )
    assert delta is not None
    assert delta > 0


def test_lookup_calibrated_delta_nested_huber() -> None:
    delta = lookup_calibrated_delta(
        "huber", 200, 80, calibration_path="calibration/edge_delta_thresholds.json"
    )
    assert delta is not None
    assert delta > 0
