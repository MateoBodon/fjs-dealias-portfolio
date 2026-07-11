from __future__ import annotations

import pandas as pd
import pytest

from fjs.rolling_geometry_contract import (
    GEOMETRY_COLUMNS,
    MAX_PANEL_MISSING_FRACTION,
    MIN_OBSERVED_ASSETS_PER_DATE,
    RollingGeometrySpec,
    headline_calibration_claim,
    rolling_geometry_contract,
    rolling_window_start,
)
from fjs.seasoned_geometry_contract import derive_seasoned_selection
from fjs.two_stratum_geometry_contract import (
    derive_missingness_stress_selection,
    two_stratum_geometry_contract,
)


def _calendar() -> pd.DatetimeIndex:
    formation = pd.Timestamp("2013-01-31")
    return pd.bdate_range(rolling_window_start(formation), formation)


def _spec() -> RollingGeometrySpec:
    formation = pd.Timestamp("2013-01-31")
    return RollingGeometrySpec(
        endpoint_month="2013-01",
        formation_date=formation.date().isoformat(),
        window_start=rolling_window_start(formation).date().isoformat(),
        window_end=formation.date().isoformat(),
        proof_only=True,
    )


def _frame(*, naturally_incomplete: bool) -> pd.DataFrame:
    dates = _calendar()
    rows = []
    for index, permno in enumerate(range(10001, 10081), start=1):
        for date_index, date in enumerate(dates):
            drop = False
            if naturally_incomplete and index <= 12 and date_index < 36:
                # Some candidates compete for the same date capacity while
                # others carry distinct early-window natural gaps. All gaps
                # precede the common most-recent 78-week anchor.
                drop = date_index % 12 == (index - 1) % 12
            if drop:
                continue
            rows.append(
                {
                    "permno": permno,
                    "dlycaldt": date,
                    "dlycap": 1_000_000.0 + index * 10_000.0,
                }
            )
    return pd.DataFrame(rows, columns=list(GEOMETRY_COLUMNS))


def _seasoned(source: pd.DataFrame):
    return derive_seasoned_selection(
        source,
        _calendar(),
        spec=_spec(),
        original_scan_receipt={"sha256": "fixture"},
    )


@pytest.mark.unit
def test_m7_preserves_all_numeric_gates_and_headline_claim() -> None:
    contract = two_stratum_geometry_contract()
    v5 = rolling_geometry_contract()
    assert contract["unchanged_coverage_gates"] == v5["coverage_gates"]
    assert contract["unchanged_headline_calibration_claim"] == (
        headline_calibration_claim()
    )
    assert (
        contract["method_change"]["geometry_result_used_to_set_selection_rule"] is False
    )
    assert contract["role_gate_policy"]["future_detector_claim"] == (
        "both strata must pass without pooled rescue"
    )


@pytest.mark.unit
def test_m7_stress_selection_is_deterministic_and_capacity_safe() -> None:
    source = _frame(naturally_incomplete=True)
    seasoned, _, seasoned_receipt = _seasoned(source)
    selected, _, selection = derive_missingness_stress_selection(
        seasoned,
        _calendar(),
        spec=_spec(),
        original_scan_receipt={"sha256": "fixture"},
        seasoned_selection_receipt=seasoned_receipt,
    )
    shuffled = seasoned.sample(frac=1.0, random_state=7).reset_index(drop=True)
    _, _, repeat = derive_missingness_stress_selection(
        shuffled,
        _calendar(),
        spec=_spec(),
        original_scan_receipt={"sha256": "fixture"},
        seasoned_selection_receipt=seasoned_receipt,
    )
    assert (
        selection["selected_member_set_sha256"] == repeat["selected_member_set_sha256"]
    )
    assert selected["permno"].nunique() == 60
    assert selection["selected_total_missing_cells"] >= 1
    assert selection["selected_naturally_incomplete_names"] >= 1
    assert selection["selected_max_missing_names_per_date"] <= (
        60 - MIN_OBSERVED_ASSETS_PER_DATE
    )
    assert selection["selected_missing_fraction"] <= MAX_PANEL_MISSING_FRACTION
    assert selection["anchor_gate_guaranteed"] is True
    assert selection["pairwise_gate_guaranteed"] is True
    assert selection["per_date_gate_guaranteed"] is True
    assert selection["panel_missing_gate_guaranteed"] is True


@pytest.mark.unit
def test_m7_stress_fails_closed_without_natural_missingness() -> None:
    source = _frame(naturally_incomplete=False)
    seasoned, _, seasoned_receipt = _seasoned(source)
    with pytest.raises(ValueError, match="no naturally incomplete candidate"):
        derive_missingness_stress_selection(
            seasoned,
            _calendar(),
            spec=_spec(),
            original_scan_receipt={"sha256": "fixture"},
            seasoned_selection_receipt=seasoned_receipt,
        )
