from __future__ import annotations

import pandas as pd
import pytest

from fjs.rolling_geometry_contract import (
    GEOMETRY_COLUMNS,
    MIN_COMPLETE_BALANCED_WEEKS,
    MIN_PAIRWISE_OBSERVATIONS,
    RollingGeometrySpec,
    headline_calibration_claim,
    rolling_geometry_contract,
    rolling_window_start,
)
from fjs.seasoned_geometry_contract import (
    derive_seasoned_selection,
    minimum_asset_observations_for_pairwise,
    seasoned_geometry_contract,
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


def _frame(*, missing_date: pd.Timestamp | None) -> pd.DataFrame:
    dates = _calendar()
    rows = []
    for index, permno in enumerate(range(10001, 10062), start=1):
        for date in dates:
            if permno == 10061 and missing_date is not None and date == missing_date:
                continue
            rows.append(
                {
                    "permno": permno,
                    "dlycaldt": date,
                    "dlycap": 1_000_000.0 + index * 10_000.0,
                }
            )
    return pd.DataFrame(rows, columns=list(GEOMETRY_COLUMNS))


@pytest.mark.unit
def test_m6_pairwise_rule_is_derived_from_the_frozen_gate() -> None:
    n_dates = len(_calendar())
    minimum = minimum_asset_observations_for_pairwise(n_dates)
    assert minimum == 640
    assert 2 * minimum - n_dates >= MIN_PAIRWISE_OBSERVATIONS
    contract = seasoned_geometry_contract()
    assert (
        contract["unchanged_coverage_gates"]
        == rolling_geometry_contract()["coverage_gates"]
    )
    assert contract["unchanged_headline_calibration_claim"] == (
        headline_calibration_claim()
    )
    assert contract["method_change"]["v5_observed_438_or_72_used_to_set_rule"] is False


@pytest.mark.unit
def test_m6_seasoned_selection_guarantees_both_feasibility_requirements() -> None:
    calendar = _calendar()
    source = _frame(missing_date=calendar[0])
    filtered, _, selection = derive_seasoned_selection(
        source,
        calendar,
        spec=_spec(),
        original_scan_receipt={"sha256": "fixture"},
    )
    assert selection["anchor_week_count"] == MIN_COMPLETE_BALANCED_WEEKS
    assert selection["guaranteed_pairwise_lower_bound"] >= (MIN_PAIRWISE_OBSERVATIONS)
    assert selection["complete_week_gate_guaranteed"] is True
    assert selection["pairwise_gate_guaranteed"] is True
    assert selection["eligible_candidate_count"] == 61
    assert len(filtered) == len(source)


@pytest.mark.unit
def test_m6_anchor_week_requirement_excludes_unseasoned_candidate() -> None:
    calendar = _calendar()
    labels = pd.DatetimeIndex(
        [value - pd.Timedelta(days=value.weekday()) for value in calendar]
    )
    complete = [
        label
        for label in labels.unique().sort_values()
        if int((labels == label).sum()) == 5
    ]
    first_anchor = complete[-MIN_COMPLETE_BALANCED_WEEKS]
    missing = calendar[labels == first_anchor][0]
    _, _, selection = derive_seasoned_selection(
        _frame(missing_date=missing),
        calendar,
        spec=_spec(),
        original_scan_receipt={"sha256": "fixture"},
    )
    assert selection["eligible_candidate_count"] == 60
