from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fjs.real_design_contract import (
    FactorBinding,
    SourcePartitionBinding,
    stable_sha256,
)
from fjs.rolling_geometry_contract import (
    BOUNDED_PROOF_ENDPOINT_MONTH,
    GEOMETRY_COLUMNS,
    MIN_COMPLETE_BALANCED_WEEKS,
    MIN_PAIRWISE_OBSERVATIONS,
    ROLLING_GEOMETRY_SCHEMA,
    RollingGeometrySpec,
    build_rolling_geometry_proof,
    geometry_logical_sha256,
    headline_calibration_claim,
    load_bound_factor_calendar,
    rolling_geometry_contract,
    validate_rolling_geometry_proof,
    week_start,
)

SEASONED_PROOF_SCHEMA = "fjs-seasoned-geometry-proof/v1"
SEASONED_MANIFEST_SCHEMA = "fjs-seasoned-geometry-manifest/v1"
SEASONED_CONTRACT_ID = "fjs-m6-seasoned-universe-geometry-v1"


def minimum_asset_observations_for_pairwise(
    calendar_dates: int, pairwise_required: int = MIN_PAIRWISE_OBSERVATIONS
) -> int:
    if calendar_dates <= 0 or pairwise_required <= 0:
        raise ValueError("Seasoned coverage inputs must be positive.")
    if pairwise_required > calendar_dates:
        raise ValueError("Pairwise requirement cannot exceed the calendar length.")
    return int(math.ceil((calendar_dates + pairwise_required) / 2.0))


def seasoned_geometry_contract() -> dict[str, Any]:
    v5 = rolling_geometry_contract()
    payload: dict[str, Any] = {
        "contract_id": SEASONED_CONTRACT_ID,
        "base_v5_contract_sha256": v5["sha256"],
        "method_change": {
            "name": "point_in_time_seasoned_universe",
            "common_anchor_week_count": MIN_COMPLETE_BALANCED_WEEKS,
            "anchor_week_selection": (
                "most_recent_complete_five_factor_date_weeks_before_formation"
            ),
            "anchor_observation_requirement": "every_anchor_date_observed",
            "minimum_asset_observations_formula": (
                "ceil((calendar_dates + frozen_pairwise_minimum) / 2)"
            ),
            "pairwise_lower_bound_formula": (
                "observed_i + observed_j - calendar_dates"
            ),
            "derived_only_from_frozen_gates": True,
            "v5_observed_438_or_72_used_to_set_rule": False,
        },
        "unchanged_coverage_gates": copy.deepcopy(v5["coverage_gates"]),
        "unchanged_headline_calibration_claim": headline_calibration_claim(),
        "unchanged_endpoint_month": BOUNDED_PROOF_ENDPOINT_MONTH,
        "point_in_time_top_60_ranking_unchanged": True,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def derive_seasoned_selection(
    source_frame: pd.DataFrame,
    factor_dates: Sequence[pd.Timestamp],
    *,
    spec: RollingGeometrySpec,
    original_scan_receipt: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    spec.validate()
    if tuple(source_frame.columns) != GEOMETRY_COLUMNS:
        raise ValueError("M6 seasoned selection requires exact geometry columns.")
    start = pd.Timestamp(spec.window_start)
    end = pd.Timestamp(spec.window_end)
    calendar = (
        pd.DatetimeIndex(pd.to_datetime(list(factor_dates))).sort_values().unique()
    )
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    labels = pd.DatetimeIndex([week_start(value) for value in calendar])
    complete_labels = [
        label
        for label in labels.unique().sort_values()
        if int((labels == label).sum()) == 5
    ]
    if len(complete_labels) < MIN_COMPLETE_BALANCED_WEEKS:
        raise ValueError("Calendar cannot supply the frozen common anchor weeks.")
    anchor_labels = complete_labels[-MIN_COMPLETE_BALANCED_WEEKS:]
    anchor_dates = calendar[labels.isin(anchor_labels)]
    minimum_observations = minimum_asset_observations_for_pairwise(len(calendar))

    frame = source_frame.loc[
        source_frame["dlycaldt"].between(start, end)
        & source_frame["dlycaldt"].isin(calendar)
    ].copy()
    total_counts = frame.groupby("permno")["dlycaldt"].nunique()
    anchor_counts = (
        frame.loc[frame["dlycaldt"].isin(anchor_dates)]
        .groupby("permno")["dlycaldt"]
        .nunique()
    )
    eligible = total_counts.loc[
        total_counts.ge(minimum_observations)
    ].index.intersection(
        anchor_counts.loc[anchor_counts.eq(len(anchor_dates))].index,
        sort=False,
    )
    filtered = frame.loc[frame["permno"].isin(eligible)].copy()
    filtered = filtered.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)
    if len(eligible) < 60:
        raise ValueError(f"Only {len(eligible)} candidates satisfy M6 seasoning.")

    eligible_permnos = sorted(int(value) for value in eligible)
    pairwise_lower_bound = 2 * minimum_observations - len(calendar)
    selection: dict[str, Any] = {
        "contract": seasoned_geometry_contract(),
        "calendar_date_count": len(calendar),
        "minimum_asset_observations": minimum_observations,
        "frozen_pairwise_requirement": MIN_PAIRWISE_OBSERVATIONS,
        "guaranteed_pairwise_lower_bound": pairwise_lower_bound,
        "anchor_week_count": len(anchor_labels),
        "anchor_date_count": len(anchor_dates),
        "anchor_week_labels": [
            pd.Timestamp(value).date().isoformat() for value in anchor_labels
        ],
        "anchor_calendar_sha256": stable_sha256(
            [pd.Timestamp(value).date().isoformat() for value in anchor_dates]
        ),
        "eligible_candidate_count": len(eligible_permnos),
        "eligible_permno_set_sha256": stable_sha256(eligible_permnos),
        "original_geometry_sha256": geometry_logical_sha256(source_frame),
        "filtered_geometry_sha256": geometry_logical_sha256(filtered),
        "filtered_row_count": len(filtered),
        "complete_week_gate_guaranteed": len(anchor_labels)
        >= MIN_COMPLETE_BALANCED_WEEKS,
        "pairwise_gate_guaranteed": pairwise_lower_bound >= MIN_PAIRWISE_OBSERVATIONS,
        "outcomes_observed": False,
    }
    selection["selection_digest"] = stable_sha256(selection)

    filtered_scan = copy.deepcopy(dict(original_scan_receipt))
    filtered_scan["rows_after_all_filters"] = len(filtered)
    filtered_scan["logical_geometry_sha256"] = geometry_logical_sha256(filtered)
    filtered_scan["seasoned_selection_digest"] = selection["selection_digest"]
    filtered_scan["sha256"] = stable_sha256(
        {key: value for key, value in filtered_scan.items() if key != "sha256"}
    )
    return filtered, filtered_scan, selection


def build_seasoned_geometry_proof(
    source_frame: pd.DataFrame,
    *,
    spec: RollingGeometrySpec,
    source_bindings: Sequence[SourcePartitionBinding],
    factor_binding: FactorBinding,
    scan_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    factor_dates = load_bound_factor_calendar(
        factor_binding, start=spec.window_start, end=spec.window_end
    )
    filtered, filtered_scan, selection = derive_seasoned_selection(
        source_frame,
        factor_dates,
        spec=spec,
        original_scan_receipt=scan_receipt,
    )
    base = build_rolling_geometry_proof(
        filtered,
        spec=spec,
        source_bindings=source_bindings,
        factor_binding=factor_binding,
        scan_receipt=filtered_scan,
    )
    if base.get("schema") != ROLLING_GEOMETRY_SCHEMA:
        raise ValueError("M6 base geometry proof schema mismatch.")
    proof: dict[str, Any] = {
        "schema": SEASONED_PROOF_SCHEMA,
        "cell_id": "fjs-seasoned-geometry-2013-01-v6",
        "contract": seasoned_geometry_contract(),
        "headline_calibration_claim": headline_calibration_claim(),
        "selection_receipt": selection,
        "base_v5_geometry_proof": base,
        "coverage_gates": copy.deepcopy(base["coverage_gates"]),
        "coverage_proof_passed": base["coverage_proof_passed"],
        "full_72_endpoint_derivation_run": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    proof["proof_digest"] = stable_sha256(proof)
    validate_seasoned_geometry_proof(proof, filtered_frame=filtered)
    return proof, filtered


def validate_seasoned_geometry_proof(
    proof: Mapping[str, Any], *, filtered_frame: pd.DataFrame | None = None
) -> None:
    if proof.get("schema") != SEASONED_PROOF_SCHEMA:
        raise ValueError("M6 seasoned geometry schema mismatch.")
    if proof.get("contract") != seasoned_geometry_contract():
        raise ValueError("M6 seasoned geometry contract mismatch.")
    if proof.get("headline_calibration_claim") != headline_calibration_claim():
        raise ValueError("M6 headline claim changed from M5.")
    base = proof["base_v5_geometry_proof"]
    validate_rolling_geometry_proof(base, source_frame=filtered_frame)
    if proof.get("coverage_gates") != base["coverage_gates"]:
        raise ValueError("M6 coverage gates changed from the M5 base proof.")
    if proof.get("coverage_proof_passed") is not base["coverage_proof_passed"]:
        raise ValueError("M6 aggregate gate decision mismatch.")
    selection = proof["selection_receipt"]
    if selection.get("selection_digest") != stable_sha256(
        {key: value for key, value in selection.items() if key != "selection_digest"}
    ):
        raise ValueError("M6 seasoned selection digest mismatch.")
    if (
        selection["guaranteed_pairwise_lower_bound"] < MIN_PAIRWISE_OBSERVATIONS
        or selection["anchor_week_count"] < MIN_COMPLETE_BALANCED_WEEKS
    ):
        raise ValueError("M6 seasoned eligibility does not imply frozen feasibility.")
    if proof.get("proof_digest") != stable_sha256(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    ):
        raise ValueError("M6 seasoned geometry proof digest mismatch.")


def build_seasoned_geometry_manifest(proof: Mapping[str, Any]) -> dict[str, Any]:
    validate_seasoned_geometry_proof(proof)
    base = proof["base_v5_geometry_proof"]
    selection = proof["selection_receipt"]
    manifest: dict[str, Any] = {
        "schema": SEASONED_MANIFEST_SCHEMA,
        "contract": seasoned_geometry_contract(),
        "headline_calibration_claim": headline_calibration_claim(),
        "proof_cell": {
            "cell_id": proof["cell_id"],
            "proof_digest": proof["proof_digest"],
            "coverage_proof_passed": proof["coverage_proof_passed"],
        },
        "seasoned_eligibility": {
            "calendar_date_count": selection["calendar_date_count"],
            "minimum_asset_observations": selection["minimum_asset_observations"],
            "guaranteed_pairwise_lower_bound": selection[
                "guaranteed_pairwise_lower_bound"
            ],
            "anchor_week_count": selection["anchor_week_count"],
            "anchor_date_count": selection["anchor_date_count"],
            "eligible_candidate_count": selection["eligible_candidate_count"],
            "selection_digest": selection["selection_digest"],
        },
        "geometry_metrics": copy.deepcopy(base["geometry_metrics"]),
        "target_boundary_feasibility": copy.deepcopy(
            base["target_boundary_feasibility"]
        ),
        "coverage_gates": copy.deepcopy(base["coverage_gates"]),
        "coverage_proof_passed": base["coverage_proof_passed"],
        "full_72_endpoint_derivation_run": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    manifest["manifest_digest"] = stable_sha256(manifest)
    return manifest
