from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fjs.real_design_contract import (
    FactorBinding,
    SourcePartitionBinding,
    stable_sha256,
)
from fjs.rolling_geometry_contract import (
    BOUNDED_PROOF_ENDPOINT_MONTH,
    GEOMETRY_COLUMNS,
    MAX_CAP_STALENESS_DAYS,
    MAX_PANEL_MISSING_FRACTION,
    MIN_COMPLETE_BALANCED_WEEKS,
    MIN_OBSERVED_ASSETS_PER_DATE,
    MIN_PAIRWISE_OBSERVATIONS,
    ROLLING_GEOMETRY_SCHEMA,
    UNIVERSE_SIZE,
    RollingGeometrySpec,
    build_rolling_geometry_proof,
    geometry_logical_sha256,
    headline_calibration_claim,
    load_bound_factor_calendar,
    rolling_geometry_contract,
    validate_rolling_geometry_proof,
)
from fjs.seasoned_geometry_contract import (
    SEASONED_PROOF_SCHEMA,
    build_seasoned_geometry_proof,
    seasoned_geometry_contract,
    validate_seasoned_geometry_proof,
)

TWO_STRATUM_PROOF_SCHEMA = "fjs-two-stratum-geometry-proof/v1"
TWO_STRATUM_MANIFEST_SCHEMA = "fjs-two-stratum-geometry-manifest/v1"
TWO_STRATUM_CONTRACT_ID = "fjs-m7-two-stratum-geometry-v1"
CONTROL_EXEMPT_GATE = "natural_missing_cells_min"


def two_stratum_geometry_contract() -> dict[str, Any]:
    v5 = rolling_geometry_contract()
    v6 = seasoned_geometry_contract()
    payload: dict[str, Any] = {
        "contract_id": TWO_STRATUM_CONTRACT_ID,
        "base_v5_contract_sha256": v5["sha256"],
        "base_v6_contract_sha256": v6["sha256"],
        "method_change": {
            "name": "role_separated_balanced_control_and_missingness_stress",
            "balanced_control": {
                "source": "unchanged_m6_seasoned_point_in_time_top_60",
                "purpose": "balanced_real_geometry_control",
                "required_gate_exception": CONTROL_EXEMPT_GATE,
                "exception_reason": (
                    "natural missingness is measured in the separate stress stratum"
                ),
            },
            "missingness_stress": {
                "candidate_pool": "same_m6_seasoned_eligible_names",
                "point_in_time": True,
                "formation_recency_days_max": MAX_CAP_STALENESS_DAYS,
                "candidate_order": [
                    "natural_missingness_first",
                    "missing_cells_descending",
                    "lagged_market_cap_descending",
                    "permno_ascending",
                ],
                "selection": (
                    "greedily_accept_in_frozen_order_if_per_date_and_total_"
                    "missing_capacities_remain_satisfied_then_stop_at_60"
                ),
                "per_date_missing_capacity_formula": (
                    "universe_size - frozen_minimum_observed_assets_per_date"
                ),
                "total_missing_capacity_formula": (
                    "floor(calendar_dates * universe_size * frozen_panel_cap)"
                ),
                "per_asset_missing_cap_formula": (
                    "floor(calendar_dates * frozen_panel_cap)"
                ),
                "natural_missing_cells_min": v5["coverage_gates"][
                    "natural_missing_cells_min"
                ],
            },
            "one_universe_required_to_be_both_complete_and_incomplete": False,
            "geometry_result_used_to_set_selection_rule": False,
        },
        "role_gate_policy": {
            "balanced_control": (
                "all unchanged v5 computed gates except natural_missing_cells_min"
            ),
            "missingness_stress": "all unchanged v5 computed gates",
            "future_detector_claim": "both strata must pass without pooled rescue",
        },
        "unchanged_coverage_gates": copy.deepcopy(v5["coverage_gates"]),
        "unchanged_headline_calibration_claim": headline_calibration_claim(),
        "unchanged_endpoint_month": BOUNDED_PROOF_ENDPOINT_MONTH,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def derive_missingness_stress_selection(
    seasoned_frame: pd.DataFrame,
    factor_dates: Sequence[pd.Timestamp],
    *,
    spec: RollingGeometrySpec,
    original_scan_receipt: Mapping[str, Any],
    seasoned_selection_receipt: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Apply the frozen M7 stress ranking without reading detector outcomes."""

    spec.validate()
    if tuple(seasoned_frame.columns) != GEOMETRY_COLUMNS:
        raise ValueError("M7 stress selection requires exact geometry columns.")
    start = pd.Timestamp(spec.window_start)
    formation = pd.Timestamp(spec.window_end)
    calendar = (
        pd.DatetimeIndex(pd.to_datetime(list(factor_dates))).sort_values().unique()
    )
    calendar = calendar[(calendar >= start) & (calendar <= formation)]
    if len(calendar) == 0:
        raise ValueError("M7 stress selection received an empty calendar.")
    if (
        int(seasoned_selection_receipt["anchor_week_count"])
        < MIN_COMPLETE_BALANCED_WEEKS
        or int(seasoned_selection_receipt["guaranteed_pairwise_lower_bound"])
        < MIN_PAIRWISE_OBSERVATIONS
    ):
        raise ValueError("M7 candidate pool is not bound to valid M6 seasoning.")

    frame = seasoned_frame.loc[
        seasoned_frame["dlycaldt"].between(start, formation)
        & seasoned_frame["dlycaldt"].isin(calendar)
    ].copy()
    ranking = frame.sort_values(["permno", "dlycaldt"])
    ranking = ranking.groupby("permno", as_index=False).tail(1)
    ranking = ranking.loc[
        (formation - ranking["dlycaldt"]).dt.days.le(MAX_CAP_STALENESS_DAYS)
    ].copy()
    ranking["permno"] = ranking["permno"].astype(int)
    candidate_permnos = sorted(ranking["permno"].tolist())
    if len(candidate_permnos) < UNIVERSE_SIZE:
        raise ValueError("M7 stress pool has fewer than 60 formation-current names.")

    observations = frame.loc[
        frame["permno"].isin(candidate_permnos), ["dlycaldt", "permno"]
    ].drop_duplicates()
    presence = pd.crosstab(observations["dlycaldt"], observations["permno"])
    presence = presence.reindex(
        index=calendar, columns=candidate_permnos, fill_value=0
    ).to_numpy(dtype=bool)
    missing_counts = (~presence).sum(axis=0).astype(int)
    missing_by_permno = dict(zip(candidate_permnos, missing_counts, strict=True))
    ranking["missing_cells"] = ranking["permno"].map(missing_by_permno).astype(int)
    ranking["natural_missing"] = ranking["missing_cells"].gt(0)

    per_asset_missing_cap = int(math.floor(len(calendar) * MAX_PANEL_MISSING_FRACTION))
    total_missing_capacity = int(
        math.floor(len(calendar) * UNIVERSE_SIZE * MAX_PANEL_MISSING_FRACTION)
    )
    per_date_missing_capacity = UNIVERSE_SIZE - MIN_OBSERVED_ASSETS_PER_DATE
    ranking = ranking.loc[ranking["missing_cells"].le(per_asset_missing_cap)].copy()
    ranking = ranking.sort_values(
        ["natural_missing", "missing_cells", "dlycap", "permno"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    natural_candidate_count = int(ranking["natural_missing"].sum())
    if natural_candidate_count == 0:
        raise ValueError("M7 stress pool contains no naturally incomplete candidate.")

    column_index = {permno: index for index, permno in enumerate(candidate_permnos)}
    per_date_selected_missing = np.zeros(len(calendar), dtype=np.int16)
    selected_rows: list[Any] = []
    selected_total_missing = 0
    for row in ranking.itertuples(index=False):
        column = column_index[int(row.permno)]
        missing = ~presence[:, column]
        missing_count = int(row.missing_cells)
        if selected_total_missing + missing_count > total_missing_capacity:
            continue
        if bool(
            np.any(per_date_selected_missing[missing] + 1 > per_date_missing_capacity)
        ):
            continue
        selected_rows.append(row)
        per_date_selected_missing[missing] += 1
        selected_total_missing += missing_count
        if len(selected_rows) == UNIVERSE_SIZE:
            break
    if len(selected_rows) != UNIVERSE_SIZE:
        raise ValueError(
            f"Only {len(selected_rows)} candidates fit the frozen M7 capacities."
        )
    if selected_total_missing < 1:
        raise ValueError("M7 stress selection did not retain natural missingness.")

    selected_permnos = [int(row.permno) for row in selected_rows]
    selected_members = [
        {
            "stress_rank": rank,
            "permno": int(row.permno),
            "missing_cells": int(row.missing_cells),
            "lagged_market_cap": float(row.dlycap),
            "cap_observation_date": pd.Timestamp(row.dlycaldt).date().isoformat(),
        }
        for rank, row in enumerate(selected_rows, start=1)
    ]
    filtered = frame.loc[frame["permno"].isin(selected_permnos)].copy()
    filtered = filtered.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)
    selected_missing_fraction = float(
        selected_total_missing / (len(calendar) * UNIVERSE_SIZE)
    )
    selection: dict[str, Any] = {
        "contract": two_stratum_geometry_contract(),
        "calendar_date_count": len(calendar),
        "source_seasoned_candidate_count": int(
            seasoned_selection_receipt["eligible_candidate_count"]
        ),
        "formation_current_candidate_count": len(candidate_permnos),
        "capacity_eligible_candidate_count": len(ranking),
        "natural_missing_candidate_count": natural_candidate_count,
        "per_asset_missing_cap": per_asset_missing_cap,
        "per_date_missing_capacity": per_date_missing_capacity,
        "total_missing_capacity": total_missing_capacity,
        "selected_total_missing_cells": selected_total_missing,
        "selected_missing_fraction": selected_missing_fraction,
        "selected_max_missing_names_per_date": int(per_date_selected_missing.max()),
        "selected_naturally_incomplete_names": sum(
            int(row.missing_cells) > 0 for row in selected_rows
        ),
        "selected_members": selected_members,
        "selected_member_set_sha256": stable_sha256(selected_members),
        "source_seasoned_geometry_sha256": geometry_logical_sha256(seasoned_frame),
        "filtered_stress_geometry_sha256": geometry_logical_sha256(filtered),
        "filtered_stress_row_count": len(filtered),
        "anchor_week_count": int(seasoned_selection_receipt["anchor_week_count"]),
        "guaranteed_pairwise_lower_bound": int(
            seasoned_selection_receipt["guaranteed_pairwise_lower_bound"]
        ),
        "anchor_gate_guaranteed": int(seasoned_selection_receipt["anchor_week_count"])
        >= MIN_COMPLETE_BALANCED_WEEKS,
        "pairwise_gate_guaranteed": int(
            seasoned_selection_receipt["guaranteed_pairwise_lower_bound"]
        )
        >= MIN_PAIRWISE_OBSERVATIONS,
        "per_date_gate_guaranteed": int(per_date_selected_missing.max())
        <= per_date_missing_capacity,
        "panel_missing_gate_guaranteed": selected_total_missing
        <= total_missing_capacity,
        "natural_missingness_guaranteed": selected_total_missing >= 1,
        "point_in_time": True,
        "detector_outcomes_present": False,
    }
    selection["selection_digest"] = stable_sha256(selection)

    filtered_scan = copy.deepcopy(dict(original_scan_receipt))
    filtered_scan["rows_after_all_filters"] = len(filtered)
    filtered_scan["logical_geometry_sha256"] = geometry_logical_sha256(filtered)
    filtered_scan["m7_stress_selection_digest"] = selection["selection_digest"]
    filtered_scan["sha256"] = stable_sha256(
        {key: value for key, value in filtered_scan.items() if key != "sha256"}
    )
    return filtered, filtered_scan, selection


def build_two_stratum_geometry_proof(
    source_frame: pd.DataFrame,
    *,
    spec: RollingGeometrySpec,
    source_bindings: Sequence[SourcePartitionBinding],
    factor_binding: FactorBinding,
    scan_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    control, seasoned_frame = build_seasoned_geometry_proof(
        source_frame,
        spec=spec,
        source_bindings=source_bindings,
        factor_binding=factor_binding,
        scan_receipt=scan_receipt,
    )
    if control.get("schema") != SEASONED_PROOF_SCHEMA:
        raise ValueError("M7 balanced control is not the unchanged M6 proof.")
    factor_dates = load_bound_factor_calendar(
        factor_binding, start=spec.window_start, end=spec.window_end
    )
    stress_frame, stress_scan, stress_selection = derive_missingness_stress_selection(
        seasoned_frame,
        factor_dates,
        spec=spec,
        original_scan_receipt=scan_receipt,
        seasoned_selection_receipt=control["selection_receipt"],
    )
    stress = build_rolling_geometry_proof(
        stress_frame,
        spec=spec,
        source_bindings=source_bindings,
        factor_binding=factor_binding,
        scan_receipt=stress_scan,
    )
    if stress.get("schema") != ROLLING_GEOMETRY_SCHEMA:
        raise ValueError("M7 stress stratum base proof schema mismatch.")

    control_gates = copy.deepcopy(control["base_v5_geometry_proof"]["coverage_gates"])
    control_required = {
        key: bool(value)
        for key, value in control_gates.items()
        if key != CONTROL_EXEMPT_GATE
    }
    control_role_passed = all(control_required.values())
    stress_required = copy.deepcopy(stress["coverage_gates"])
    stress_role_passed = all(stress_required.values())
    proof: dict[str, Any] = {
        "schema": TWO_STRATUM_PROOF_SCHEMA,
        "cell_id": "fjs-two-stratum-geometry-2013-01-v7",
        "contract": two_stratum_geometry_contract(),
        "headline_calibration_claim": headline_calibration_claim(),
        "balanced_control": {
            "role": "seasoned_balanced_real_control",
            "m6_proof": control,
            "all_computed_gate_results": control_gates,
            "required_gate_results": control_required,
            "natural_missingness_gate_applies": False,
            "role_passed": control_role_passed,
        },
        "missingness_stress": {
            "role": "seasoned_real_missingness_stress",
            "selection_receipt": stress_selection,
            "base_v5_geometry_proof": stress,
            "required_gate_results": stress_required,
            "natural_missingness_gate_applies": True,
            "role_passed": stress_role_passed,
        },
        "both_strata_required": True,
        "coverage_proof_passed": control_role_passed and stress_role_passed,
        "full_72_endpoint_derivation_run": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    proof["proof_digest"] = stable_sha256(proof)
    validate_two_stratum_geometry_proof(
        proof,
        seasoned_frame=seasoned_frame,
        stress_frame=stress_frame,
    )
    return proof, seasoned_frame, stress_frame


def validate_two_stratum_geometry_proof(
    proof: Mapping[str, Any],
    *,
    seasoned_frame: pd.DataFrame | None = None,
    stress_frame: pd.DataFrame | None = None,
    revalidate_external: bool = False,
) -> None:
    if proof.get("schema") != TWO_STRATUM_PROOF_SCHEMA:
        raise ValueError("M7 two-stratum geometry schema mismatch.")
    if proof.get("contract") != two_stratum_geometry_contract():
        raise ValueError("M7 two-stratum contract mismatch.")
    if proof.get("headline_calibration_claim") != headline_calibration_claim():
        raise ValueError("M7 headline claim changed from M5.")

    control = proof["balanced_control"]
    control_proof = control["m6_proof"]
    validate_seasoned_geometry_proof(control_proof, filtered_frame=seasoned_frame)
    control_base = control_proof["base_v5_geometry_proof"]
    if revalidate_external:
        validate_rolling_geometry_proof(
            control_base,
            source_frame=seasoned_frame,
            revalidate_external=True,
        )
    if int(control_base["geometry_metrics"]["missing_cells"]) != 0:
        raise ValueError("M7 balanced control is not the archived complete M6 panel.")
    expected_control_gates = copy.deepcopy(control_base["coverage_gates"])
    expected_control_required = {
        key: bool(value)
        for key, value in expected_control_gates.items()
        if key != CONTROL_EXEMPT_GATE
    }
    if (
        control.get("all_computed_gate_results") != expected_control_gates
        or control.get("required_gate_results") != expected_control_required
        or control.get("natural_missingness_gate_applies") is not False
        or control.get("role_passed") is not all(expected_control_required.values())
    ):
        raise ValueError("M7 balanced-control role decision mismatch.")

    stress_wrapper = proof["missingness_stress"]
    stress = stress_wrapper["base_v5_geometry_proof"]
    validate_rolling_geometry_proof(
        stress,
        source_frame=stress_frame,
        revalidate_external=revalidate_external,
    )
    selection = stress_wrapper["selection_receipt"]
    if selection.get("selection_digest") != stable_sha256(
        {key: value for key, value in selection.items() if key != "selection_digest"}
    ):
        raise ValueError("M7 stress selection digest mismatch.")
    if selection.get("selected_member_set_sha256") != stable_sha256(
        selection["selected_members"]
    ):
        raise ValueError("M7 stress member-set digest mismatch.")
    if stress_frame is not None and (
        geometry_logical_sha256(stress_frame)
        != selection["filtered_stress_geometry_sha256"]
        or len(stress_frame) != int(selection["filtered_stress_row_count"])
    ):
        raise ValueError("M7 stress frame changed after selection.")
    guarantees = (
        "anchor_gate_guaranteed",
        "pairwise_gate_guaranteed",
        "per_date_gate_guaranteed",
        "panel_missing_gate_guaranteed",
        "natural_missingness_guaranteed",
    )
    if not all(selection.get(key) is True for key in guarantees):
        raise ValueError("M7 stress selection does not guarantee frozen feasibility.")
    expected_stress_required = copy.deepcopy(stress["coverage_gates"])
    if (
        stress_wrapper.get("required_gate_results") != expected_stress_required
        or stress_wrapper.get("natural_missingness_gate_applies") is not True
        or stress_wrapper.get("role_passed")
        is not all(expected_stress_required.values())
    ):
        raise ValueError("M7 stress role decision mismatch.")

    expected_aggregate = bool(control["role_passed"]) and bool(
        stress_wrapper["role_passed"]
    )
    if (
        proof.get("both_strata_required") is not True
        or proof.get("coverage_proof_passed") is not expected_aggregate
    ):
        raise ValueError("M7 two-stratum aggregate decision mismatch.")
    if proof.get("proof_digest") != stable_sha256(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    ):
        raise ValueError("M7 two-stratum proof digest mismatch.")


def build_two_stratum_geometry_manifest(
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    validate_two_stratum_geometry_proof(proof)
    control = proof["balanced_control"]
    control_base = control["m6_proof"]["base_v5_geometry_proof"]
    stress = proof["missingness_stress"]
    stress_base = stress["base_v5_geometry_proof"]
    selection = stress["selection_receipt"]
    manifest: dict[str, Any] = {
        "schema": TWO_STRATUM_MANIFEST_SCHEMA,
        "contract": two_stratum_geometry_contract(),
        "headline_calibration_claim": headline_calibration_claim(),
        "proof_cell": {
            "cell_id": proof["cell_id"],
            "proof_digest": proof["proof_digest"],
            "coverage_proof_passed": proof["coverage_proof_passed"],
        },
        "balanced_control": {
            "geometry_metrics": copy.deepcopy(control_base["geometry_metrics"]),
            "target_boundary_feasibility": copy.deepcopy(
                control_base["target_boundary_feasibility"]
            ),
            "all_computed_gate_results": copy.deepcopy(
                control["all_computed_gate_results"]
            ),
            "required_gate_results": copy.deepcopy(control["required_gate_results"]),
            "role_passed": control["role_passed"],
        },
        "missingness_stress": {
            "selection_aggregates": {
                "source_seasoned_candidate_count": selection[
                    "source_seasoned_candidate_count"
                ],
                "formation_current_candidate_count": selection[
                    "formation_current_candidate_count"
                ],
                "capacity_eligible_candidate_count": selection[
                    "capacity_eligible_candidate_count"
                ],
                "natural_missing_candidate_count": selection[
                    "natural_missing_candidate_count"
                ],
                "selected_naturally_incomplete_names": selection[
                    "selected_naturally_incomplete_names"
                ],
                "selected_total_missing_cells": selection[
                    "selected_total_missing_cells"
                ],
                "selected_missing_fraction": selection["selected_missing_fraction"],
                "selected_max_missing_names_per_date": selection[
                    "selected_max_missing_names_per_date"
                ],
                "selection_digest": selection["selection_digest"],
            },
            "geometry_metrics": copy.deepcopy(stress_base["geometry_metrics"]),
            "target_boundary_feasibility": copy.deepcopy(
                stress_base["target_boundary_feasibility"]
            ),
            "required_gate_results": copy.deepcopy(stress["required_gate_results"]),
            "role_passed": stress["role_passed"],
        },
        "coverage_proof_passed": proof["coverage_proof_passed"],
        "full_72_endpoint_derivation_run": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    manifest["manifest_digest"] = stable_sha256(manifest)
    return manifest
