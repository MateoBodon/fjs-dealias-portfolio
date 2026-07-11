from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tools.fjs_m4_contract import (
    ROOT,
    build_cells,
    file_sha256,
    manifest_profile,
    stable_sha256,
)

from fjs.invariance_contract import (
    REQUIRED_INVARIANCE_CHECKS,
    InvarianceTolerances,
)

SCHEMA_VERSION = 3
FULL_MANIFEST_ID = "fjs-m4-full-target-between-v3"
SMOKE_MANIFEST_ID = "fjs-m4-smoke-target-between-v3"
BOUNDARY_CONTRACT_ID = "fjs-balanced-oneway-isolated-target-edge-v1"
INVARIANCE_CONTRACT_ID = "fjs-decision-invariance-v1"
POWER_BOUNDARY_MULTIPLIER = 1.5

CONTRACT_BINDING_PATHS = (
    "src/fjs/invariance_contract.py",
    "tools/fjs_m4_contract_v3.py",
    "tools/freeze_fjs_m4_manifest_v3.py",
)


@dataclass(frozen=True)
class OneWayBoundaryContract:
    p_assets: int
    n_groups: int
    replicates: int
    target_rank: int
    between_bulk_variance: float
    residual_bulk_variance: float
    bulk_dimension: int
    between_degrees_of_freedom: int
    within_degrees_of_freedom: int
    between_aspect_ratio: float
    within_aspect_ratio: float
    between_mean_square_bulk: float
    within_mean_square_bulk: float
    excess_eigenvalue_boundary: float
    population_eigenvalue_boundary: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": BOUNDARY_CONTRACT_ID,
            "boundary_kind": "population_between_covariance_eigenvalue",
            "formula": (
                "sigma_between + sqrt(y_between*C_between^2 + "
                "y_within*C_within^2) / replicates"
            ),
            "inputs": {
                "p_assets": self.p_assets,
                "n_groups": self.n_groups,
                "replicates": self.replicates,
                "target_rank": self.target_rank,
                "between_bulk_variance": self.between_bulk_variance,
                "residual_bulk_variance": self.residual_bulk_variance,
            },
            "derived": {
                "bulk_dimension": self.bulk_dimension,
                "between_degrees_of_freedom": self.between_degrees_of_freedom,
                "within_degrees_of_freedom": self.within_degrees_of_freedom,
                "between_aspect_ratio": self.between_aspect_ratio,
                "within_aspect_ratio": self.within_aspect_ratio,
                "between_mean_square_bulk": self.between_mean_square_bulk,
                "within_mean_square_bulk": self.within_mean_square_bulk,
                "excess_eigenvalue_boundary": self.excess_eigenvalue_boundary,
                "population_eigenvalue_boundary": (self.population_eigenvalue_boundary),
            },
            "scope": {
                "asymptotic_phase_transition": True,
                "finite_sample_operating_margin_included": False,
                "delta_and_edge_mode_calibrated_separately": True,
                "finite_rank_residual_nuisance_excluded_from_bulk": True,
                "nuisance_attribution_required_separately": True,
            },
        }


def independently_computed_oneway_boundary(
    *,
    p_assets: int,
    n_groups: int,
    replicates: int,
    target_rank: int = 1,
    between_bulk_variance: float = 0.0,
    residual_bulk_variance: float = 1.0,
) -> OneWayBoundaryContract:
    p_value = int(p_assets)
    group_value = int(n_groups)
    replicate_value = int(replicates)
    rank_value = int(target_rank)
    between_variance = float(between_bulk_variance)
    residual_variance = float(residual_bulk_variance)
    if p_value < 2:
        raise ValueError("p_assets must be at least two.")
    if group_value < 2:
        raise ValueError("n_groups must be at least two.")
    if replicate_value < 2:
        raise ValueError("replicates must be at least two.")
    if rank_value < 0 or rank_value >= p_value:
        raise ValueError("target_rank must satisfy 0 <= target_rank < p_assets.")
    if not math.isfinite(between_variance) or between_variance < 0.0:
        raise ValueError("between_bulk_variance must be finite and non-negative.")
    if not math.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("residual_bulk_variance must be finite and positive.")

    bulk_dimension = p_value - rank_value
    between_dof = group_value - 1
    within_dof = group_value * (replicate_value - 1)
    y_between = bulk_dimension / float(between_dof)
    y_within = bulk_dimension / float(within_dof)
    c_between = replicate_value * between_variance + residual_variance
    c_within = residual_variance
    excess = (
        math.sqrt(y_between * c_between * c_between + y_within * c_within * c_within)
        / replicate_value
    )
    boundary = between_variance + excess
    if not math.isfinite(boundary) or boundary <= 0.0:
        raise ValueError("The derived detection boundary must be positive and finite.")
    return OneWayBoundaryContract(
        p_assets=p_value,
        n_groups=group_value,
        replicates=replicate_value,
        target_rank=rank_value,
        between_bulk_variance=between_variance,
        residual_bulk_variance=residual_variance,
        bulk_dimension=bulk_dimension,
        between_degrees_of_freedom=between_dof,
        within_degrees_of_freedom=within_dof,
        between_aspect_ratio=y_between,
        within_aspect_ratio=y_within,
        between_mean_square_bulk=c_between,
        within_mean_square_bulk=c_within,
        excess_eigenvalue_boundary=excess,
        population_eigenvalue_boundary=boundary,
    )


def _contract_bindings() -> dict[str, dict[str, Any]]:
    bindings = {}
    for relative in CONTRACT_BINDING_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Required v3 contract input is missing: {relative}"
            )
        bindings[relative] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return bindings


def validate_contract_bindings(manifest: Mapping[str, Any]) -> None:
    observed = manifest.get("contract_bindings")
    expected = _contract_bindings()
    if observed != expected:
        raise ValueError("V3 scientific contract binding mismatch.")


def _invariance_contract() -> dict[str, Any]:
    tolerances = InvarianceTolerances()
    payload = {
        "contract_id": INVARIANCE_CONTRACT_ID,
        "required_checks": list(REQUIRED_INVARIANCE_CHECKS),
        "evaluation_mu_roles": ["null", "power"],
        "candidate_matching": "exhaustive_minimum_mismatch_qmax2",
        "decisions_and_counts": "exact",
        "candidate_sources": "exact_and_fjs_only",
        "candidate_directions": "sign_invariant_squared_cosine",
        "tolerances": tolerances.to_dict(),
        "transformations": {
            "standardized_rescaling": {
                "reference": "column_zscore(Y, ddof=1)",
                "variant": ("column_zscore(Y * exp(linspace(-1.25,1.25,p)), ddof=1)"),
            },
            "deterministic_row_order": {
                "permutation": "reverse_rows",
                "labels": "carried_with_rows",
            },
            "asset_permutation": {
                "permutation": "PCG64(invariance_seed).permutation(p)",
                "comparison": "candidate_directions_mapped_back_to_canonical_assets",
            },
            "group_label_permutation": {
                "mapping": "sorted_unique_labels_to_reverse_sorted_unique_labels",
                "rows": "unchanged",
            },
        },
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def build_manifest_v3(
    *,
    profile_name: str,
    seed_base: int,
    limit_cells: int | None = None,
) -> dict[str, Any]:
    profile = manifest_profile(profile_name)
    cells = build_cells(profile, seed_base=seed_base, limit_cells=limit_cells)
    manifest_id = FULL_MANIFEST_ID if profile_name == "full" else SMOKE_MANIFEST_ID
    invariance_contract = _invariance_contract()
    cell_specs = []
    for cell in cells:
        boundary = independently_computed_oneway_boundary(
            p_assets=int(cell["p_assets"]),
            n_groups=int(cell["n_groups"]),
            replicates=int(cell["replicates"]),
        )
        boundary_payload = boundary.to_dict()
        boundary_payload["sha256"] = stable_sha256(boundary_payload)
        cell_specs.append(
            {
                **cell,
                "detection_boundary": boundary_payload,
                "power_mu": (
                    POWER_BOUNDARY_MULTIPLIER * boundary.population_eigenvalue_boundary
                ),
                "invariance_seed": int(cell["seed"]) * 1009 + 37,
                "invariance_contract_sha256": invariance_contract["sha256"],
                "nominal_size": 0.05,
                "null_upper_bound_max": 0.075,
                "power_detection_min": 0.80,
                "power_acceptance_min": 0.80,
                "power_gain_min": 0.50,
                "direction_squared_cosine_min": 0.80,
                "planted_component_accept_share_min": 0.90,
                "nuisance_component_accept_share_max": 0.10,
            }
        )

    cell_digest_map = {spec["cell_id"]: stable_sha256(spec) for spec in cell_specs}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "profile": profile_name,
        "purpose": (
            "V3 FJS M4 calibration generation with a closed-form cell boundary "
            "and hash-bound invariance reducer; no empirical or promotion claim."
        ),
        "promotion_allowed": False,
        "execution_readiness": {
            "scientific_definition_ready": True,
            "full_execution_ready": False,
            "aws_execution_authorized": False,
            "blockers": [
                "real_design_cell_manifest_not_yet_bound",
                "fresh_authoritative_aws_admission_required",
            ],
            "smoke_scope": (
                "boundary_invariance_and_checkpoint_integrity_only"
                if profile_name == "smoke"
                else None
            ),
        },
        "claim_boundary": {
            "mechanism_calibration_only": True,
            "empirical_claims_forbidden": True,
            "full_outcomes_unobserved_when_contract_frozen": True,
        },
        "predeclaration_contract": {
            "nominal_size": 0.05,
            "exact_binomial_interval": "95%",
            "power_boundary_multiplier": POWER_BOUNDARY_MULTIPLIER,
            "detection_and_acceptance_separate": True,
            "monotonic_power_required": True,
            "real_design_adequacy_required": True,
            "inject_mode": "between",
            "paired_trial_seeds_across_mu": True,
            "boundary_status": "cell_specific_closed_form_hash_bound",
            "boundary_contract_id": BOUNDARY_CONTRACT_ID,
            "finite_sample_margin_policy": (
                "delta_abs and calibrated delta_frac are separate operating "
                "margins and do not redefine the asymptotic phase transition"
            ),
        },
        "invariance_contract": invariance_contract,
        "contract_bindings": _contract_bindings(),
        "sweep": {
            "trials_null": profile.trials_null,
            "trials_alt": profile.trials_alt,
            "delta_frac_grid": list(profile.delta_frac_grid),
            "stability_grid": list(profile.stability_grid),
            "q_max": 2,
            "eps": 0.02,
            "p_assets": list(profile.p_assets),
            "n_groups": list(profile.n_groups),
            "replicates": list(profile.replicates),
            "delta_abs_grid": list(profile.delta_abs),
            "edge_modes": list(profile.edge_modes),
            "cells_total": len(cell_specs),
            "expected_cell_ids": [spec["cell_id"] for spec in cell_specs],
        },
        "artifacts": {
            "run_root_template": "reports/synthetic/calib/{run_id}",
            "smoke_outputs_must_be_temp": profile_name == "smoke",
            "canonical_promotion_requires_explicit_action": True,
        },
        "cells": cell_specs,
        "cell_digests": cell_digest_map,
    }
    manifest["expected_cell_set_digest"] = stable_sha256(
        [
            {"cell_id": key, "sha256": cell_digest_map[key]}
            for key in sorted(cell_digest_map)
        ]
    )
    manifest["manifest_digest"] = stable_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )
    return manifest
