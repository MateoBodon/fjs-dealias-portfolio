from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from experiments.synthetic.harness_utils import simulate_panel
from tools.fjs_m4_contract import file_sha256
from tools.fjs_m4_contract_v3 import (
    BOUNDARY_CONTRACT_ID,
    FULL_MANIFEST_ID,
    INVARIANCE_CONTRACT_ID,
    OneWayBoundaryContract,
    independently_computed_oneway_boundary,
    validate_contract_bindings,
)

from fjs.invariance_contract import (
    REQUIRED_INVARIANCE_CHECKS,
    InvarianceTolerances,
    assess_invariance,
    build_detector_signature,
    deterministic_rescaling,
    standardize_columns,
)
from fjs.overlay import OverlayConfig
from fjs.reference_oracle import (
    BalancedReferenceDesign,
    t_vector_reference,
    upper_edge_reference,
    z_prime_reference,
)
from tools import freeze_fjs_m4_manifest_v3, run_fjs_calibration_manifest


def _edge_isolating_reference_design(
    boundary: OneWayBoundaryContract,
) -> tuple[BalancedReferenceDesign, float]:
    contract = boundary
    y_between = float(contract.between_aspect_ratio)
    y_within = float(contract.within_aspect_ratio)
    c_between = float(contract.between_mean_square_bulk)
    c_within = float(contract.within_mean_square_bulk)
    q_between = -1.0 / math.sqrt(y_between + y_within * (c_within / c_between) ** 2)
    q_within = -(c_within / c_between) * q_between
    x_between = q_between / (1.0 - y_between * q_between)
    x_within = q_within / (1.0 - y_within * q_within)
    raw_a = np.array([-x_between / c_between, -x_within / c_within], dtype=np.float64)
    norm = float(np.linalg.norm(raw_a))
    design = BalancedReferenceDesign(
        a=raw_a / norm,
        bulk_scales=np.array([c_between, c_within], dtype=np.float64),
        degrees_of_freedom=np.array(
            [
                contract.between_degrees_of_freedom,
                contract.within_degrees_of_freedom,
            ],
            dtype=np.float64,
        ),
        bulk_dimension=float(contract.bulk_dimension),
        component_scales=np.array([contract.replicates, 1.0], dtype=np.float64),
        strata_by_component=((0,), (0, 1)),
    )
    return design, -norm


@pytest.mark.unit
@pytest.mark.parametrize(
    ("p_assets", "n_groups", "replicates"),
    [
        (p_assets, n_groups, replicates)
        for p_assets in (64, 80, 96, 128, 160, 188, 200)
        for n_groups in (36, 60, 80)
        for replicates in (12, 16, 20)
    ],
)
def test_closed_form_boundary_matches_independent_edge_oracle(
    p_assets: int, n_groups: int, replicates: int
) -> None:
    boundary = independently_computed_oneway_boundary(
        p_assets=p_assets,
        n_groups=n_groups,
        replicates=replicates,
    )
    design, edge_m = _edge_isolating_reference_design(boundary)
    edge = upper_edge_reference(design)
    t_values = t_vector_reference(edge_m, design)
    observed_boundary = boundary.between_bulk_variance - 1.0 / (
        edge_m * float(t_values[0])
    )

    assert edge.m == pytest.approx(edge_m, rel=1e-9, abs=1e-10)
    assert z_prime_reference(edge_m, design) == pytest.approx(0.0, abs=1e-9)
    assert t_values[1] == pytest.approx(0.0, abs=1e-9)
    assert observed_boundary == pytest.approx(
        boundary.population_eigenvalue_boundary, rel=1e-9, abs=1e-10
    )


@pytest.mark.unit
def test_frozen_generator_boundary_reduces_to_declared_formula() -> None:
    boundary = independently_computed_oneway_boundary(
        p_assets=80,
        n_groups=36,
        replicates=12,
    )
    expected = math.sqrt(79.0 / 35.0 + 79.0 / (36.0 * 11.0)) / 12.0
    assert boundary.population_eigenvalue_boundary == pytest.approx(expected)
    assert boundary.to_dict()["contract_id"] == BOUNDARY_CONTRACT_ID


@pytest.mark.unit
def test_v3_manifest_is_byte_stable_and_v2_files_are_untouched(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    v2_paths = (
        root / "calibration/manifests/fjs_m4_full_target_between_v2.json",
        root / "calibration/manifests/fjs_m4_smoke_target_between_v2.json",
    )
    v2_before = {path: file_sha256(path) for path in v2_paths}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    freeze_fjs_m4_manifest_v3.main(["--profile", "full", "--out", str(first)])
    freeze_fjs_m4_manifest_v3.main(["--profile", "full", "--out", str(second)])
    payload = json.loads(first.read_text(encoding="utf-8"))

    assert first.read_bytes() == second.read_bytes()
    assert payload["manifest_id"] == FULL_MANIFEST_ID
    assert len(payload["cells"]) == 252
    assert payload["predeclaration_contract"]["boundary_status"] == (
        "cell_specific_closed_form_hash_bound"
    )
    assert payload["invariance_contract"]["contract_id"] == INVARIANCE_CONTRACT_ID
    assert payload["invariance_contract"]["required_checks"] == list(
        REQUIRED_INVARIANCE_CHECKS
    )
    assert payload["execution_readiness"]["full_execution_ready"] is False
    assert payload["execution_readiness"]["aws_execution_authorized"] is False
    assert payload["execution_readiness"]["blockers"] == [
        "real_design_cell_manifest_not_yet_bound",
        "trusted_route_admission_required",
        "fresh_authoritative_aws_admission_required",
    ]
    assert len({cell["power_mu"] for cell in payload["cells"]}) > 1
    validate_contract_bindings(payload)
    assert run_fjs_calibration_manifest._load_manifest(first) == payload
    assert {path: file_sha256(path) for path in v2_paths} == v2_before


@pytest.mark.unit
def test_v3_default_manifest_paths_use_canonical_underscore_namespace() -> None:
    assert freeze_fjs_m4_manifest_v3.default_manifest_path(FULL_MANIFEST_ID).name == (
        "fjs_m4_full_target_between_v3.json"
    )


@pytest.mark.unit
def test_v3_runner_gate_keeps_external_stop_lines_explicit(tmp_path: Path) -> None:
    manifest_path = tmp_path / "smoke.json"
    freeze_fjs_m4_manifest_v3.main(["--profile", "smoke", "--out", str(manifest_path)])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cell = manifest["cells"][0]
    metrics = {
        "null_detection_ci_low": 0.04,
        "null_detection_ci_high": 0.06,
        "strong_detection_rate": 0.90,
        "strong_acceptance_rate": 0.85,
        "detection_gain": 0.85,
        "monotone_detection": True,
        "monotone_acceptance": True,
        "acceptance_exceeds_detection": False,
        "direction_squared_cosine_mean": 0.90,
        "planted_component_accept_share": 0.95,
        "nuisance_component_accept_share": 0.05,
        "non_fjs_accept_count": 0,
    }
    invariance = {
        "passed": True,
        "evaluations": [
            {"role": "null", "assessment": {"passed": True}},
            {"role": "power", "assessment": {"passed": True}},
        ],
    }
    assessment = run_fjs_calibration_manifest._assess_cell_gates(
        manifest=manifest,
        cell=cell,
        gate_metrics=metrics,
        invariance=invariance,
    )
    assert assessment["local_scientific_gate_pass"] is True
    assert assessment["full_detector_gate_pass"] is False
    assert assessment["full_detector_gate_blockers"] == [
        "real_design_cell_manifest_not_yet_bound",
        "trusted_route_admission_required",
        "fresh_authoritative_aws_admission_required",
    ]


def _candidate(direction: np.ndarray) -> dict[str, object]:
    return {
        "candidate_source": "fjs",
        "mu_hat": 4.0,
        "lambda_hat": 5.0,
        "z_plus": 3.0,
        "stability_margin": 1.0,
        "edge_margin": 2.0,
        "buffer_margin": 1.5,
        "target_energy": 4.0,
        "off_component_ratio": 0.01,
        "admissible_root": True,
        "solver_used": "rootfind",
        "target_index": 0,
        "pre_outlier_count": 1,
        "edge_mode": "scm",
        "eigvec": direction,
        "a": [0.5, -math.sqrt(0.75)],
    }


def _pre_gate() -> dict[str, object]:
    return {
        "raw_outliers_found": 1,
        "candidate_sources": "fjs:1",
        "bracket_status": "rootfind",
        "mp_edge_margin": 2.0,
        "leakage_offcomp": 0.01,
        "stability_eta_pass": 1.0,
    }


@pytest.mark.unit
def test_invariance_reducer_is_sign_invariant_and_fails_on_decision_drift() -> None:
    direction = np.array([1.0, 2.0, -1.0], dtype=np.float64)
    reference = build_detector_signature(_pre_gate(), [_candidate(direction)])
    sign_flipped = build_detector_signature(_pre_gate(), [_candidate(-direction)])
    comparisons = {
        check: (reference, sign_flipped) for check in REQUIRED_INVARIANCE_CHECKS
    }
    assessment = assess_invariance(comparisons)
    assert assessment["passed"] is True
    assert all(result["passed"] for result in assessment["checks"].values())

    drifted = json.loads(json.dumps(sign_flipped))
    drifted["accepted"] = False
    failed = assess_invariance(
        {
            **comparisons,
            "asset_permutation": (reference, drifted),
        }
    )
    assert failed["passed"] is False
    assert failed["failed_checks"] == ["asset_permutation"]


@pytest.mark.unit
def test_standardized_rescaling_contract_removes_positive_asset_scales() -> None:
    rng = np.random.default_rng(7)
    observations = rng.normal(size=(40, 8))
    scale = deterministic_rescaling(observations.shape[1])
    baseline = standardize_columns(observations)
    rescaled = standardize_columns(observations * scale)
    assert (
        np.max(np.abs(baseline - rescaled))
        <= InvarianceTolerances().standardized_matrix_atol
    )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_real_kernel_mechanism_fixture_passes_all_four_invariances() -> None:
    observations, groups, _, _ = simulate_panel(
        np.random.default_rng(20260710),
        n_assets=10,
        n_groups=60,
        replicates=3,
        spike_strength=6.0,
        noise_variance=1.0,
        signal_to_noise=0.35,
        return_dirs=True,
    )
    config = OverlayConfig(
        q_max=2,
        delta=0.3,
        delta_frac=0.01,
        eps=0.02,
        stability_eta_deg=0.4,
        a_grid=120,
        require_isolated=True,
        off_component_cap=0.3,
        edge_mode="scm",
    )

    assessment = run_fjs_calibration_manifest._evaluate_invariance_for_panel(
        observations,
        groups,
        config=config,
        invariance_seed=813,
        tolerances=InvarianceTolerances(),
    )
    assert assessment["passed"] is True
