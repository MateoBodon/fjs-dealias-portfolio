from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from fjs.dealias import Detection, _default_design
from fjs.detector_contract import require_candidate_source
from fjs.mp import admissible_m_from_lambda, mp_edge, t_vec
from fjs.overlay import OverlayConfig, apply_overlay
from fjs.reference_oracle import (
    BalancedReferenceDesign,
    ReferenceContractError,
    admissible_root_reference,
    oracle_and_sham_candidates,
    outlier_reference,
    require_reference_close,
    spectral_reconstruction_reference,
    t_vector_reference,
    upper_edge_reference,
    z_prime_reference,
    z_reference,
)

pytestmark = pytest.mark.unit


def _scalar_design() -> BalancedReferenceDesign:
    return BalancedReferenceDesign(
        a=np.array([1.0], dtype=np.float64),
        bulk_scales=np.array([2.0], dtype=np.float64),
        degrees_of_freedom=np.array([20.0], dtype=np.float64),
        bulk_dimension=5.0,
        component_scales=np.array([1.0], dtype=np.float64),
        strata_by_component=((0,),),
    )


def _oneway_component_design() -> BalancedReferenceDesign:
    # I=5, J=2, p=5 and one planted spike direction, hence N=p-L=4.
    root_2041 = math.sqrt(2041.0)
    return BalancedReferenceDesign(
        a=np.array([21.0 / root_2041, -40.0 / root_2041], dtype=np.float64),
        bulk_scales=np.array([3.0, 1.0], dtype=np.float64),
        degrees_of_freedom=np.array([4.0, 5.0], dtype=np.float64),
        bulk_dimension=4.0,
        component_scales=np.array([2.0, 1.0], dtype=np.float64),
        # Group effect sees stratum 0; observational noise sees both strata.
        strata_by_component=((0,), (0, 1)),
    )


def test_reference_oracle_has_no_production_fjs_imports() -> None:
    import fjs.reference_oracle as reference_oracle

    source_path = Path(cast(str, reference_oracle.__file__))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert "fjs" not in imported_roots


def test_scalar_mp_edge_root_component_and_boundary_have_exact_values() -> None:
    design = _scalar_design()
    edge = upper_edge_reference(design)

    assert edge.m == pytest.approx(-2.0 / 3.0, abs=1e-10)
    assert edge.value == pytest.approx(4.5, abs=1e-10)
    assert edge.curvature > 0.0

    lambda_hat = 5.0
    m_value = admissible_root_reference(lambda_hat, design)
    assert m_value == pytest.approx(-0.4, abs=1e-10)
    assert z_reference(m_value, design) == pytest.approx(lambda_hat, abs=1e-10)
    assert z_prime_reference(m_value, design) > 0.0

    outlier = outlier_reference(lambda_hat, design, target_component=0)
    assert outlier.t_values == pytest.approx(np.array([1.25]), abs=1e-10)
    assert outlier.mu_hat == pytest.approx(4.0, abs=1e-10)

    t_at_edge = t_vector_reference(edge.m, design)[0]
    spike_excess_boundary = -1.0 / (edge.m * t_at_edge)
    assert spike_excess_boundary == pytest.approx(1.0, abs=1e-10)
    assert 2.0 + spike_excess_boundary == pytest.approx(3.0, abs=1e-10)


def test_two_stratum_oneway_reference_isolates_between_component_exactly() -> None:
    design = _oneway_component_design()
    root_2041 = math.sqrt(2041.0)
    expected_m = -root_2041 / 168.0
    expected_lambda = 1176.0 / (5.0 * root_2041)

    assert np.linalg.norm(design.a) == pytest.approx(1.0, abs=1e-12)
    assert z_reference(expected_m, design) == pytest.approx(expected_lambda, abs=1e-10)
    assert z_prime_reference(expected_m, design) > 0.0

    edge = upper_edge_reference(design)
    assert edge.m == pytest.approx(-0.35393240222815614, abs=1e-9)
    assert edge.value == pytest.approx(4.871008798276616, abs=1e-9)
    assert expected_lambda > edge.value

    outlier = outlier_reference(
        expected_lambda,
        design,
        target_component=0,
        off_component_tolerance=1e-9,
    )
    assert outlier.m == pytest.approx(expected_m, abs=1e-9)
    assert outlier.t_values[0] == pytest.approx(336.0 / (5.0 * root_2041), abs=1e-9)
    assert outlier.t_values[1] == pytest.approx(0.0, abs=1e-10)
    assert outlier.mu_hat == pytest.approx(3.5, abs=1e-9)


def test_reference_is_homogeneous_and_invariant_to_stratum_order() -> None:
    design = _oneway_component_design()
    m_value = -math.sqrt(2041.0) / 168.0
    lambda_value = z_reference(m_value, design)
    scale = 1.7
    scaled = BalancedReferenceDesign(
        a=scale * design.a,
        bulk_scales=design.bulk_scales,
        degrees_of_freedom=design.degrees_of_freedom,
        bulk_dimension=design.bulk_dimension,
        component_scales=design.component_scales,
        strata_by_component=design.strata_by_component,
    )
    assert z_reference(m_value / scale, scaled) == pytest.approx(
        scale * lambda_value, abs=1e-10
    )
    assert t_vector_reference(m_value / scale, scaled) == pytest.approx(
        scale * t_vector_reference(m_value, design), abs=1e-10
    )

    permutation = np.array([1, 0])
    permuted = BalancedReferenceDesign(
        a=design.a[permutation],
        bulk_scales=design.bulk_scales[permutation],
        degrees_of_freedom=design.degrees_of_freedom[permutation],
        bulk_dimension=design.bulk_dimension,
        component_scales=design.component_scales,
        strata_by_component=((1,), (1, 0)),
    )
    assert z_reference(m_value, permuted) == pytest.approx(lambda_value, abs=1e-10)
    assert t_vector_reference(m_value, permuted) == pytest.approx(
        t_vector_reference(m_value, design), abs=1e-10
    )


def test_oracle_and_sham_are_deterministic_orthogonal_and_magnitude_matched() -> None:
    oracle, sham = oracle_and_sham_candidates(
        np.array([3.0, 4.0, 0.0]),
        mu_hat=4.0,
        lambda_hat=5.0,
    )

    assert oracle.candidate_source == "oracle"
    assert sham.candidate_source == "sham"
    assert require_candidate_source(oracle.as_mapping()) == "oracle"
    assert require_candidate_source(sham.as_mapping()) == "sham"
    assert oracle.mu_hat == sham.mu_hat == 4.0
    assert oracle.lambda_hat == sham.lambda_hat == 5.0
    assert np.linalg.norm(oracle.eigvec) == pytest.approx(1.0)
    assert np.linalg.norm(sham.eigvec) == pytest.approx(1.0)
    assert float(np.dot(oracle.eigvec, sham.eigvec)) == pytest.approx(0.0, abs=1e-12)
    assert sham.eigvec == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=1e-12)


def test_reference_reconstruction_sets_eigenpair_and_preserves_orthogonal_block() -> (
    None
):
    baseline = np.array(
        [[2.0, 1.0, 0.2], [1.0, 3.0, -0.1], [0.2, -0.1, 1.5]],
        dtype=np.float64,
    )
    direction = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    reconstructed = spectral_reconstruction_reference(baseline, direction, 4.0)
    projector = np.eye(3) - np.outer(direction, direction)

    assert reconstructed @ direction == pytest.approx(4.0 * direction, abs=1e-10)
    assert projector @ reconstructed @ projector == pytest.approx(
        projector @ baseline @ projector, abs=1e-10
    )
    assert spectral_reconstruction_reference(
        baseline, -direction, 4.0
    ) == pytest.approx(reconstructed, abs=1e-12)


def test_production_default_oneway_design_matches_reference() -> None:
    production = _default_design({"I": 5, "J": 2, "n": 10, "p": 5})
    require_reference_close("bulk_dimension", production["N"], 4.0)
    assert production["order"] == [[1], [1, 2]]


def test_production_explicit_cs_map_matches_two_stratum_reference() -> None:
    design = _oneway_component_design()
    edge = upper_edge_reference(design)
    lambda_hat = 1176.0 / (5.0 * math.sqrt(2041.0))
    production_edge = mp_edge(
        design.a,
        np.ones(2, dtype=np.float64),
        design.degrees_of_freedom,
        design.bulk_dimension,
        Cs=design.bulk_scales,
    )
    production_root = admissible_m_from_lambda(
        lambda_hat,
        design.a,
        np.ones(2, dtype=np.float64),
        design.degrees_of_freedom,
        design.bulk_dimension,
        Cs=design.bulk_scales,
    )
    production_t = t_vec(
        lambda_hat,
        design.a,
        np.ones(2, dtype=np.float64),
        design.degrees_of_freedom,
        design.bulk_dimension,
        design.component_scales,
        [[1], [1, 2]],
        Cs=design.bulk_scales,
    )
    require_reference_close("upper_edge", production_edge, edge.value)
    require_reference_close(
        "admissible_root",
        production_root,
        admissible_root_reference(lambda_hat, design),
    )
    require_reference_close(
        "t_vector", production_t, t_vector_reference(production_root, design)
    )


def test_production_reconstruction_matches_reference_for_non_aligned_baseline() -> None:
    baseline = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    expected = spectral_reconstruction_reference(baseline, direction, 4.0)
    candidate = cast(
        Detection,
        cast(
            Any,
            {
                "candidate_source": "oracle",
                "mu_hat": 4.0,
                "lambda_hat": 5.0,
                "eigvec": direction,
            },
        ),
    )
    observed = apply_overlay(
        baseline,
        [candidate],
        baseline_covariance=baseline,
        config=OverlayConfig(q_max=1),
    )
    require_reference_close("spectral_reconstruction", observed, expected)


def test_reference_contract_fails_loud_on_invalid_inputs() -> None:
    design = _scalar_design()
    with pytest.raises(ReferenceContractError, match="strictly above"):
        admissible_root_reference(upper_edge_reference(design).value, design)
    with pytest.raises(ReferenceContractError, match="square"):
        spectral_reconstruction_reference(np.ones((2, 3)), np.ones(2), 1.0)
    with pytest.raises(ReferenceContractError, match="at least two"):
        oracle_and_sham_candidates(np.ones(1), mu_hat=1.0, lambda_hat=1.0)
