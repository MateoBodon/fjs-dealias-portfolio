from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from fjs.dealias import Detection, _default_design
from fjs.detector_contract import assess_power_curve
from fjs.mp import admissible_m_from_lambda, mp_edge, t_vec
from fjs.overlay import OverlayConfig, apply_overlay
from fjs.reference_oracle import (
    BalancedReferenceDesign,
    ReferenceContractError,
    admissible_root_reference,
    require_reference_close,
    spectral_reconstruction_reference,
    t_vector_reference,
    upper_edge_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POWER_CURVE = (
    REPO_ROOT
    / "docs/artifacts/detector-contract-reference/ticket24_week_full_fix/curve.csv"
)


def _oneway_reference_design() -> BalancedReferenceDesign:
    root_2041 = math.sqrt(2041.0)
    return BalancedReferenceDesign(
        a=np.array([21.0 / root_2041, -40.0 / root_2041], dtype=np.float64),
        bulk_scales=np.array([3.0, 1.0], dtype=np.float64),
        degrees_of_freedom=np.array([4.0, 5.0], dtype=np.float64),
        bulk_dimension=4.0,
        component_scales=np.array([2.0, 1.0], dtype=np.float64),
        strata_by_component=((0,), (0, 1)),
    )


def _issue(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def audit_reference_contract(power_curve: Path) -> list[dict[str, str]]:
    """Return deterministic stop-line issues; an empty list is the pass condition."""

    issues: list[dict[str, str]] = []

    production_design = _default_design({"I": 5, "J": 2, "n": 10, "p": 5})
    if float(production_design["N"]) != 4.0:
        issues.append(
            _issue(
                "oneway_bulk_dimension_mismatch",
                "Expected N=p-L=4 for the rank-one p=5 reference; "
                f"observed N={float(production_design['N']):g}.",
            )
        )
    if production_design["order"] != [[1], [1, 2]]:
        issues.append(
            _issue(
                "oneway_inclusion_order_mismatch",
                "Expected group strata [[1]] and residual strata [[1,2]]; "
                f"observed {production_design['order']!r}.",
            )
        )

    design = _oneway_reference_design()
    lambda_hat = 1176.0 / (5.0 * math.sqrt(2041.0))
    try:
        production_edge = mp_edge(
            design.a,
            np.ones(2, dtype=np.float64),
            design.degrees_of_freedom,
            design.bulk_dimension,
            Cs=design.bulk_scales,
        )
        require_reference_close(
            "explicit-Cs upper edge",
            production_edge,
            upper_edge_reference(design).value,
        )
        production_root = admissible_m_from_lambda(
            lambda_hat,
            design.a,
            np.ones(2, dtype=np.float64),
            design.degrees_of_freedom,
            design.bulk_dimension,
            Cs=design.bulk_scales,
        )
        require_reference_close(
            "explicit-Cs admissible root",
            production_root,
            admissible_root_reference(lambda_hat, design),
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
        require_reference_close(
            "explicit-Cs t-vector",
            production_t,
            t_vector_reference(production_root, design),
        )
    except (ReferenceContractError, RuntimeError, ValueError) as exc:
        issues.append(_issue("explicit_cs_mp_map_mismatch", str(exc)))

    baseline = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    expected_reconstruction = spectral_reconstruction_reference(
        baseline, direction, 4.0
    )
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
    try:
        production_reconstruction = apply_overlay(
            baseline,
            [candidate],
            baseline_covariance=baseline,
            config=OverlayConfig(q_max=1),
        )
        require_reference_close(
            "spectral reconstruction",
            production_reconstruction,
            expected_reconstruction,
        )
    except (ReferenceContractError, RuntimeError, ValueError) as exc:
        issues.append(_issue("spectral_reconstruction_mismatch", str(exc)))

    try:
        with power_curve.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assess_power_curve(rows, expected_inject_mode="between")
    except (OSError, ValueError) as exc:
        issues.append(_issue("target_power_provenance_invalid", str(exc)))

    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-loud deterministic FJS reference and target-power gate."
    )
    parser.add_argument(
        "--power-curve",
        type=Path,
        default=DEFAULT_POWER_CURVE,
        help="Power curve that must carry inject_mode=between provenance.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues = audit_reference_contract(args.power_curve)
    payload = {
        "schema_version": "fjs-reference-gate/v1",
        "status": "passed" if not issues else "blocked",
        "power_curve": str(args.power_curve),
        "issue_count": len(issues),
        "issues": issues,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
