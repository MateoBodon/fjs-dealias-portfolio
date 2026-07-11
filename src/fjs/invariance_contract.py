from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np

REQUIRED_INVARIANCE_CHECKS = (
    "standardized_rescaling",
    "deterministic_row_order",
    "asset_permutation",
    "group_label_permutation",
)

_NUMERIC_PRE_GATE_FIELDS = (
    "mp_edge_margin",
    "leakage_offcomp",
    "stability_eta_pass",
)
_NUMERIC_CANDIDATE_FIELDS = (
    "mu_hat",
    "lambda_hat",
    "z_plus",
    "stability_margin",
    "edge_margin",
    "buffer_margin",
    "target_energy",
    "off_component_ratio",
)
_EXACT_CANDIDATE_FIELDS = (
    "candidate_source",
    "admissible_root",
    "solver_used",
    "target_index",
    "pre_outlier_count",
    "edge_mode",
)


@dataclass(frozen=True)
class InvarianceTolerances:
    scalar_rtol: float = 1e-8
    scalar_atol: float = 1e-10
    direction_squared_cosine_min: float = 1.0 - 1e-10
    standardized_matrix_atol: float = 1e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.scalar_rtol) or self.scalar_rtol < 0.0:
            raise ValueError("scalar_rtol must be finite and non-negative.")
        if not math.isfinite(self.scalar_atol) or self.scalar_atol < 0.0:
            raise ValueError("scalar_atol must be finite and non-negative.")
        if not 0.0 <= self.direction_squared_cosine_min <= 1.0:
            raise ValueError("direction_squared_cosine_min must lie in [0, 1].")
        if (
            not math.isfinite(self.standardized_matrix_atol)
            or self.standardized_matrix_atol < 0.0
        ):
            raise ValueError(
                "standardized_matrix_atol must be finite and non-negative."
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "scalar_rtol": float(self.scalar_rtol),
            "scalar_atol": float(self.scalar_atol),
            "direction_squared_cosine_min": float(self.direction_squared_cosine_min),
            "standardized_matrix_atol": float(self.standardized_matrix_atol),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> InvarianceTolerances:
        return cls(
            scalar_rtol=float(payload["scalar_rtol"]),
            scalar_atol=float(payload["scalar_atol"]),
            direction_squared_cosine_min=float(payload["direction_squared_cosine_min"]),
            standardized_matrix_atol=float(payload["standardized_matrix_atol"]),
        )


def standardize_columns(observations: np.ndarray) -> np.ndarray:
    matrix = np.asarray(observations, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError(
            "observations must be a non-empty 2D matrix with at least two rows."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observations must contain only finite values.")
    centered = matrix - np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Every asset must have positive finite sample variance.")
    return np.asarray(centered / scale, dtype=np.float64)


def deterministic_rescaling(asset_count: int) -> np.ndarray:
    if int(asset_count) <= 0:
        raise ValueError("asset_count must be positive.")
    return np.exp(np.linspace(-1.25, 1.25, int(asset_count), dtype=np.float64))


def deterministic_row_permutation(row_count: int) -> np.ndarray:
    if int(row_count) <= 0:
        raise ValueError("row_count must be positive.")
    return np.arange(int(row_count) - 1, -1, -1, dtype=np.intp)


def deterministic_asset_permutation(asset_count: int, seed: int) -> np.ndarray:
    if int(asset_count) <= 0:
        raise ValueError("asset_count must be positive.")
    return np.asarray(
        np.random.default_rng(int(seed)).permutation(int(asset_count)),
        dtype=np.intp,
    )


def deterministic_group_label_permutation(groups: np.ndarray) -> np.ndarray:
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array.")
    unique, inverse = np.unique(labels, return_inverse=True)
    return np.asarray(unique[::-1][inverse], dtype=labels.dtype)


def _parse_source_counts(raw: Any) -> dict[str, int]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        counts = {str(key): int(value) for key, value in raw.items()}
    elif isinstance(raw, str):
        counts: dict[str, int] = {}
        for token in raw.split("|"):
            source, separator, value = token.partition(":")
            if not separator or not source:
                raise ValueError(f"Invalid candidate-source count token: {token!r}")
            counts[source] = int(value)
    else:
        raise TypeError("candidate source counts must be a mapping or stable string.")
    if any(value < 0 for value in counts.values()):
        raise ValueError("candidate source counts must be non-negative.")
    return {source: counts[source] for source in sorted(counts) if counts[source]}


def _normalise_direction(values: Any, *, indexer: np.ndarray | None) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("candidate eigvec must be a finite non-empty vector.")
    if indexer is not None:
        permutation = np.asarray(indexer, dtype=np.intp)
        if permutation.shape != vector.shape or not np.array_equal(
            np.sort(permutation), np.arange(vector.size)
        ):
            raise ValueError("direction_indexer must be a full asset permutation.")
        vector = vector[permutation]
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("candidate eigvec must have positive finite norm.")
    return np.asarray(vector / norm, dtype=np.float64).tolist()


def _optional_scalar(candidate: Mapping[str, Any], field: str) -> float | None:
    value = candidate.get(field)
    if value is None:
        return None
    return float(value)


def build_detector_signature(
    pre_gate: Mapping[str, Any],
    accepted_candidates: Sequence[Mapping[str, Any]],
    *,
    direction_indexer: np.ndarray | None = None,
    require_fjs_only: bool = True,
) -> dict[str, Any]:
    pre_count = int(pre_gate.get("raw_outliers_found", 0))
    if pre_count < 0:
        raise ValueError("raw_outliers_found must be non-negative.")
    pre_sources = _parse_source_counts(pre_gate.get("candidate_sources", ""))
    if sum(pre_sources.values()) != pre_count:
        raise ValueError("pre-gate candidate-source counts do not match raw count.")

    candidates = []
    accepted_counter: Counter[str] = Counter()
    for candidate in accepted_candidates:
        source = str(candidate.get("candidate_source", "")).strip().lower()
        if source not in {"fjs", "coarse", "oracle", "sham"}:
            raise ValueError(f"Unknown candidate_source {source!r}.")
        accepted_counter[source] += 1
        signature = {field: candidate.get(field) for field in _EXACT_CANDIDATE_FIELDS}
        signature.update(
            {
                field: _optional_scalar(candidate, field)
                for field in _NUMERIC_CANDIDATE_FIELDS
            }
        )
        signature["direction"] = _normalise_direction(
            candidate.get("eigvec"), indexer=direction_indexer
        )
        if candidate.get("a") is None:
            signature["design_direction"] = None
        else:
            signature["design_direction"] = _normalise_direction(
                candidate.get("a"), indexer=None
            )
        candidates.append(signature)

    accepted_sources = {
        source: int(accepted_counter[source]) for source in sorted(accepted_counter)
    }
    if require_fjs_only and (
        any(source != "fjs" for source in pre_sources)
        or any(source != "fjs" for source in accepted_sources)
    ):
        raise ValueError("The FJS invariance arm may contain only fjs candidates.")

    return {
        "detected": bool(pre_count > 0),
        "accepted": bool(candidates),
        "pre_candidate_count": pre_count,
        "accepted_candidate_count": len(candidates),
        "pre_source_counts": pre_sources,
        "accepted_source_counts": accepted_sources,
        "pre_bracket_status": str(pre_gate.get("bracket_status", "none")),
        "pre_gate_numeric": {
            field: _optional_scalar(pre_gate, field)
            for field in _NUMERIC_PRE_GATE_FIELDS
        },
        "candidates": candidates,
    }


def _scalar_close(
    left: float | None,
    right: float | None,
    tolerances: InvarianceTolerances,
) -> tuple[bool, float]:
    if left is None or right is None:
        return left is right, 0.0 if left is right else math.inf
    left_value = float(left)
    right_value = float(right)
    if math.isnan(left_value) or math.isnan(right_value):
        return math.isnan(left_value) and math.isnan(right_value), 0.0
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return left_value == right_value, 0.0 if left_value == right_value else math.inf
    difference = abs(left_value - right_value)
    limit = tolerances.scalar_atol + tolerances.scalar_rtol * max(
        abs(left_value), abs(right_value)
    )
    return difference <= limit, difference


def _squared_cosine(left: Any, right: Any) -> float:
    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    if left_vector.shape != right_vector.shape or left_vector.ndim != 1:
        return float("nan")
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return float("nan")
    cosine = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    return float(min(1.0, max(0.0, cosine * cosine)))


def _candidate_comparison(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
    tolerances: InvarianceTolerances,
) -> tuple[list[str], float, float]:
    reasons = []
    max_scalar_error = 0.0
    for field in _EXACT_CANDIDATE_FIELDS:
        if reference.get(field) != observed.get(field):
            reasons.append(f"candidate_{field}_mismatch")
    for field in _NUMERIC_CANDIDATE_FIELDS:
        close, error = _scalar_close(
            reference.get(field), observed.get(field), tolerances
        )
        max_scalar_error = max(max_scalar_error, error)
        if not close:
            reasons.append(f"candidate_{field}_outside_tolerance")

    direction_cosine = _squared_cosine(
        reference.get("direction"), observed.get("direction")
    )
    if (
        not math.isfinite(direction_cosine)
        or direction_cosine < tolerances.direction_squared_cosine_min
    ):
        reasons.append("candidate_direction_outside_tolerance")
    design_reference = reference.get("design_direction")
    design_observed = observed.get("design_direction")
    if design_reference is None or design_observed is None:
        if design_reference is not design_observed:
            reasons.append("candidate_design_direction_mismatch")
        design_cosine = 1.0 if design_reference is design_observed else float("nan")
    else:
        design_cosine = _squared_cosine(design_reference, design_observed)
        if (
            not math.isfinite(design_cosine)
            or design_cosine < tolerances.direction_squared_cosine_min
        ):
            reasons.append("candidate_design_direction_outside_tolerance")
    return reasons, max_scalar_error, min(direction_cosine, design_cosine)


def _compare_signature_pair(
    reference: Mapping[str, Any],
    observed: Mapping[str, Any],
    tolerances: InvarianceTolerances,
) -> dict[str, Any]:
    reasons = []
    exact_fields = (
        "detected",
        "accepted",
        "pre_candidate_count",
        "accepted_candidate_count",
        "pre_source_counts",
        "accepted_source_counts",
        "pre_bracket_status",
    )
    for field in exact_fields:
        if reference.get(field) != observed.get(field):
            reasons.append(f"{field}_mismatch")

    max_scalar_error = 0.0
    reference_pre = reference.get("pre_gate_numeric", {})
    observed_pre = observed.get("pre_gate_numeric", {})
    for field in _NUMERIC_PRE_GATE_FIELDS:
        close, error = _scalar_close(
            reference_pre.get(field), observed_pre.get(field), tolerances
        )
        max_scalar_error = max(max_scalar_error, error)
        if not close:
            reasons.append(f"pre_gate_{field}_outside_tolerance")

    reference_candidates = list(reference.get("candidates", []))
    observed_candidates = list(observed.get("candidates", []))
    min_direction_cosine = 1.0
    if len(reference_candidates) == len(observed_candidates):
        best: tuple[int, float, float, list[str]] | None = None
        for ordering in permutations(observed_candidates):
            candidate_reasons: list[str] = []
            candidate_error = 0.0
            candidate_cosine = 1.0
            for index, (left, right) in enumerate(
                zip(reference_candidates, ordering, strict=True)
            ):
                local_reasons, local_error, local_cosine = _candidate_comparison(
                    left, right, tolerances
                )
                candidate_reasons.extend(
                    f"candidate_{index}:{reason}" for reason in local_reasons
                )
                candidate_error = max(candidate_error, local_error)
                candidate_cosine = min(candidate_cosine, local_cosine)
            score = (
                len(candidate_reasons),
                candidate_error,
                -candidate_cosine,
                candidate_reasons,
            )
            if best is None or score[:3] < best[:3]:
                best = score
        if best is not None:
            reasons.extend(best[3])
            max_scalar_error = max(max_scalar_error, best[1])
            min_direction_cosine = -best[2]
    else:
        min_direction_cosine = float("nan")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "max_scalar_abs_error": float(max_scalar_error),
        "min_direction_squared_cosine": float(min_direction_cosine),
    }


def assess_invariance(
    comparisons: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    tolerances: InvarianceTolerances | None = None,
) -> dict[str, Any]:
    tolerance = tolerances or InvarianceTolerances()
    observed_checks = set(comparisons)
    required_checks = set(REQUIRED_INVARIANCE_CHECKS)
    if observed_checks != required_checks:
        missing = sorted(required_checks - observed_checks)
        extra = sorted(observed_checks - required_checks)
        raise ValueError(
            f"Invariance comparison set mismatch; missing={missing}, extra={extra}."
        )
    check_results = {
        check: _compare_signature_pair(*comparisons[check], tolerance)
        for check in REQUIRED_INVARIANCE_CHECKS
    }
    failed = [
        check
        for check in REQUIRED_INVARIANCE_CHECKS
        if not check_results[check]["passed"]
    ]
    return {
        "contract_id": "fjs-decision-invariance-v1",
        "passed": not failed,
        "failed_checks": failed,
        "tolerances": tolerance.to_dict(),
        "checks": check_results,
    }
