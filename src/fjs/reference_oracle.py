from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray


class ReferenceContractError(ValueError):
    """Raised when a detector value cannot satisfy the frozen reference contract."""


@dataclass(frozen=True)
class BalancedReferenceDesign:
    """Parameters for the balanced-design equations in FJS Proposition 5.4.

    This module deliberately does not import the production MP, de-aliasing, or
    overlay implementations.  ``bulk_scales`` are the paper's ``C_s`` values,
    ``bulk_dimension`` is ``N = p - L``, and ``strata_by_component`` encodes
    the inclusion relation ``s \\preceq r`` directly.
    """

    a: NDArray[np.float64]
    bulk_scales: NDArray[np.float64]
    degrees_of_freedom: NDArray[np.float64]
    bulk_dimension: float
    component_scales: NDArray[np.float64]
    strata_by_component: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        a = _one_dimensional_finite("a", self.a)
        bulk_scales = _one_dimensional_finite("bulk_scales", self.bulk_scales)
        dof = _one_dimensional_finite("degrees_of_freedom", self.degrees_of_freedom)
        component_scales = _one_dimensional_finite(
            "component_scales", self.component_scales
        )
        if not (a.shape == bulk_scales.shape == dof.shape):
            raise ReferenceContractError(
                "a, bulk_scales, and degrees_of_freedom must have identical shapes."
            )
        if not np.any(np.abs(a) > 0.0):
            raise ReferenceContractError("a must contain a non-zero coefficient.")
        if np.any(bulk_scales < 0.0) or not np.any(bulk_scales > 0.0):
            raise ReferenceContractError(
                "bulk_scales must be non-negative with at least one positive value."
            )
        if np.any(dof <= 0.0):
            raise ReferenceContractError("degrees_of_freedom must be positive.")
        if not np.isfinite(self.bulk_dimension) or self.bulk_dimension <= 0.0:
            raise ReferenceContractError("bulk_dimension N=p-L must be positive.")
        if len(self.strata_by_component) != component_scales.size:
            raise ReferenceContractError(
                "strata_by_component must contain one entry per component."
            )
        for indices in self.strata_by_component:
            if not indices:
                raise ReferenceContractError(
                    "Every component must contain at least one balanced stratum."
                )
            if len(set(indices)) != len(indices):
                raise ReferenceContractError(
                    "Component stratum indices must be unique."
                )
            if min(indices) < 0 or max(indices) >= a.size:
                raise ReferenceContractError(
                    "Component stratum index is out of bounds."
                )

        object.__setattr__(self, "a", a.copy())
        object.__setattr__(self, "bulk_scales", bulk_scales.copy())
        object.__setattr__(self, "degrees_of_freedom", dof.copy())
        object.__setattr__(self, "component_scales", component_scales.copy())

    @property
    def aspect_ratios(self) -> NDArray[np.float64]:
        return np.asarray(
            self.bulk_dimension / self.degrees_of_freedom,
            dtype=np.float64,
        )

    @property
    def products(self) -> NDArray[np.float64]:
        """Return ``k_s=(N/d_s) a_s C_s`` from the paper."""

        return np.asarray(
            self.aspect_ratios * self.a * self.bulk_scales,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ReferenceEdge:
    m: float
    value: float
    curvature: float


@dataclass(frozen=True)
class ReferenceOutlier:
    lambda_hat: float
    m: float
    t_values: NDArray[np.float64]
    target_component: int
    mu_hat: float


ReferenceSource = Literal["oracle", "sham"]


@dataclass(frozen=True)
class ReferenceCandidate:
    candidate_source: ReferenceSource
    mu_hat: float
    lambda_hat: float
    eigvec: NDArray[np.float64]

    def as_mapping(self) -> Mapping[str, object]:
        return {
            "candidate_source": self.candidate_source,
            "mu_hat": float(self.mu_hat),
            "lambda_hat": float(self.lambda_hat),
            "eigvec": np.asarray(self.eigvec, dtype=np.float64).copy(),
        }


def _one_dimensional_finite(
    name: str, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ReferenceContractError(
            f"{name} must be a non-empty one-dimensional array."
        )
    if not np.all(np.isfinite(array)):
        raise ReferenceContractError(f"{name} must contain only finite values.")
    return array


def _denominators(m: float, design: BalancedReferenceDesign) -> NDArray[np.float64]:
    m_value = float(m)
    if not np.isfinite(m_value) or m_value == 0.0:
        raise ReferenceContractError("m must be a non-zero finite scalar.")
    denominators = 1.0 + design.products * m_value
    if np.any(np.abs(denominators) <= 1e-10):
        raise ReferenceContractError("m lies at a balanced-design pole.")
    return np.asarray(denominators, dtype=np.float64)


def z_reference(m: float, design: BalancedReferenceDesign) -> float:
    """Evaluate FJS equation (5.5) without production-code dependencies."""

    denominators = _denominators(m, design)
    numerators = design.bulk_scales * design.a
    return float(-1.0 / float(m) + np.sum(numerators / denominators))


def z_prime_reference(m: float, design: BalancedReferenceDesign) -> float:
    denominators = _denominators(m, design)
    numerators = design.bulk_scales * design.a
    derivative_terms = design.aspect_ratios * numerators**2 / denominators**2
    return float(1.0 / float(m) ** 2 - np.sum(derivative_terms))


def z_double_prime_reference(m: float, design: BalancedReferenceDesign) -> float:
    denominators = _denominators(m, design)
    numerators = design.bulk_scales * design.a
    curvature_terms = 2.0 * design.aspect_ratios**2 * numerators**3 / denominators**3
    return float(-2.0 / float(m) ** 3 + np.sum(curvature_terms))


def _product(polynomials: list[Polynomial]) -> Polynomial:
    result = Polynomial([1.0])
    for polynomial in polynomials:
        result *= polynomial
    return result


def _real_polynomial_roots(polynomial: Polynomial) -> list[float]:
    roots: list[float] = []
    for root in polynomial.roots():
        if abs(float(np.imag(root))) <= 1e-8 * max(1.0, abs(float(np.real(root)))):
            value = float(np.real(root))
            if np.isfinite(value):
                roots.append(value)
    roots.sort()
    deduplicated: list[float] = []
    for value in roots:
        if not deduplicated or abs(value - deduplicated[-1]) > 1e-7 * max(
            1.0, abs(value)
        ):
            deduplicated.append(value)
    return deduplicated


def stationary_roots_reference(design: BalancedReferenceDesign) -> tuple[float, ...]:
    """Enumerate every finite real stationary root of the rational MP map."""

    linear = [Polynomial([1.0, value]) for value in design.products]
    squared = [polynomial * polynomial for polynomial in linear]
    derivative_polynomial = _product(squared)
    m_squared = Polynomial([0.0, 0.0, 1.0])
    numerators = design.bulk_scales * design.a
    for index, coefficient in enumerate(design.aspect_ratios * numerators**2):
        other_factors = [
            polynomial for j, polynomial in enumerate(squared) if j != index
        ]
        derivative_polynomial -= (
            float(coefficient) * m_squared * _product(other_factors)
        )

    roots: list[float] = []
    for root in _real_polynomial_roots(derivative_polynomial):
        if root == 0.0:
            continue
        try:
            residual = z_prime_reference(root, design)
        except ReferenceContractError:
            continue
        if abs(residual) <= 1e-7 * max(1.0, abs(root) ** -2):
            roots.append(root)
    return tuple(roots)


def upper_edge_reference(design: BalancedReferenceDesign) -> ReferenceEdge:
    """Return the largest convex stationary value on the negative-real branch.

    On the branch above the largest support point, ``z(m)`` diverges upward at
    both endpoints and the upper edge is a local minimum, so ``z''(m)>0``.
    """

    candidates: list[ReferenceEdge] = []
    for root in stationary_roots_reference(design):
        if root >= 0.0:
            continue
        curvature = z_double_prime_reference(root, design)
        if not np.isfinite(curvature) or curvature <= 0.0:
            continue
        candidates.append(
            ReferenceEdge(
                m=float(root),
                value=z_reference(root, design),
                curvature=float(curvature),
            )
        )
    if not candidates:
        raise ReferenceContractError(
            "No convex negative-real stationary root defines an upper MP edge."
        )
    return max(candidates, key=lambda candidate: candidate.value)


def admissible_root_reference(
    lambda_hat: float,
    design: BalancedReferenceDesign,
    *,
    require_above_upper_edge: bool = True,
) -> float:
    """Solve ``z(m)=lambda`` by polynomial roots, independently of bracketing code."""

    lambda_value = float(lambda_hat)
    if not np.isfinite(lambda_value):
        raise ReferenceContractError("lambda_hat must be finite.")
    if require_above_upper_edge:
        edge = upper_edge_reference(design)
        if lambda_value <= edge.value:
            raise ReferenceContractError(
                "lambda_hat must lie strictly above the independently computed "
                "upper edge."
            )

    linear = [Polynomial([1.0, value]) for value in design.products]
    denominator_polynomial = _product(linear)
    m_polynomial = Polynomial([0.0, 1.0])
    equation = -denominator_polynomial - (
        lambda_value * m_polynomial * denominator_polynomial
    )
    numerators = design.bulk_scales * design.a
    for index, numerator in enumerate(numerators):
        other_factors = [
            polynomial for j, polynomial in enumerate(linear) if j != index
        ]
        equation += float(numerator) * m_polynomial * _product(other_factors)

    admissible: list[float] = []
    for root in _real_polynomial_roots(equation):
        if root >= -1e-12:
            continue
        try:
            residual = z_reference(root, design) - lambda_value
            slope = z_prime_reference(root, design)
        except ReferenceContractError:
            continue
        if slope <= 0.0:
            continue
        if abs(residual) <= 1e-8 * max(1.0, abs(lambda_value)):
            admissible.append(root)
    if not admissible:
        raise ReferenceContractError(
            "No negative-real positive-slope root satisfies z(m)=lambda_hat."
        )
    return float(max(admissible))


def t_vector_reference(
    m: float, design: BalancedReferenceDesign
) -> NDArray[np.float64]:
    """Evaluate the balanced-design t-vector using the explicit inclusion lattice."""

    base_terms = design.a / _denominators(m, design)
    values = np.zeros(design.component_scales.size, dtype=np.float64)
    for component, indices in enumerate(design.strata_by_component):
        values[component] = design.component_scales[component] * float(
            np.sum(base_terms[list(indices)])
        )
    return values


def outlier_reference(
    lambda_hat: float,
    design: BalancedReferenceDesign,
    *,
    target_component: int,
    off_component_tolerance: float = 1e-10,
) -> ReferenceOutlier:
    """Map an isolated outlier to one covariance component or fail closed."""

    target = int(target_component)
    if target < 0 or target >= design.component_scales.size:
        raise ReferenceContractError("target_component is out of bounds.")
    m_value = admissible_root_reference(lambda_hat, design)
    t_values = t_vector_reference(m_value, design)
    target_value = float(t_values[target])
    if abs(target_value) <= off_component_tolerance:
        raise ReferenceContractError("The target t-vector entry is zero.")
    off_values = np.delete(t_values, target)
    if off_values.size and np.max(np.abs(off_values)) > off_component_tolerance:
        raise ReferenceContractError(
            "The candidate is not isolated to the requested covariance component."
        )
    mu_hat = float(lambda_hat) / target_value
    if not np.isfinite(mu_hat) or mu_hat <= 0.0:
        raise ReferenceContractError(
            "The mapped covariance eigenvalue is not positive."
        )
    return ReferenceOutlier(
        lambda_hat=float(lambda_hat),
        m=m_value,
        t_values=t_values,
        target_component=target,
        mu_hat=mu_hat,
    )


def require_reference_close(
    name: str,
    observed: float | NDArray[np.float64],
    expected: float | NDArray[np.float64],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> None:
    """Fail loudly when a production value disagrees with the frozen oracle."""

    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if observed_array.shape != expected_array.shape or not np.all(
        np.isfinite(observed_array)
    ):
        raise ReferenceContractError(
            f"{name} has invalid shape or non-finite production values."
        )
    if not np.allclose(observed_array, expected_array, rtol=rtol, atol=atol):
        max_error = float(np.max(np.abs(observed_array - expected_array)))
        raise ReferenceContractError(
            f"{name} disagrees with the independent reference; "
            f"max_abs_error={max_error:.12g}."
        )


def spectral_reconstruction_reference(
    baseline: NDArray[np.float64],
    direction: NDArray[np.float64],
    mu_hat: float,
) -> NDArray[np.float64]:
    """Replace an eigenpair while preserving the baseline on its orthogonal block."""

    matrix = np.asarray(baseline, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ReferenceContractError("baseline must be a square matrix.")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ReferenceContractError("baseline must be finite and symmetric.")
    vector = _normalise_direction(direction, expected_size=matrix.shape[0])
    mu_value = float(mu_hat)
    if not np.isfinite(mu_value) or mu_value <= 0.0:
        raise ReferenceContractError("mu_hat must be positive and finite.")
    projector = np.eye(matrix.shape[0], dtype=np.float64) - np.outer(vector, vector)
    reconstructed = projector @ matrix @ projector + mu_value * np.outer(vector, vector)
    return np.asarray(0.5 * (reconstructed + reconstructed.T), dtype=np.float64)


def _normalise_direction(
    direction: NDArray[np.float64], *, expected_size: int | None = None
) -> NDArray[np.float64]:
    vector = _one_dimensional_finite("direction", direction)
    if expected_size is not None and vector.size != expected_size:
        raise ReferenceContractError("direction dimension does not match the target.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ReferenceContractError("direction must have positive norm.")
    vector = vector / norm
    nonzero = np.flatnonzero(np.abs(vector) > 1e-14)
    if nonzero.size and vector[int(nonzero[0])] < 0.0:
        vector = -vector
    return np.asarray(vector, dtype=np.float64)


def oracle_and_sham_candidates(
    planted_direction: NDArray[np.float64],
    *,
    mu_hat: float,
    lambda_hat: float,
) -> tuple[ReferenceCandidate, ReferenceCandidate]:
    """Return a planted oracle and deterministic magnitude-matched orthogonal sham."""

    oracle_direction = _normalise_direction(planted_direction)
    if oracle_direction.size < 2:
        raise ReferenceContractError("A sham control requires at least two dimensions.")
    sham_direction: NDArray[np.float64] | None = None
    for index in np.argsort(np.abs(oracle_direction), kind="stable"):
        basis = np.zeros_like(oracle_direction)
        basis[int(index)] = 1.0
        residual = basis - float(np.dot(basis, oracle_direction)) * oracle_direction
        if np.linalg.norm(residual) > 1e-12:
            sham_direction = _normalise_direction(residual)
            break
    if sham_direction is None:
        raise ReferenceContractError(
            "Unable to construct an orthogonal sham direction."
        )
    if abs(float(np.dot(oracle_direction, sham_direction))) > 1e-12:
        raise ReferenceContractError(
            "Constructed sham is not orthogonal to the oracle."
        )
    mu_value = float(mu_hat)
    lambda_value = float(lambda_hat)
    if not np.isfinite(mu_value) or mu_value <= 0.0:
        raise ReferenceContractError("mu_hat must be positive and finite.")
    if not np.isfinite(lambda_value):
        raise ReferenceContractError("lambda_hat must be finite.")
    return (
        ReferenceCandidate("oracle", mu_value, lambda_value, oracle_direction),
        ReferenceCandidate("sham", mu_value, lambda_value, sham_direction),
    )
