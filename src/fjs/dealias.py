from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import os
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from fjs.balanced import mean_squares
from fjs.config import DetectionSettings, get_detection_settings
from fjs.mp import admissible_m_from_lambda, estimate_Cs_from_MS, mp_edge, t_vec
from fjs.theta_solver import ThetaSolverParams, solve_theta_for_t2_zero

_FJS_DEBUG_RAW = os.getenv("FJS_DEBUG", "").strip().lower()
_FJS_DEBUG_ENABLED = _FJS_DEBUG_RAW in {"1", "true", "yes", "on", "debug"}


def _debug_log(payload: dict[str, Any]) -> None:
    if not _FJS_DEBUG_ENABLED:
        return

    def _convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, (float, int, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        try:
            return float(value)
        except Exception:
            return str(value)

    sanitized = {str(k): _convert(v) for k, v in payload.items()}
    print(f"[FJS_DEBUG] {json.dumps(sanitized, sort_keys=True)}", flush=True)


class DesignParams(TypedDict):
    c: NDArray[np.float64]
    C: NDArray[np.float64]
    d: NDArray[np.float64]
    N: float
    order: list[list[int]]


class Detection(TypedDict):
    mu_hat: float
    lambda_hat: float
    a: list[float]
    components: list[float]
    eigvec: NDArray[np.float64]
    stability_margin: float
    edge_margin: float | None
    buffer_margin: float | None
    t_values: list[float] | None
    admissible_root: bool | None
    solver_used: str | None
    # Optional diagnostics
    z_plus: float | None
    threshold_main: float | None
    # Optional sensitivity band diagnostics (±fraction on Cs)
    z_plus_low: float | None
    z_plus_high: float | None
    z_plus_scm: float | None
    threshold_low: float | None
    threshold_high: float | None
    stability_margin_low: float | None
    stability_margin_high: float | None
    sensitivity_low_accept: bool | None
    sensitivity_high_accept: bool | None
    # Target component diagnostics
    target_energy: float | None
    target_index: int | None
    off_component_ratio: float | None
    # Pre-gating diagnostics
    pre_outlier_count: int | None
    # Edge diagnostics
    edge_mode: str | None
    edge_scale: float | None


@dataclass
class DealiasingResult:
    """Container for the results of spectral de-aliasing."""

    covariance: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    iterations: int


def _compute_admissible_root(
    lam_val: float,
    a_vec: np.ndarray,
    C_for_mp: np.ndarray,
    d_vec: np.ndarray,
    n_total: float,
    cs_vec: np.ndarray,
) -> bool:
    try:
        admissible_m_from_lambda(
            float(lam_val),
            a_vec.tolist(),
            C_for_mp.tolist(),
            d_vec.tolist(),
            n_total,
            Cs=cs_vec,
        )
        return True
    except Exception:
        return False


def _orthonormal_tangent_basis(a_vec: np.ndarray) -> list[np.ndarray]:
    """Return an orthonormal basis for the tangent space at ``a_vec`` on the sphere."""
    dim = int(a_vec.shape[0])
    if dim < 2:
        return []
    basis: list[np.ndarray] = []
    identity = np.eye(dim, dtype=np.float64)
    for e_vec in identity:
        candidate = e_vec - np.dot(e_vec, a_vec) * a_vec
        if np.linalg.norm(candidate) <= 1e-12:
            continue
        for existing in basis:
            candidate -= np.dot(candidate, existing) * existing
        norm = np.linalg.norm(candidate)
        if norm > 1e-12:
            basis.append(candidate / norm)
        if len(basis) >= dim - 1:
            break
    if not basis:
        fallback = np.zeros(dim, dtype=np.float64)
        fallback[0] = 1.0
        candidate = fallback - np.dot(fallback, a_vec) * a_vec
        norm = np.linalg.norm(candidate)
        if norm > 1e-12:
            basis.append(candidate / norm)
    return basis


def _rotate_on_sphere(
    base: np.ndarray,
    tangent: np.ndarray,
    angle: float,
) -> np.ndarray:
    rotated = np.cos(angle) * base + np.sin(angle) * tangent
    norm = np.linalg.norm(rotated)
    if norm <= 0.0:
        raise ValueError("Rotation yielded a zero vector.")
    return rotated / norm


def _generate_unit_vectors(
    component_count: int,
    a_grid: int,
    *,
    nonnegative: bool,
) -> list[np.ndarray]:
    if component_count < 2:
        raise ValueError("At least two components are required for scanning.")
    vectors: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    def _record(vec: np.ndarray) -> None:
        if nonnegative and np.any(vec < -1e-8):
            return
        key = tuple(np.round(vec, decimals=12))
        if key in seen:
            return
        seen.add(key)
        vectors.append(vec)

    if component_count == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, num=a_grid, endpoint=False, dtype=np.float64)
        for angle in angles:
            vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            _record(vec)
        return vectors

    if component_count == 3:
        num_phi = max(3, int(np.sqrt(max(a_grid, 1))))
        num_theta = max(6, int(np.ceil(max(a_grid, 1) / num_phi)))
        theta_vals = np.linspace(0.0, 2.0 * np.pi, num=num_theta, endpoint=False, dtype=np.float64)
        phi_vals = np.linspace(0.0, np.pi, num=num_phi, endpoint=False, dtype=np.float64)
        for phi in phi_vals:
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            if sin_phi < 1e-8:
                continue
            for theta in theta_vals:
                vec = np.array(
                    [np.cos(theta) * sin_phi, np.sin(theta) * sin_phi, cos_phi],
                    dtype=np.float64,
                )
                _record(vec / np.linalg.norm(vec))
        _record(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        _record(np.array([0.0, 0.0, -1.0], dtype=np.float64))
        return vectors

    raise NotImplementedError("Scanning beyond three components is not implemented.")


# Helper for k=2 solver bookkeeping
def _normalise_angle(theta: float) -> float:
    tau = 2.0 * math.pi
    return float((theta % tau + tau) % tau)


_ANGLE_TOL = 1e-6


def _angle_key(theta: float) -> float:
    return round(_normalise_angle(theta), 12)


# fmt: off
def _sigma_of_a_from_MS(a: np.ndarray, MS_list: list[np.ndarray]) -> np.ndarray:  # noqa: N802,N803
# fmt: on
    """Return Σ̂(a)=∑_s a_s MS_s (balanced design)."""
    a_vec = np.asarray(a, dtype=np.float64)
    if a_vec.ndim != 1:
        raise ValueError("a must be a one-dimensional array.")
    if not MS_list:
        raise ValueError("MS_list must contain at least one matrix.")
    if len(MS_list) != a_vec.shape[0]:
        raise ValueError("a and MS_list must have matching lengths.")

    reference = np.asarray(MS_list[0], dtype=np.float64)
    sigma = np.zeros_like(reference, dtype=np.float64)
    for idx, ms in enumerate(MS_list):
        ms_array = np.asarray(ms, dtype=np.float64)
        if ms_array.shape != reference.shape:
            raise ValueError("All mean square matrices must share the same shape.")
        sigma += a_vec[idx] * ms_array
    return sigma


def dealias_covariance(
    covariance: NDArray[np.float64],
    spectrum: NDArray[np.float64] | Sequence[Any],
) -> DealiasingResult:
    """
    Remove aliasing artefacts from a sample covariance matrix.

    Parameters
    ----------
    covariance:
        Sample covariance matrix shaped `(n_assets, n_assets)`.
    spectrum:
        Estimated eigenvalue spectrum used for calibration.

    Returns
    -------
    DealiasingResult
        Structured result containing the refined covariance and metadata.
    """
    if spectrum is None:
        raise ValueError("spectrum must be provided for covariance de-aliasing.")

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must contain finite entries.")

    # Symmetrise to guard against minor numerical asymmetry.
    matrix = 0.5 * (matrix + matrix.T)
    adjusted = matrix.copy()
    n = adjusted.shape[0]
    iterations = 0

    def _apply_update(vec: NDArray[np.float64], target: float) -> None:
        nonlocal adjusted, iterations
        vector = np.asarray(vec, dtype=np.float64).reshape(n)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0.0:
            return
        unit = vector / norm
        mu_target = float(target)
        if not np.isfinite(mu_target):
            return
        rayleigh = float(unit.T @ adjusted @ unit)
        adjusted += (mu_target - rayleigh) * np.outer(unit, unit)
        iterations += 1

    handled = False

    # Case 1: spectrum provided as detection dictionaries (dealias_search output)
    if isinstance(spectrum, Sequence) and spectrum and not np.isscalar(spectrum):
        mu_candidates: list[float] = []
        vec_candidates: list[NDArray[np.float64]] = []
        for entry in spectrum:
            if isinstance(entry, dict):
                mu_val = entry.get("mu_hat")
                vec_val = entry.get("eigvec")
                if mu_val is not None and vec_val is not None:
                    mu_candidates.append(float(mu_val))
                    vec_candidates.append(np.asarray(vec_val, dtype=np.float64))
        if mu_candidates and vec_candidates and len(mu_candidates) == len(vec_candidates):
            for mu_val, vec_val in zip(mu_candidates, vec_candidates):
                if vec_val.shape[0] != n:
                    raise ValueError("eigvec dimensions must match covariance.")
                _apply_update(vec_val, mu_val)
            handled = True

    if not handled:
        values = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        if values.ndim != 1:
            raise ValueError("spectrum must be one-dimensional when given as array.")
        if values.size == 0:
            spectrum_sorted = np.linalg.eigvalsh(adjusted)
            return DealiasingResult(covariance=adjusted, spectrum=spectrum_sorted, iterations=0)

        eigvals, eigvecs = np.linalg.eigh(adjusted)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        replacements = min(values.size, eigvecs.shape[1])
        target_vals = np.sort(values)[::-1]
        for idx in range(replacements):
            _apply_update(eigvecs[:, idx], target_vals[idx])
        handled = True

    final_spectrum = np.linalg.eigvalsh(adjusted)
    return DealiasingResult(
        covariance=adjusted,
        spectrum=final_spectrum,
        iterations=iterations,
    )


def _validate_inputs(
    y: np.ndarray, groups: np.ndarray
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    observations = np.asarray(y, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("Y must be a two-dimensional array shaped (n, p).")
    assignments = np.asarray(groups)
    if assignments.ndim != 1:
        raise ValueError("groups must be a one-dimensional array.")
    if observations.shape[0] != assignments.shape[0]:
        raise ValueError("Y and groups must contain the same number of rows.")
    return observations, assignments.astype(np.intp, copy=False)


def _default_design(stats: dict[str, int]) -> DesignParams:
    n_groups = int(stats["I"])
    replicates = int(stats["J"])
    total_samples = int(stats["n"])
    d_vals = np.array(
        [float(n_groups - 1), float(total_samples - n_groups)], dtype=np.float64
    )
    c_vals = np.array([float(replicates), 1.0], dtype=np.float64)
    weight_vals = np.ones(2, dtype=np.float64)
    order = [[1, 2], [2]]
    return DesignParams(
        c=c_vals,
        C=weight_vals,
        d=d_vals,
        N=float(replicates),
        order=order,
    )


def _merge_detections(
    detections: list[Detection], eps_factor: float = 0.05
) -> list[Detection]:
    if not detections:
        return []

    mu_vals = np.array([det["mu_hat"] for det in detections], dtype=np.float64)
    order = np.argsort(mu_vals)
    mu_sorted = mu_vals[order]
    detections_sorted = [detections[idx] for idx in order]

    n = len(detections_sorted)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mu_i = mu_sorted[i]
            mu_j = mu_sorted[j]
            radius = eps_factor * max(mu_i, mu_j, 1e-12)
            if abs(mu_i - mu_j) <= radius:
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = [False] * n
    merged: list[Detection] = []
    for idx in range(n):
        if visited[idx]:
            continue
        stack = [idx]
        cluster_indices: list[int] = []
        visited[idx] = True
        while stack:
            current = stack.pop()
            cluster_indices.append(current)
            for neighbour in adjacency[current]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        cluster = [detections_sorted[i] for i in cluster_indices]
        # Prefer higher stability; break ties by larger target component energy when available
        best = max(
            cluster,
            key=lambda item: (
                item["stability_margin"],
                item.get("target_energy", item["lambda_hat"]),
            ),
        )
        merged.append(best)
    merged.sort(key=lambda det: det["mu_hat"])
    return merged


def dealias_search(
    y: np.ndarray,
    groups: np.ndarray,
    target_r: int,
    *,
    Cs: Sequence[float] | None = None,  # noqa: N803
    a_grid: int | None = None,
    delta: float | None = None,
    delta_frac: float | None = None,
    eps: float | None = None,
    energy_min_abs: float | None = None,
    stability_eta_deg: float = 1.0,
    use_tvector: bool = True,
    nonnegative_a: bool = False,
    design: dict | None = None,
    cs_drop_top_frac: float | None = None,
    cs_sensitivity_frac: float | None = None,
    use_design_c_for_C: bool = False,
    scan_basis: str = "ms",
    oneway_a_solver: str = "auto",
    off_component_leak_cap: float | None = None,
    cs_scale: float | None = None,
    diagnostics: dict[str, int] | None = None,
    stats: dict[str, Any] | None = None,
    edge_scale: float | None = None,
    edge_mode: str | None = None,
    settings: DetectionSettings | None = None,
) -> list[Detection]:
    """
    Perform Algorithm 1 de-aliasing search for one-way balanced designs.
    """
    observations, assignments = _validate_inputs(y, groups)
    if stats is None:
        stats_dict: dict[str, Any] = mean_squares(observations, assignments)
    elif isinstance(stats, dict):
        stats_dict = dict(stats)
    else:
        raise TypeError("stats must be a mapping when provided explicitly.")

    if design is None:
        design_params = _default_design(stats_dict)
    else:
        required_keys = {"c", "C", "d", "N", "order"}
        missing = required_keys - set(design)
        if missing:
            raise ValueError(f"Design dictionary missing keys: {sorted(missing)}")
        design_params = DesignParams(
            c=np.asarray(design["c"], dtype=np.float64),
            C=np.asarray(design["C"], dtype=np.float64),
            d=np.asarray(design["d"], dtype=np.float64),
            N=float(design["N"]),
            order=list(design["order"]),
        )

    ms_list: list[np.ndarray] = []
    sigma_components: list[np.ndarray] = []
    idx = 1
    while True:
        ms_key = f"MS{idx}"
        sigma_key = f"Sigma{idx}_hat"
        if ms_key in stats_dict and sigma_key in stats_dict:
            ms_list.append(np.asarray(stats_dict[ms_key], dtype=np.float64))
            sigma_components.append(np.asarray(stats_dict[sigma_key], dtype=np.float64))
            idx += 1
            continue
        break

    if len(ms_list) != len(sigma_components) or not ms_list:
        raise ValueError("stats must provide matched mean squares and sigma components.")
    # Design parameters:
    # - c_vec holds the design coefficients (e.g., [J, 1] in one-way)
    # - design_params["C"] defaults to ones and is not used for MP mapping
    #   in the de-aliasing search; we use c_vec consistently for the MP calls.
    c_weights = np.asarray(design_params["C"], dtype=np.float64)
    c_vec = np.asarray(design_params["c"], dtype=np.float64)
    d_vec = np.asarray(design_params["d"], dtype=np.float64)
    n_total = float(design_params["N"])
    component_count = len(sigma_components)

    if c_vec.shape[0] != component_count:
        raise ValueError("Design parameters must align with sigma components.")

    solver_mode = (oneway_a_solver or "grid").strip().lower()
    if solver_mode not in {"auto", "rootfind", "grid"}:
        raise ValueError("oneway_a_solver must be 'auto', 'rootfind', or 'grid'.")
    use_theta_solver = component_count == 2 and solver_mode in {"auto", "rootfind"}

    if not (0 <= target_r < component_count):
        raise ValueError("target_r must reference a valid component index.")

    if diagnostics is not None:
        diagnostics.clear()

    settings_obj = settings or get_detection_settings()

    if delta is None:
        delta = settings_obj.delta
    delta = float(delta)

    if delta_frac is None:
        delta_frac = settings_obj.delta_frac
    delta_frac = float(delta_frac) if delta_frac is not None else None

    if a_grid is None:
        a_grid = settings_obj.a_grid_size
    a_grid = int(a_grid)

    if cs_drop_top_frac is None:
        cs_drop_top_frac = settings_obj.cs_drop_top_frac
    cs_drop_top_frac = float(cs_drop_top_frac) if cs_drop_top_frac is not None else None

    angle_min_cos = float(settings_obj.angle_min_cos)
    require_isolated = bool(settings_obj.require_isolated)
    max_q = max(0, int(settings_obj.q_max)) if settings_obj.q_max is not None else 0
    if eps is None:
        eps = settings_obj.t_eps
    eps = float(eps)
    if off_component_leak_cap is None:
        off_component_leak_cap = settings_obj.off_component_cap

    def _diag_inc(name: str) -> None:
        if diagnostics is not None:
            diagnostics[name] = diagnostics.get(name, 0) + 1

    if Cs is None:
        # Determine trimming via fraction when provided; else fall back to heuristic
        if cs_drop_top_frac is not None:
            p_dim = int(ms_list[0].shape[0])
            fraction = float(cs_drop_top_frac)
            drop_top = min(p_dim - 1, max(1, int(round(p_dim * fraction))))
        else:
            drop_top = min(5, max(1, ms_list[0].shape[0] // 20))
        cs_vec = np.asarray(
            estimate_Cs_from_MS(
                ms_list,
                d_vec.tolist(),
                c_vec.tolist(),
                drop_top=drop_top,
            ),
            dtype=np.float64,
        )
    else:
        cs_vec = np.asarray(Cs, dtype=np.float64)
        if cs_vec.ndim != 1 or cs_vec.shape[0] != component_count:
            raise ValueError("Cs must match the number of mean square components.")

    # Auto-scale Cs to the Σ̂ basis magnitude when scanning Σ̂(a)
    scan_basis_norm = (scan_basis or "sigma").strip().lower()
    if scan_basis_norm not in {"sigma", "ms"}:
        raise ValueError("scan_basis must be 'sigma' or 'ms'.")
    # Prepare Σ̂(a)-aware scaling for MP mapping
    lam_mean = float("nan")
    if scan_basis_norm == "sigma":
        try:
            sigma_total = np.zeros_like(sigma_components[0], dtype=np.float64)
            for component in sigma_components:
                sigma_total = sigma_total + component
            eigvals_total = np.linalg.eigvalsh(0.5 * (sigma_total + sigma_total.T))
            lam_mean = float(np.mean(eigvals_total)) if eigvals_total.size else float(np.nan)
        except Exception:
            lam_mean = float("nan")
        cs_mean = float(np.mean(cs_vec)) if cs_vec.size else float("nan")
        if np.isfinite(lam_mean) and np.isfinite(cs_mean) and cs_mean > 0:
            auto_alpha = lam_mean / cs_mean
        else:
            auto_alpha = 1.0
        alpha = float(cs_scale) if (cs_scale is not None and np.isfinite(cs_scale)) else auto_alpha
        # Guard against underflow/overflow
        if not np.isfinite(alpha) or alpha <= 0.0:
            alpha = 1.0
        cs_vec = (alpha * cs_vec).astype(np.float64, copy=False)

    candidate_vectors = _generate_unit_vectors(
        len(c_vec),
        int(a_grid),
        nonnegative=nonnegative_a,
    )
    angle_source: dict[float, str] = {}
    if use_theta_solver:
        for vec in candidate_vectors:
            theta_key = _angle_key(float(math.atan2(vec[1], vec[0])))
            angle_source.setdefault(theta_key, "grid")
    eta_rad = np.deg2rad(stability_eta_deg)
    detections: list[Detection] = []

    def _threshold_from_z(z_plus_val: float) -> float:
        # Per-window delta: add the larger of absolute and relative offsets
        rel = 0.0 if (delta_frac is None) else float(delta_frac) * float(z_plus_val)
        return float(z_plus_val + max(float(delta), rel))

    C_for_mp = c_vec if use_design_c_for_C else c_weights

    edge_scale_val = 1.0
    if edge_scale is not None:
        try:
            edge_scale_val = float(edge_scale)
        except (TypeError, ValueError):
            edge_scale_val = 1.0
        if not np.isfinite(edge_scale_val) or edge_scale_val <= 0.0:
            edge_scale_val = 1.0
    resolved_edge_mode = edge_mode if edge_mode is not None else settings_obj.edge_mode
    edge_mode_name = (resolved_edge_mode or "scm").strip()

    def _edge_margin_for(Cs_local: np.ndarray):
        def margin(a_vec_local: np.ndarray, lam_val: float) -> float | None:
            if nonnegative_a and np.any(a_vec_local < -1e-8):
                return float("inf")
            try:
                z_plus_local = mp_edge(
                    a_vec_local.tolist(),
                    # Use chosen mapping for MP edge
                    C_for_mp.tolist(),
                    d_vec.tolist(),
                    n_total,
                    Cs=Cs_local,
                )
            except (RuntimeError, ValueError):
                return None
            z_scaled = float(z_plus_local) * edge_scale_val
            threshold_local = _threshold_from_z(z_scaled)
            return float(lam_val - threshold_local)

        return margin

    _edge_margin = _edge_margin_for(cs_vec)

    # Sensitivity band configuration (optional diagnostics only)
    band_frac = None if cs_sensitivity_frac is None else float(cs_sensitivity_frac)
    if band_frac is not None and band_frac < 0.0:
        band_frac = abs(band_frac)
    cs_vec_low: np.ndarray | None
    cs_vec_high: np.ndarray | None
    if band_frac is not None and band_frac > 0.0:
        cs_vec_low = (1.0 - band_frac) * cs_vec
        cs_vec_high = (1.0 + band_frac) * cs_vec
        _edge_margin_low = _edge_margin_for(cs_vec_low)
        _edge_margin_high = _edge_margin_for(cs_vec_high)
    else:
        cs_vec_low = None
        cs_vec_high = None
        _edge_margin_low = None  # type: ignore[assignment]
        _edge_margin_high = None  # type: ignore[assignment]

    # Choose C mapping for MP computations
    if use_design_c_for_C:
        C_for_mp = c_vec
    else:
        if scan_basis_norm == "sigma" and np.isfinite(lam_mean) and lam_mean > 0.0:
            C_for_mp = np.full_like(c_vec, lam_mean, dtype=np.float64)
        else:
            C_for_mp = c_weights

    terminate = False
    for candidate_index, a_vec in enumerate(candidate_vectors):
        if terminate:
            break
        if nonnegative_a and np.any(a_vec < -1e-8):
            if _FJS_DEBUG_ENABLED:
                _debug_log(
                    {
                        "candidate_index": int(candidate_index),
                        "decision": "reject",
                        "reason": "nonnegative_constraint",
                        "min_cos": float(np.min(np.abs(a_vec))),
                    }
                )
            continue
        theta_current = None
        theta_key = None
        solver_tag = "grid"
        if use_theta_solver:
            theta_current = float(math.atan2(a_vec[1], a_vec[0]))
            theta_key = _angle_key(theta_current)
            solver_tag = angle_source.get(theta_key, "grid")
        debug_base = {
            "candidate_index": int(candidate_index),
            "a_grid": int(a_grid),
            "min_cos": float(np.min(np.abs(a_vec))),
            "solver": solver_tag,
        }
        if theta_current is not None:
            debug_base["theta_rad"] = float(theta_current)
        if angle_min_cos > 0.0 and debug_base["min_cos"] < angle_min_cos:
            _debug_log({**debug_base, "decision": "reject", "reason": "angle_threshold", "angle_min_cos": angle_min_cos})
            _diag_inc("angle_gate")
            continue
        try:
            z_plus_base = mp_edge(
                a_vec.tolist(),
                C_for_mp.tolist(),
                d_vec.tolist(),
                n_total,
                Cs=cs_vec,
            )
        except (RuntimeError, ValueError):
            _debug_log({**debug_base, "decision": "reject", "reason": "mp_edge_fail"})
            continue

        z_plus_scaled = float(z_plus_base) * edge_scale_val
        debug_base["z_plus"] = float(z_plus_scaled)

        # Evaluate scanning matrix per chosen basis
        if scan_basis_norm == "sigma":
            sigma_a = np.zeros_like(sigma_components[0], dtype=np.float64)
            for weight, sigma_component in zip(a_vec, sigma_components):
                sigma_a = sigma_a + float(weight) * sigma_component
        else:
            sigma_a = _sigma_of_a_from_MS(a_vec, ms_list)
        try:
            eigvals, eigvecs = np.linalg.eigh(sigma_a)
        except np.linalg.LinAlgError:
            continue

        order_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order_idx]
        eigvecs = eigvecs[:, order_idx]

        threshold_main = _threshold_from_z(z_plus_scaled)

        # Optional band diagnostics for this orientation
        if cs_vec_low is not None:
            try:
                z_plus_low = mp_edge(
                    a_vec.tolist(),
                    C_for_mp.tolist(),
                    d_vec.tolist(),
                    n_total,
                    Cs=cs_vec_low,
                )
                z_plus_low = float(z_plus_low) * edge_scale_val
                threshold_low = _threshold_from_z(float(z_plus_low))
            except (RuntimeError, ValueError):
                z_plus_low = float("nan")
                threshold_low = float("nan")
        else:
            z_plus_low = None
            threshold_low = None
        if cs_vec_high is not None:
            try:
                z_plus_high = mp_edge(
                    a_vec.tolist(),
                    C_for_mp.tolist(),
                    d_vec.tolist(),
                    n_total,
                    Cs=cs_vec_high,
                )
                z_plus_high = float(z_plus_high) * edge_scale_val
                threshold_high = _threshold_from_z(float(z_plus_high))
            except (RuntimeError, ValueError):
                z_plus_high = float("nan")
                threshold_high = float("nan")
        else:
            z_plus_high = None
            threshold_high = None

        # Pre-gating count of outliers for this angle
        pre_outlier_count_vec = int(np.count_nonzero(eigvals >= threshold_main))

        if require_isolated and pre_outlier_count_vec > 1:
            _debug_log({**debug_base, "decision": "reject", "reason": "multi_outlier", "pre_outliers": int(pre_outlier_count_vec)})
            _diag_inc("multi_outlier")
            continue

        for idx, lam_val in enumerate(eigvals):
            debug_ctx = {
                **debug_base,
                "eigen_index": int(idx),
                "lambda": float(lam_val),
                "threshold": float(threshold_main),
                "pre_outliers": int(pre_outlier_count_vec),
            }
            if lam_val < threshold_main:
                _debug_log({**debug_ctx, "decision": "reject", "reason": "below_threshold"})
                break
            margin_main = lam_val - threshold_main
            debug_ctx["margin_main"] = float(margin_main)
            if margin_main < 0.0:
                _debug_log({**debug_ctx, "decision": "reject", "reason": "edge_buffer"})
                _diag_inc("edge_buffer")
                continue
            edge_margin_raw = float(lam_val - z_plus_scaled)
            debug_ctx["edge_margin"] = float(edge_margin_raw)
            t_vals: np.ndarray | None
            t_target: float | None
            try:
                raw_t = t_vec(
                    float(lam_val),
                    a_vec.tolist(),
                    C_for_mp.tolist(),
                    d_vec.tolist(),
                    n_total,
                    c_vec.tolist(),
                    design_params["order"],
                )
                t_vals = np.asarray(raw_t, dtype=np.float64)
                if t_vals.shape[0] <= target_r:
                    t_target = None
                else:
                    t_target = float(t_vals[target_r])
            except (RuntimeError, ValueError):
                if use_tvector:
                    _debug_log({**debug_ctx, "decision": "reject", "reason": "t_vec_fail"})
                    _diag_inc("other")
                    continue
                t_vals = None
                t_target = None

            if use_tvector:
                if t_vals is None or t_target is None or abs(t_target) <= eps:
                    _debug_log({**debug_ctx, "decision": "reject", "reason": "t_target_small"})
                    _diag_inc("other")
                    continue
                t_off = np.delete(t_vals, target_r)
                # Stricter absolute off-component cap to match guardrail tests
                if t_off.size and float(np.max(np.abs(t_off))) > float(eps):
                    _debug_log({**debug_ctx, "decision": "reject", "reason": "t_off_caps"})
                    _diag_inc("other")
                    continue

            if t_target is None or abs(t_target) < 1e-12:
                mu_hat = float(lam_val)
            else:
                mu_hat = float(lam_val / t_target)
            if not np.isfinite(mu_hat):
                _debug_log({**debug_ctx, "decision": "reject", "reason": "mu_nonfinite"})
                _diag_inc("other")
                continue
            if mu_hat <= 0.0:
                mu_hat = abs(mu_hat)
                if mu_hat <= 0.0:
                    _debug_log({**debug_ctx, "decision": "reject", "reason": "mu_nonpositive"})
                    _diag_inc("neg_mu")
                    continue

            component_vals = [
                float(eigvecs[:, idx].T @ component @ eigvecs[:, idx])
                for component in sigma_components
            ]
            target_component_val = component_vals[target_r]
            if target_component_val < 0.0:
                target_component_val = abs(target_component_val)
            energy_magnitude = abs(target_component_val)
            if (
                energy_min_abs is not None
                and energy_magnitude <= float(energy_min_abs)
            ):
                _debug_log(
                    {
                        **debug_ctx,
                        "decision": "reject",
                        "reason": "energy_floor",
                        "energy": float(energy_magnitude),
                    }
                )
                _diag_inc("energy_floor")
                continue
            denom = max(energy_magnitude, 1e-12)
            worst_off = max(
                (abs(component_vals[j]) for j in range(len(component_vals)) if j != target_r),
                default=0.0,
            )
            off_component_ratio = float(worst_off / denom) if denom > 0 else float("inf")
            debug_ctx["off_component_ratio"] = float(off_component_ratio)
            if (
                off_component_leak_cap is not None
                and off_component_ratio > float(off_component_leak_cap)
            ):
                _debug_log({**debug_ctx, "decision": "reject", "reason": "off_component_ratio"})
                _diag_inc("off_component_ratio")
                continue

            basis_dirs = _orthonormal_tangent_basis(a_vec)
            if basis_dirs:
                tangent_dir = basis_dirs[0]
                try:
                    a_plus = _rotate_on_sphere(a_vec, tangent_dir, eta_rad)
                    a_minus = _rotate_on_sphere(a_vec, -tangent_dir, eta_rad)
                except ValueError:
                    _debug_log({**debug_ctx, "decision": "reject", "reason": "stability_rotate"})
                    _diag_inc("stability_fail")
                    continue
                margin_plus = _edge_margin(a_plus, lam_val)
                margin_minus = _edge_margin(a_minus, lam_val)
            else:
                a_plus = a_vec
                a_minus = a_vec
                margin_plus = margin_main
                margin_minus = margin_main
            if margin_plus is None or margin_plus < 0.0:
                _debug_log({**debug_ctx, "decision": "reject", "reason": "stability_plus"})
                _diag_inc("stability_fail")
                continue
            if margin_minus is None or margin_minus < 0.0:
                _debug_log({**debug_ctx, "decision": "reject", "reason": "stability_minus"})
                _diag_inc("stability_fail")
                continue
            stability_margin = float(min(margin_main, margin_plus, margin_minus))
            debug_ctx["stability_margin"] = float(stability_margin)

            # Sensitivity margins/decisions (diagnostics only)
            stability_margin_low: float | None
            stability_margin_high: float | None
            accept_low: bool | None
            accept_high: bool | None
            if cs_vec_low is not None and _edge_margin_low is not None:
                mp_main_low = (
                    float(lam_val - threshold_low)
                    if (isinstance(threshold_low, float) and np.isfinite(threshold_low))
                    else float("nan")
                )
                mp_plus_low = _edge_margin_low(a_plus, lam_val)
                mp_minus_low = _edge_margin_low(a_minus, lam_val)
                if (
                    np.isfinite(mp_main_low)
                    and mp_plus_low is not None
                    and mp_minus_low is not None
                ):
                    stability_margin_low = float(
                        min(mp_main_low, mp_plus_low, mp_minus_low)
                    )
                    accept_low = bool(stability_margin_low >= 0.0)
                else:
                    stability_margin_low = float("nan")
                    accept_low = None
            else:
                stability_margin_low = None
                accept_low = None

            if cs_vec_high is not None and _edge_margin_high is not None:
                mp_main_high = (
                    float(lam_val - threshold_high)
                    if (isinstance(threshold_high, float) and np.isfinite(threshold_high))
                    else float("nan")
                )
                mp_plus_high = _edge_margin_high(a_plus, lam_val)
                mp_minus_high = _edge_margin_high(a_minus, lam_val)
                if (
                    np.isfinite(mp_main_high)
                    and mp_plus_high is not None
                    and mp_minus_high is not None
                ):
                    stability_margin_high = float(
                        min(mp_main_high, mp_plus_high, mp_minus_high)
                    )
                    accept_high = bool(stability_margin_high >= 0.0)
                else:
                    stability_margin_high = float("nan")
                    accept_high = None
            else:
                stability_margin_high = None
                accept_high = None

            detection = Detection(
                mu_hat=mu_hat,
                lambda_hat=float(lam_val),
                a=a_vec.tolist(),
                components=component_vals,
                eigvec=eigvecs[:, idx].copy(),
                stability_margin=stability_margin,
                z_plus=float(z_plus_scaled),
                z_plus_scm=float(z_plus_base),
                threshold_main=float(threshold_main),
                z_plus_low=(None if z_plus_low is None else float(z_plus_low)),
                z_plus_high=(None if z_plus_high is None else float(z_plus_high)),
                threshold_low=(
                    None if threshold_low is None else float(threshold_low)
                ),
                threshold_high=(
                    None if threshold_high is None else float(threshold_high)
                ),
                stability_margin_low=stability_margin_low,
                stability_margin_high=stability_margin_high,
                sensitivity_low_accept=accept_low,
                sensitivity_high_accept=accept_high,
                target_energy=float(target_component_val),
                target_index=int(target_r),
                off_component_ratio=float(off_component_ratio),
                pre_outlier_count=int(pre_outlier_count_vec),
                edge_margin=edge_margin_raw,
                buffer_margin=float(margin_main),
                t_values=(
                    None if t_vals is None else np.abs(t_vals).astype(float).tolist()
                ),
                admissible_root=_compute_admissible_root(
                    lam_val,
                    a_vec,
                    C_for_mp,
                    d_vec,
                    n_total,
                    cs_vec,
                ),
                edge_mode=edge_mode_name,
                edge_scale=float(edge_scale_val),
            )
            detection["solver_used"] = solver_tag
            detections.append(detection)
            debug_payload = {
                **debug_ctx,
                "decision": "accept",
                "mu_hat": float(mu_hat),
                "stability_margin": float(stability_margin),
                "buffer_margin": float(margin_main),
                "edge_margin": float(edge_margin_raw),
                "admissible_root": bool(detection["admissible_root"]),
            }
            _debug_log(debug_payload)
            if max_q > 0 and len(detections) >= max_q:
                terminate = True
                break

            if (
                use_theta_solver
                and solver_tag == "grid"
                and theta_current is not None
                and theta_key is not None
            ):
                solver_params = ThetaSolverParams(
                    C=C_for_mp.copy(),
                    d=d_vec.copy(),
                    N=float(n_total),
                    c=c_vec.copy(),
                    order=[list(item) for item in design_params["order"]],
                    Cs=None if cs_vec is None else cs_vec.copy(),
                    eps=float(eps),
                    delta=float(eta_rad if eta_rad > 0.0 else 1e-3),
                    grid_size=max(72, int(a_grid)),
                    tol=1e-8,
                    max_iter=60,
                )
                theta_solution = solve_theta_for_t2_zero(float(lam_val), solver_params)
                if theta_solution is not None:
                    theta_norm = _angle_key(theta_solution)
                    duplicate = any(
                        abs(theta_norm - existing) <= _ANGLE_TOL
                        for existing in angle_source.keys()
                    )
                    if not duplicate:
                        angle_source[theta_norm] = "rootfind"
                        candidate_vectors.append(
                            np.array(
                                [math.cos(theta_norm), math.sin(theta_norm)],
                                dtype=np.float64,
                            )
                        )
                elif solver_mode == "rootfind":
                    angle_source[theta_key] = "grid"

    merged = _merge_detections(detections)
    return merged
