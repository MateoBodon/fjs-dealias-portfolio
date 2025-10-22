from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fjs.balanced import mean_squares
from fjs.mp import mp_edge


@dataclass
class DealiasingResult:
    """Container for the results of spectral de-aliasing."""

    covariance: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    iterations: int


def dealias_covariance(
    covariance: NDArray[np.float64],
    spectrum: NDArray[np.float64],
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
    raise NotImplementedError("Covariance de-aliasing routine is not implemented yet.")


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


def _default_design(stats: dict[str, int]) -> dict[str, object]:
    n_groups = int(stats["I"])
    replicates = int(stats["J"])
    total_samples = int(stats["n"])
    d_vals = np.array(
        [float(n_groups - 1), float(total_samples - n_groups)], dtype=np.float64
    )
    c_vals = np.array([float(replicates), 1.0], dtype=np.float64)
    weight_vals = np.ones(2, dtype=np.float64)
    order = [[1, 2], [2]]
    return {
        "c": c_vals,
        "C": weight_vals,
        "d": d_vals,
        "N": float(replicates),
        "order": order,
    }


def _merge_detections(
    detections: list[dict[str, object]], tol: float = 0.25
) -> list[dict[str, object]]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: float(item["mu_hat"]))
    clusters: list[list[dict[str, object]]] = []
    current_cluster: list[dict[str, object]] = [ordered[0]]
    cluster_min = cluster_max = float(ordered[0]["mu_hat"])

    for candidate in ordered[1:]:
        mu_val = float(candidate["mu_hat"])
        center = 0.5 * (cluster_min + cluster_max)
        if abs(mu_val - center) <= tol:
            current_cluster.append(candidate)
            cluster_min = min(cluster_min, mu_val)
            cluster_max = max(cluster_max, mu_val)
        else:
            clusters.append(current_cluster)
            current_cluster = [candidate]
            cluster_min = cluster_max = mu_val
    clusters.append(current_cluster)

    merged: list[dict[str, object]] = []
    for cluster in clusters:
        best = max(cluster, key=lambda item: float(item["lambda_hat"]))
        merged.append(best)
    return merged


def dealias_search(
    y: np.ndarray,
    groups: np.ndarray,
    target_r: int,
    *,
    a_grid: int = 120,
    delta: float = 0.5,
    eps: float = 0.02,
    stability_eta_deg: float = 1.0,
    design: dict | None = None,
) -> list[dict[str, object]]:
    """
    Perform Algorithm 1 de-aliasing search for one-way balanced designs.
    """
    observations, assignments = _validate_inputs(y, groups)
    stats = mean_squares(observations, assignments)

    if design is None:
        design_params = _default_design(stats)
    else:
        required_keys = {"c", "C", "d", "N", "order"}
        missing = required_keys - set(design)
        if missing:
            raise ValueError(f"Design dictionary missing keys: {sorted(missing)}")
        design_params = {
            "c": np.asarray(design["c"], dtype=np.float64),
            "C": np.asarray(design["C"], dtype=np.float64),
            "d": np.asarray(design["d"], dtype=np.float64),
            "N": float(design["N"]),
            "order": design["order"],
        }

    ms1_scaled = stats["MS1"].astype(np.float64)
    ms2_scaled = stats["MS2"].astype(np.float64)
    sigma_components = [
        stats["Sigma1_hat"].astype(np.float64),
        stats["Sigma2_hat"].astype(np.float64),
    ]
    c_weights = design_params["C"]
    d_vec = design_params["d"]
    n_total = design_params["N"]
    component_count = len(sigma_components)

    if not (0 <= target_r < component_count):
        raise ValueError("target_r must reference a valid component index.")

    angles = np.linspace(0.0, 2.0 * np.pi, num=a_grid, endpoint=False, dtype=np.float64)
    eta_rad = np.deg2rad(stability_eta_deg)
    detections: list[dict[str, object]] = []

    def _check_angle(angle: float, lam_val: float) -> bool:
        a_vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        if np.any(a_vec < -1e-8):
            return True
        try:
            z_plus = mp_edge(a_vec, c_weights, d_vec, n_total)
        except (RuntimeError, ValueError):
            return False
        return lam_val >= z_plus + delta

    for theta in angles:
        a_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        if np.any(a_vec < -1e-8):
            continue
        try:
            z_plus = mp_edge(a_vec, c_weights, d_vec, n_total)
        except (RuntimeError, ValueError):
            continue

        sigma_hat = a_vec[0] * ms1_scaled + a_vec[1] * ms2_scaled
        try:
            eigvals, eigvecs = np.linalg.eigh(sigma_hat)
        except np.linalg.LinAlgError:
            continue

        order_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order_idx]
        eigvecs = eigvecs[:, order_idx]

        for idx, lam_val in enumerate(eigvals):
            if lam_val < z_plus + delta:
                break
            component_vals = [
                float(eigvecs[:, idx].T @ component @ eigvecs[:, idx])
                for component in sigma_components
            ]

            target_val = component_vals[target_r]
            if target_val <= eps:
                continue
            if any(
                component_vals[j] > max(eps, 0.5 * target_val)
                for j in range(len(component_vals))
                if j != target_r
            ):
                continue

            if not (
                _check_angle(theta + eta_rad, lam_val)
                and _check_angle(theta - eta_rad, lam_val)
            ):
                continue

            detection = {
                "mu_hat": float(target_val),
                "lambda_hat": float(lam_val),
                "a": a_vec.tolist(),
                "components": component_vals,
                "eigvec": eigvecs[:, idx].copy(),
            }
            detections.append(detection)

    merged = _merge_detections(detections)
    return merged
