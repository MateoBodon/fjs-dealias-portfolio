from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from fjs.balanced import mean_squares
from fjs.mp import estimate_Cs_from_MS, mp_edge, t_vec


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
    # Optional diagnostics
    z_plus: float | None
    threshold_main: float | None
    # Optional sensitivity band diagnostics (±fraction on Cs)
    z_plus_low: float | None
    z_plus_high: float | None
    threshold_low: float | None
    threshold_high: float | None
    stability_margin_low: float | None
    stability_margin_high: float | None
    sensitivity_low_accept: bool | None
    sensitivity_high_accept: bool | None


@dataclass
class DealiasingResult:
    """Container for the results of spectral de-aliasing."""

    covariance: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    iterations: int


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
        best = max(
            cluster,
            key=lambda item: (item["stability_margin"], item["lambda_hat"]),
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
    a_grid: int = 120,
    delta: float = 0.5,
    delta_frac: float | None = None,
    eps: float = 0.02,
    stability_eta_deg: float = 1.0,
    use_tvector: bool = True,
    nonnegative_a: bool = False,
    design: dict | None = None,
    cs_drop_top_frac: float | None = None,
    cs_sensitivity_frac: float | None = None,
    use_design_c_for_C: bool = False,
) -> list[Detection]:
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
        design_params = DesignParams(
            c=np.asarray(design["c"], dtype=np.float64),
            C=np.asarray(design["C"], dtype=np.float64),
            d=np.asarray(design["d"], dtype=np.float64),
            N=float(design["N"]),
            order=list(design["order"]),
        )

    ms1_scaled = stats["MS1"].astype(np.float64)
    ms2_scaled = stats["MS2"].astype(np.float64)
    sigma_components = [
        stats["Sigma1_hat"].astype(np.float64),
        stats["Sigma2_hat"].astype(np.float64),
    ]
    # Design parameters:
    # - c_vec holds the design coefficients (e.g., [J, 1] in one-way)
    # - design_params["C"] defaults to ones and is not used for MP mapping
    #   in the de-aliasing search; we use c_vec consistently for the MP calls.
    c_weights = np.asarray(design_params["C"], dtype=np.float64)
    c_vec = np.asarray(design_params["c"], dtype=np.float64)
    d_vec = np.asarray(design_params["d"], dtype=np.float64)
    n_total = float(design_params["N"])
    component_count = len(sigma_components)

    if not (0 <= target_r < component_count):
        raise ValueError("target_r must reference a valid component index.")

    if Cs is None:
        # Determine trimming via fraction when provided; else fall back to heuristic
        if cs_drop_top_frac is not None:
            p_dim = int(ms1_scaled.shape[0])
            fraction = float(cs_drop_top_frac)
            drop_top = min(p_dim - 1, max(1, int(round(p_dim * fraction))))
        else:
            drop_top = min(5, max(1, ms1_scaled.shape[0] // 20))
        cs_vec = np.asarray(
            estimate_Cs_from_MS(
                [ms1_scaled, ms2_scaled],
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

    angles = np.linspace(0.0, 2.0 * np.pi, num=a_grid, endpoint=False, dtype=np.float64)
    eta_rad = np.deg2rad(stability_eta_deg)
    detections: list[Detection] = []

    def _threshold_from_z(z_plus_val: float) -> float:
        # Per-window delta: add the larger of absolute and relative offsets
        rel = 0.0 if (delta_frac is None) else float(delta_frac) * float(z_plus_val)
        return float(z_plus_val + max(float(delta), rel))

    C_for_mp = c_vec if use_design_c_for_C else c_weights

    def _edge_margin_for(Cs_local: np.ndarray):
        def margin(angle: float, lam_val: float) -> float | None:
            a_vec = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            if nonnegative_a and np.any(a_vec < -1e-8):
                return float("inf")
            try:
                z_plus_local = mp_edge(
                    a_vec.tolist(),
                    # Use chosen mapping for MP edge
                    C_for_mp.tolist(),
                    d_vec.tolist(),
                    n_total,
                    Cs=Cs_local,
                )
            except (RuntimeError, ValueError):
                return None
            threshold_local = _threshold_from_z(z_plus_local)
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

    for theta in angles:
        a_vec = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        if nonnegative_a and np.any(a_vec < -1e-8):
            continue
        try:
            z_plus = mp_edge(
                a_vec.tolist(),
                C_for_mp.tolist(),
                d_vec.tolist(),
                n_total,
                Cs=cs_vec,
            )
        except (RuntimeError, ValueError):
            continue

        sigma_a = _sigma_of_a_from_MS(a_vec, [ms1_scaled, ms2_scaled])
        try:
            eigvals, eigvecs = np.linalg.eigh(sigma_a)
        except np.linalg.LinAlgError:
            continue

        order_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order_idx]
        eigvecs = eigvecs[:, order_idx]

        threshold_main = _threshold_from_z(z_plus)

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
                threshold_low = _threshold_from_z(z_plus_low)
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
                threshold_high = _threshold_from_z(z_plus_high)
            except (RuntimeError, ValueError):
                z_plus_high = float("nan")
                threshold_high = float("nan")
        else:
            z_plus_high = None
            threshold_high = None

        for idx, lam_val in enumerate(eigvals):
            if lam_val < threshold_main:
                break
            margin_main = lam_val - threshold_main
            if margin_main < 0.0:
                continue
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
                    continue
                t_vals = None
                t_target = None

            if use_tvector:
                if t_vals is None or t_target is None or abs(t_target) <= eps:
                    continue
                t_off = np.delete(t_vals, target_r)
                if t_off.size and float(np.max(np.abs(t_off))) > eps:
                    continue

            if t_target is None or abs(t_target) < 1e-12:
                mu_hat = float(lam_val)
            else:
                mu_hat = float(lam_val / t_target)
            component_vals = [
                float(eigvecs[:, idx].T @ component @ eigvecs[:, idx])
                for component in sigma_components
            ]
            target_component = component_vals[target_r]
            # Component-energy gating: keep modest guard above epsilon
            threshold = max(eps, 0.5 * abs(mu_hat))
            if abs(target_component) <= threshold:
                continue
            if any(
                abs(component_vals[j]) > threshold
                for j in range(len(component_vals))
                if j != target_r
            ):
                continue

            margin_plus = _edge_margin(theta + eta_rad, lam_val)
            if margin_plus is None or margin_plus < 0.0:
                continue
            margin_minus = _edge_margin(theta - eta_rad, lam_val)
            if margin_minus is None or margin_minus < 0.0:
                continue
            stability_margin = float(min(margin_main, margin_plus, margin_minus))

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
                mp_plus_low = _edge_margin_low(theta + eta_rad, lam_val)
                mp_minus_low = _edge_margin_low(theta - eta_rad, lam_val)
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
                mp_plus_high = _edge_margin_high(theta + eta_rad, lam_val)
                mp_minus_high = _edge_margin_high(theta - eta_rad, lam_val)
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
                z_plus=float(z_plus),
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
            )
            detections.append(detection)

    merged = _merge_detections(detections)
    return merged
