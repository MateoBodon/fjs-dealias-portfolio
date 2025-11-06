from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from fjs.dealias import dealias_search
from fjs.balanced import mean_squares

DetectionList = Sequence[Mapping[str, object]]


@dataclass(slots=True)
class DetectionArrays:
    lambda_hat: np.ndarray
    z_plus: np.ndarray
    stability_margin: np.ndarray
    admissible: np.ndarray
    pre_count: np.ndarray
    alignment: np.ndarray

    @property
    def size(self) -> int:
        return int(self.lambda_hat.size)


def _extract_detection_arrays(
    detections: DetectionList,
    *,
    alignment_min: float,
    require_isolated: bool,
) -> DetectionArrays:
    if not detections:
        empty = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=bool)
        empty_int = np.empty(0, dtype=np.int8)
        return DetectionArrays(
            lambda_hat=empty,
            z_plus=empty,
            stability_margin=empty,
            admissible=empty_bool,
            pre_count=empty_int,
            alignment=empty,
        )

    lambda_vals = []
    z_vals = []
    stability_vals = []
    admissible_vals = []
    pre_counts = []
    alignments = []

    for det in detections:
        lambda_vals.append(float(det.get("lambda_hat", np.nan)))
        z_vals.append(float(det.get("z_plus", np.nan)))
        stability_vals.append(float(det.get("stability_margin", np.nan)))

        admissible = det.get("admissible_root", True)
        admissible_vals.append(bool(True if admissible is None else admissible))

        pre_count_val = det.get("pre_outlier_count")
        if pre_count_val is None:
            pre_counts.append(1)
        else:
            try:
                pre_counts.append(int(pre_count_val))
            except (TypeError, ValueError):
                pre_counts.append(-1)

        alignments.append(float(det.get("alignment_cos", 1.0)))

    lambda_arr = np.asarray(lambda_vals, dtype=np.float64)
    z_arr = np.asarray(z_vals, dtype=np.float64)
    stability_arr = np.asarray(stability_vals, dtype=np.float64)
    admissible_arr = np.asarray(admissible_vals, dtype=bool)
    pre_count_arr = np.asarray(pre_counts, dtype=np.int16)
    alignment_arr = np.asarray(alignments, dtype=np.float64)

    if require_isolated:
        isolated_mask = pre_count_arr == 1
    else:
        isolated_mask = pre_count_arr >= 1

    alignment_mask = alignment_arr >= float(alignment_min)

    admissible_arr &= isolated_mask & alignment_mask

    return DetectionArrays(
        lambda_hat=lambda_arr,
        z_plus=z_arr,
        stability_margin=stability_arr,
        admissible=admissible_arr,
        pre_count=pre_count_arr,
        alignment=alignment_arr,
    )


def _evaluate_delta_grid(
    arrays: DetectionArrays,
    *,
    delta_abs: float,
    delta_frac_values: np.ndarray,
) -> np.ndarray:
    if arrays.size == 0 or not delta_frac_values.size:
        return np.zeros((delta_frac_values.size,), dtype=bool)

    lam = arrays.lambda_hat
    z_plus = arrays.z_plus
    valid = np.isfinite(lam) & np.isfinite(z_plus) & arrays.admissible
    if not np.any(valid):
        return np.zeros((delta_frac_values.size,), dtype=bool)

    lam = lam[valid]
    z_plus = z_plus[valid]

    z_plus = np.clip(z_plus, a_min=0.0, a_max=None)

    delta_frac_values = np.asarray(delta_frac_values, dtype=np.float64)
    rel_component = delta_frac_values[:, None] * z_plus[None, :]
    rel_component = np.maximum(rel_component, delta_abs)
    thresholds = z_plus[None, :] + rel_component
    margins = lam[None, :] - thresholds
    return np.any(margins >= 0.0, axis=1)


def evaluate_threshold_grid(
    observations: np.ndarray,
    groups: np.ndarray,
    *,
    delta_abs: float,
    eps: float,
    edge_modes: Sequence[str],
    delta_frac_values: Sequence[float],
    stability_values: Sequence[float],
    q_max: int,
    a_grid: int,
    require_isolated: bool = True,
    alignment_min: float = 0.0,
    stats: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    _ = q_max
    if stats is None:
        stats = mean_squares(observations, groups)

    delta_frac_arr = np.asarray(delta_frac_values, dtype=np.float64)
    stability_arr = np.asarray(stability_values, dtype=np.float64)

    results: dict[str, np.ndarray] = {}
    for mode in edge_modes:
        mode_results = np.zeros((delta_frac_arr.size, stability_arr.size), dtype=bool)
        for idx, stability_eta in enumerate(stability_arr):
            detections = dealias_search(
                np.asarray(observations, dtype=np.float64),
                np.asarray(groups, dtype=np.intp),
                target_r=0,
                delta=float(delta_abs),
                delta_frac=None,
                eps=float(eps),
                stability_eta_deg=float(stability_eta),
                use_tvector=require_isolated,
                off_component_leak_cap=0.3,
                stats=dict(stats),
                edge_mode=str(mode),
                a_grid=int(a_grid),
            )
            arrays = _extract_detection_arrays(
                detections,
                alignment_min=alignment_min,
                require_isolated=require_isolated,
            )
            delta_mask = _evaluate_delta_grid(
                arrays,
                delta_abs=float(delta_abs),
                delta_frac_values=delta_frac_arr,
            )
            mode_results[:, idx] = delta_mask
        results[str(mode)] = mode_results
    return results
