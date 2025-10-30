from __future__ import annotations

"""Utility helpers for gating de-alias detections prior to substitution."""

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np


def _as_float(value: Any, *, default: float = float("nan")) -> float:
    """Best-effort conversion to float with NaN fallback."""

    if value is None:
        return default
    if isinstance(value, float):
        return value
    if isinstance(value, (int, np.integer)):
        return float(value)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted


def _score_detection(det: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return score tuple (primary score, edge margin, lambda) for ordering."""

    energy = _as_float(det.get("target_energy"), default=0.0)
    if math.isnan(energy):
        energy = 0.0
    stability = _as_float(det.get("stability_margin"), default=0.0)
    if math.isnan(stability):
        stability = 0.0
    # Score emphasises high energy and stability; guard against negatives.
    primary = max(energy, 0.0) * max(stability, 0.0)
    edge_margin = _as_float(det.get("edge_margin"), default=float("-inf"))
    if math.isnan(edge_margin):
        edge_margin = float("-inf")
    lam_val = _as_float(det.get("lambda_hat"), default=float("-inf"))
    if math.isnan(lam_val):
        lam_val = float("-inf")
    return (primary, edge_margin, lam_val)


def count_isolated_outliers(
    eigs: Iterable[Any],
    edge: float | None,
    stability: Iterable[Any] | None = None,
) -> int:
    """Count isolated spikes relative to the MP edge and stability.

    The function accepts either raw eigenvalues or detection dictionaries. When
    detections provide ``pre_outlier_count`` it is used directly to determine
    isolation (exactly one spike above the edge). Otherwise the count falls back
    to comparing eigenvalues against the provided edge and stability margins.
    """

    items = list(eigs)
    if not items:
        return 0

    pre_counts: list[int | None] = []
    eig_values: list[float] = []
    stab_values: list[float] = []

    for item in items:
        if isinstance(item, Mapping):
            raw_pre = item.get("pre_outlier_count")
            if raw_pre is None:
                pre_counts.append(None)
            else:
                try:
                    pre_counts.append(int(raw_pre))
                except (TypeError, ValueError):
                    pre_counts.append(None)
            eig_values.append(_as_float(item.get("lambda_hat")))
            stab_values.append(_as_float(item.get("stability_margin")))
        else:
            pre_counts.append(None)
            eig_values.append(_as_float(item))
            stab_values.append(float("nan"))

    # Prefer explicit counts from detections when available.
    if any(count is not None for count in pre_counts):
        return sum(1 for count in pre_counts if count == 1)

    edge_val = _as_float(edge)
    eig_array = np.asarray(eig_values, dtype=np.float64)
    mask = np.isfinite(eig_array)
    if np.isfinite(edge_val):
        mask &= eig_array > edge_val

    if stability is not None:
        stability_array = np.asarray(list(stability), dtype=np.float64)
    else:
        stability_array = np.asarray(stab_values, dtype=np.float64)
    if stability_array.size == eig_array.size:
        mask &= stability_array > 0.0

    return int(np.count_nonzero(mask))


def select_top_k(
    detections: Iterable[Mapping[str, Any]],
    k: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Select the top-k detections ranked by score = energy * stability.

    Returns
    -------
    (selected, discarded)
        The highest scoring ``k`` detections followed by the remainder.
    """

    candidates: list[Mapping[str, Any]] = [
        det for det in detections if isinstance(det, Mapping)
    ]
    if k <= 0:
        return [], candidates
    if len(candidates) <= k:
        return candidates, []

    ranked = sorted(candidates, key=_score_detection, reverse=True)
    selected = ranked[:k]
    discarded = ranked[k:]
    return selected, discarded


__all__ = ["count_isolated_outliers", "select_top_k"]
