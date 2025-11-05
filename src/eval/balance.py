from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["BalanceResult", "BalanceTelemetry", "build_balanced_window"]


@dataclass(frozen=True, slots=True)
class BalanceTelemetry:
    reps_per_group: dict[int, int]
    target_replicates: int
    rows_original: int
    rows_retained: int
    rows_dropped: int
    assets_original: int
    assets_retained: int
    assets_dropped: int


@dataclass(frozen=True, slots=True)
class BalanceResult:
    frame: pd.DataFrame
    labels: np.ndarray
    reason: str
    telemetry: BalanceTelemetry


def build_balanced_window(
    frame: pd.DataFrame,
    group_labels: Sequence[int],
    *,
    min_replicates: int,
) -> BalanceResult:
    if min_replicates < 0:
        raise ValueError("min_replicates must be non-negative.")

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    labels = np.asarray(group_labels, dtype=np.intp)
    if frame.shape[0] != labels.size:
        raise ValueError("group_labels must align with frame rows.")

    rows_original = int(frame.shape[0])
    assets_original = int(frame.shape[1])

    if rows_original == 0 or assets_original == 0:
        empty = frame.iloc[0:0].copy()
        telemetry = BalanceTelemetry(
            reps_per_group={},
            target_replicates=0,
            rows_original=rows_original,
            rows_retained=0,
            rows_dropped=rows_original,
            assets_original=assets_original,
            assets_retained=0,
            assets_dropped=assets_original,
        )
        return BalanceResult(empty, labels[:0], "empty", telemetry)

    unique_groups = pd.unique(labels)
    if unique_groups.size == 0:
        telemetry = BalanceTelemetry(
            reps_per_group={},
            target_replicates=0,
            rows_original=rows_original,
            rows_retained=0,
            rows_dropped=rows_original,
            assets_original=assets_original,
            assets_retained=0,
            assets_dropped=assets_original,
        )
        return BalanceResult(frame.iloc[0:0].copy(), labels[:0], "no_groups", telemetry)

    indices_by_group: dict[int, list[int]] = {int(group): [] for group in unique_groups}
    for idx, group in enumerate(labels):
        group_key = int(group)
        if group_key not in indices_by_group:
            indices_by_group[group_key] = []
        indices_by_group[group_key].append(idx)

    replicate_counts = {group: len(idxs) for group, idxs in indices_by_group.items()}
    non_zero_counts = [count for count in replicate_counts.values() if count > 0]
    target_replicates = int(min(non_zero_counts)) if non_zero_counts else 0

    if target_replicates == 0:
        telemetry = BalanceTelemetry(
            reps_per_group={group: 0 for group in replicate_counts},
            target_replicates=0,
            rows_original=rows_original,
            rows_retained=0,
            rows_dropped=rows_original,
            assets_original=assets_original,
            assets_retained=0,
            assets_dropped=assets_original,
        )
        return BalanceResult(frame.iloc[0:0].copy(), labels[:0], "no_replicates", telemetry)

    keep_positions: list[int] = []
    for group in unique_groups:
        group_indices = indices_by_group[int(group)]
        keep_positions.extend(group_indices[:target_replicates])
    keep_positions = sorted(keep_positions)

    balanced_frame = frame.iloc[keep_positions].copy()
    balanced_labels = labels[keep_positions]

    assets_retained = assets_original
    assets_dropped = 0
    if balanced_frame.shape[0] > 0 and balanced_frame.shape[1] > 0:
        group_asset_sets: list[set[str]] = []
        for group in pd.unique(balanced_labels):
            mask = balanced_labels == group
            group_block = balanced_frame.iloc[np.where(mask)[0]]
            valid_columns = group_block.columns[group_block.notna().all(axis=0)]
            group_asset_sets.append(set(valid_columns))
        if group_asset_sets:
            intersect_columns = set(balanced_frame.columns)
            for col_set in group_asset_sets:
                intersect_columns &= col_set
            ordered_columns = [col for col in balanced_frame.columns if col in intersect_columns]
            assets_retained = len(ordered_columns)
            assets_dropped = max(0, assets_original - assets_retained)
            balanced_frame = balanced_frame.loc[:, ordered_columns]
        else:
            balanced_frame = balanced_frame.iloc[0:0]
            balanced_labels = balanced_labels[:0]
            assets_retained = 0
            assets_dropped = assets_original

    rows_retained = int(balanced_frame.shape[0])
    rows_dropped = max(0, rows_original - rows_retained)

    final_counts = {}
    if rows_retained > 0:
        for group in pd.unique(balanced_labels):
            final_counts[int(group)] = int(np.sum(balanced_labels == group))
    else:
        final_counts = {group: 0 for group in replicate_counts}

    if rows_retained == 0 or assets_retained == 0:
        reason = "empty_after_balance"
    elif target_replicates < int(min_replicates):
        reason = "insufficient_reps"
    elif rows_dropped > 0 or assets_dropped > 0:
        reason = "trimmed"
    else:
        reason = "ok"

    telemetry = BalanceTelemetry(
        reps_per_group=final_counts,
        target_replicates=target_replicates,
        rows_original=rows_original,
        rows_retained=rows_retained,
        rows_dropped=rows_dropped,
        assets_original=assets_original,
        assets_retained=assets_retained,
        assets_dropped=assets_dropped,
    )
    return BalanceResult(balanced_frame, balanced_labels, reason, telemetry)
