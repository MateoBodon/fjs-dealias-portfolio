from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["NaNPolicyResult", "NaNPolicyTelemetry", "apply_nan_policy"]


@dataclass(frozen=True, slots=True)
class NaNPolicyTelemetry:
    assets_original: int
    assets_retained: int
    assets_dropped: int
    rows_original: int
    rows_retained: int
    rows_dropped: int


@dataclass(frozen=True, slots=True)
class NaNPolicyResult:
    frame: pd.DataFrame
    labels: np.ndarray
    telemetry: NaNPolicyTelemetry


def apply_nan_policy(
    frame: pd.DataFrame,
    group_labels: Sequence[int],
    *,
    max_missing_asset: float,
    max_missing_group_row: float,
) -> NaNPolicyResult:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if max_missing_asset < 0 or max_missing_asset > 1:
        raise ValueError("max_missing_asset must lie in [0, 1].")
    if max_missing_group_row < 0 or max_missing_group_row > 1:
        raise ValueError("max_missing_group_row must lie in [0, 1].")

    labels = np.asarray(group_labels, dtype=np.intp)
    if frame.shape[0] != labels.size:
        raise ValueError("group_labels must align with frame rows.")

    rows_original = int(frame.shape[0])
    assets_original = int(frame.shape[1])

    if rows_original == 0 or assets_original == 0:
        telemetry = NaNPolicyTelemetry(
            assets_original=assets_original,
            assets_retained=assets_original,
            assets_dropped=0,
            rows_original=rows_original,
            rows_retained=rows_original,
            rows_dropped=0,
        )
        return NaNPolicyResult(frame.copy(), labels.copy(), telemetry)

    asset_missing = frame.isna().mean(axis=0).to_numpy(dtype=np.float64)
    asset_keep_mask = asset_missing <= float(max_missing_asset) + 1e-12
    kept_columns = frame.columns[asset_keep_mask]
    filtered_frame = frame.loc[:, kept_columns].copy()

    assets_retained = int(filtered_frame.shape[1])
    assets_dropped = max(0, assets_original - assets_retained)

    if assets_retained == 0:
        empty = frame.iloc[0:0].copy()
        telemetry = NaNPolicyTelemetry(
            assets_original=assets_original,
            assets_retained=0,
            assets_dropped=assets_original,
            rows_original=rows_original,
            rows_retained=0,
            rows_dropped=rows_original,
        )
        return NaNPolicyResult(empty, labels[:0], telemetry)

    row_missing = filtered_frame.isna().mean(axis=1).to_numpy(dtype=np.float64)
    row_keep_mask = row_missing <= float(max_missing_group_row) + 1e-12

    kept_positions = np.where(row_keep_mask)[0]
    filtered_frame = filtered_frame.iloc[kept_positions].copy()
    filtered_labels = labels[kept_positions]

    rows_retained = int(filtered_frame.shape[0])
    rows_dropped = max(0, rows_original - rows_retained)

    telemetry = NaNPolicyTelemetry(
        assets_original=assets_original,
        assets_retained=assets_retained,
        assets_dropped=assets_dropped,
        rows_original=rows_original,
        rows_retained=rows_retained,
        rows_dropped=rows_dropped,
    )
    return NaNPolicyResult(filtered_frame, filtered_labels, telemetry)
