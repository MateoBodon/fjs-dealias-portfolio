from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "GroupingError",
    "group_by_week",
    "group_by_day_of_week",
    "group_by_vol_state",
    "group_by_dow_vol",
    "group_by_dow_month",
]


class GroupingError(RuntimeError):
    """Raised when a sliding window cannot be balanced for replicates."""


def _ensure_datetime_index(frame: pd.DataFrame) -> None:
    if frame.index.inferred_type != "datetime64":
        raise GroupingError("Expected a DatetimeIndex for daily grouping.")


def group_by_week(frame: pd.DataFrame, *, replicates: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance a window by complete weeks (default: Monday-aligned business weeks)."""

    _ensure_datetime_index(frame)
    if frame.empty:
        raise GroupingError("Cannot balance empty frame.")

    frame_sorted = frame.sort_index()
    week_ids = frame_sorted.index.to_period("W-MON")
    balanced_blocks: list[pd.DataFrame] = []
    for _, block in frame_sorted.groupby(week_ids):
        if block.shape[0] == replicates:
            balanced_blocks.append(block)
    if not balanced_blocks:
        raise GroupingError("Window does not contain any fully populated weeks.")

    trimmed = pd.concat(balanced_blocks, axis=0)
    trimmed = trimmed.astype(np.float64)
    group_labels = np.repeat(np.arange(len(balanced_blocks)), replicates)
    return trimmed, group_labels.astype(np.intp)


def group_by_day_of_week(
    frame: pd.DataFrame,
    *,
    min_weeks: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance a window by Day-of-Week replicates across complete weeks."""

    _ensure_datetime_index(frame)
    if frame.empty:
        raise GroupingError("Cannot balance empty frame.")

    frame_sorted = frame.sort_index()
    week_ids = frame_sorted.index.to_period("W-MON")
    weekdays = frame_sorted.index.weekday
    complete_weeks: list[list[pd.Timestamp]] = []

    unique_weeks = pd.unique(week_ids)
    for week in unique_weeks:
        mask = week_ids == week
        block_index = frame_sorted.index[mask]
        block_weekdays = weekdays[mask]
        selected: list[pd.Timestamp] = []
        valid = True
        for day in range(5):  # Monday-Friday
            day_mask = block_weekdays == day
            if not np.any(day_mask):
                valid = False
                break
            day_indices = block_index[day_mask]
            selected.append(day_indices[-1])
        if valid:
            complete_weeks.append(selected)

    if len(complete_weeks) < min_weeks:
        raise GroupingError(f"Need at least {min_weeks} complete weeks for Day-of-Week design.")

    ordered_indices: list[pd.Timestamp] = []
    group_labels: list[int] = []
    for day in range(5):
        for week in complete_weeks:
            ordered_indices.append(week[day])
            group_labels.append(day)

    trimmed = frame_sorted.loc[ordered_indices]
    trimmed = trimmed.astype(np.float64)
    return trimmed, np.asarray(group_labels, dtype=np.intp)


def _vol_state_codes(
    frame: pd.DataFrame,
    vol_proxy: pd.Series,
    *,
    calm_threshold: float,
    crisis_threshold: float,
) -> np.ndarray:
    frame_sorted = frame.sort_index()
    proxy_aligned = (
        vol_proxy.reindex(frame_sorted.index, method=None)
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    if proxy_aligned.isna().any():
        raise GroupingError("Missing volatility proxy values for window.")
    values = proxy_aligned.to_numpy(dtype=np.float64)
    states = np.full(values.shape[0], -1, dtype=np.intp)
    calm = max(float(calm_threshold), float("-inf"))
    crisis = min(float(crisis_threshold), float("inf"))
    states[values <= calm] = 0
    states[values >= crisis] = 2
    mid_mask = (values > calm) & (values < crisis)
    states[mid_mask] = 1
    return states


def group_by_vol_state(
    frame: pd.DataFrame,
    *,
    vol_proxy: pd.Series,
    calm_threshold: float,
    crisis_threshold: float,
    min_replicates: int = 4,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance a window by volatility-state buckets (calm/mid/crisis)."""

    _ensure_datetime_index(frame)
    if frame.empty:
        raise GroupingError("Cannot balance empty frame.")

    frame_sorted = frame.sort_index()
    states = _vol_state_codes(
        frame_sorted,
        vol_proxy,
        calm_threshold=calm_threshold,
        crisis_threshold=crisis_threshold,
    )

    present_states = np.sort(np.unique(states))
    valid_states = [int(state) for state in present_states if state in {0, 1, 2}]
    if len(valid_states) < 3:
        raise GroupingError("Volatility-state design requires calm/mid/crisis observations.")

    indices_by_state: dict[int, list[pd.Timestamp]] = {state: [] for state in valid_states}
    for row_index, state in zip(frame_sorted.index, states):
        if state in indices_by_state:
            indices_by_state[state].append(row_index)

    replicate_counts = [len(indices_by_state[state]) for state in valid_states]
    min_count = min(replicate_counts)
    if min_count < min_replicates:
        raise GroupingError(
            f"Need at least {min_replicates} observations per volatility state (got {min_count})."
        )

    ordered_indices: list[pd.Timestamp] = []
    group_labels: list[int] = []
    for state in (0, 1, 2):
        if state not in indices_by_state:
            raise GroupingError("Volatility-state buckets became unbalanced.")
        candidates = indices_by_state[state][:min_count]
        ordered_indices.extend(candidates)
        group_labels.extend([state] * len(candidates))

    trimmed = frame_sorted.loc[ordered_indices]
    trimmed = trimmed.astype(np.float64)
    return trimmed, np.asarray(group_labels, dtype=np.intp)


def group_by_dow_vol(
    frame: pd.DataFrame,
    *,
    vol_proxy: pd.Series,
    calm_threshold: float,
    crisis_threshold: float,
    min_replicates: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance windows by crossing day-of-week with volatility states."""

    _ensure_datetime_index(frame)
    if frame.empty:
        raise GroupingError("Cannot balance empty frame.")

    frame_sorted = frame.sort_index()
    states = _vol_state_codes(
        frame_sorted,
        vol_proxy,
        calm_threshold=calm_threshold,
        crisis_threshold=crisis_threshold,
    )
    weekdays = frame_sorted.index.weekday
    labels = states * 10 + weekdays

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        if label < 0:
            continue
        groups.setdefault(int(label), []).append(idx)

    if not groups:
        raise GroupingError("dow×vol design produced no valid groups.")

    insufficient = [lbl for lbl, idxs in groups.items() if len(idxs) < max(1, min_replicates)]
    if insufficient:
        raise GroupingError(
            "dow×vol design lacks replicates for labels: " + ",".join(map(str, insufficient))
        )

    ordered_positions: list[int] = []
    ordered_labels: list[int] = []
    for label in sorted(groups.keys()):
        idxs = sorted(groups[label])
        ordered_positions.extend(idxs)
        ordered_labels.extend([label] * len(idxs))

    trimmed = frame_sorted.iloc[ordered_positions].astype(np.float64)
    return trimmed, np.asarray(ordered_labels, dtype=np.intp)


def group_by_dow_month(
    frame: pd.DataFrame,
    *,
    min_replicates: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance windows by crossing day-of-week with calendar month."""

    _ensure_datetime_index(frame)
    if frame.empty:
        raise GroupingError("Cannot balance empty frame.")

    frame_sorted = frame.sort_index()
    weekdays = frame_sorted.index.weekday
    months = frame_sorted.index.month
    labels = months * 10 + weekdays

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(idx)

    if not groups:
        raise GroupingError("dow×month design produced no valid groups.")

    insufficient = [lbl for lbl, idxs in groups.items() if len(idxs) < max(1, min_replicates)]
    if insufficient:
        raise GroupingError(
            "dow×month design lacks replicates for labels: " + ",".join(map(str, insufficient))
        )

    ordered_positions: list[int] = []
    ordered_labels: list[int] = []
    for label in sorted(groups.keys()):
        idxs = sorted(groups[label])
        ordered_positions.extend(idxs)
        ordered_labels.extend([label] * len(idxs))

    trimmed = frame_sorted.iloc[ordered_positions].astype(np.float64)
    return trimmed, np.asarray(ordered_labels, dtype=np.intp)
