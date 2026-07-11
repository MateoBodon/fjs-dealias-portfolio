from __future__ import annotations

import base64
import copy
import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fjs.real_design_contract import (
    SOURCE_COLUMNS,
    FactorBinding,
    SourcePartitionBinding,
    _normalize_source_frame,
    _serialize_int_matrix,
    _serialize_missingness,
    file_sha256,
    stable_sha256,
)

ROLLING_GEOMETRY_SCHEMA = "fjs-rolling-geometry-proof/v1"
ROLLING_MANIFEST_SCHEMA = "fjs-rolling-geometry-manifest/v1"
ROLLING_CONTRACT_ID = "fjs-m5-rolling-156-week-geometry-v1"
HEADLINE_CLAIM_ID = "fjs-m5-headline-calibration-claim-v1"

WARMUP_START = pd.Timestamp("2010-01-01")
DEVELOPMENT_START = pd.Timestamp("2013-01-01")
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
FINAL_HOLDOUT_START = pd.Timestamp("2025-01-01")

WINDOW_WEEKS = 156
UNIVERSE_SIZE = 60
TARGET_RANK = 1
REPLICATES = 5
MIN_ASSET_OBSERVATIONS = 390
MIN_OBSERVED_ASSETS_PER_DATE = 57
MIN_PAIRWISE_OBSERVATIONS = 500
MIN_CALENDAR_DATES = 720
MIN_COMPLETE_BALANCED_WEEKS = 78
MAX_PANEL_MISSING_FRACTION = 0.10
MAX_CAP_STALENESS_DAYS = 10
BOUNDARY_MULTIPLIER = 1.5
BOUNDED_PROOF_ENDPOINT_MONTH = "2013-01"

GEOMETRY_COLUMNS = ("permno", "dlycaldt", "dlycap")
FORBIDDEN_PROOF_KEYS = {
    "aws",
    "calibration_results",
    "detector_results",
    "dlyret",
    "outcome",
    "outcomes",
    "performance",
    "returns",
    "submission",
}


def development_endpoint_months() -> list[str]:
    return [
        f"{year:04d}-{month:02d}"
        for year in range(2013, 2019)
        for month in range(1, 13)
    ]


def headline_calibration_claim() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim_id": HEADLINE_CLAIM_ID,
        "scope": (
            "Every frozen v5 development geometry stratum; no pooled rescue and "
            "no confirmation or holdout tuning."
        ),
        "fjs_only_required": True,
        "null_gate": {
            "nominal_rate": 0.05,
            "exact_95pct_interval_must_contain_nominal": True,
            "exact_95pct_upper_bound_max": 0.075,
        },
        "planted_gate": {
            "population_spike_multiple_of_boundary": BOUNDARY_MULTIPLIER,
            "detection_rate_min": 0.80,
            "acceptance_rate_min": 0.80,
            "null_to_power_gain_min": 0.50,
            "squared_cosine_min": 0.80,
            "planted_component_attribution_min": 0.90,
            "nuisance_component_attribution_max": 0.10,
            "power_curve_nondecreasing": True,
        },
        "invariance_required": [
            "standardized_rescaling",
            "deterministic_row_order",
            "asset_permutation_with_direction_mapback",
            "group_label_permutation",
        ],
        "headline_success_policy": "all_frozen_geometry_strata_must_pass",
        "outcomes_observed_when_frozen": False,
        "holdout_2025_opened": False,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def rolling_geometry_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": ROLLING_CONTRACT_ID,
        "window": {
            "calendar_week_count": WINDOW_WEEKS,
            "week_anchor": "monday",
            "replicate_slots": REPLICATES,
            "endpoint_schedule": "last_factor_date_of_each_development_month",
            "development_endpoint_months": development_endpoint_months(),
            "primary_forward_blocks": (
                "monthly endpoint-to-next-endpoint and therefore non-overlapping"
            ),
            "bounded_real_proof_endpoint_month": BOUNDED_PROOF_ENDPOINT_MONTH,
        },
        "universe": {
            "identity": "PERMNO",
            "size": UNIVERSE_SIZE,
            "point_in_time": True,
            "rank_field": "dlycap",
            "rank_direction": "descending",
            "tie_break": "permno_ascending",
            "min_past_observations": MIN_ASSET_OBSERVATIONS,
            "max_cap_staleness_days": MAX_CAP_STALENESS_DAYS,
            "future_backfill_forbidden": True,
        },
        "coverage_gates": {
            "calendar_week_count_exact": WINDOW_WEEKS,
            "calendar_dates_min": MIN_CALENDAR_DATES,
            "selected_assets_exact": UNIVERSE_SIZE,
            "observations_per_asset_min": MIN_ASSET_OBSERVATIONS,
            "observed_assets_per_date_min": MIN_OBSERVED_ASSETS_PER_DATE,
            "pairwise_observations_min": MIN_PAIRWISE_OBSERVATIONS,
            "complete_balanced_weeks_min": MIN_COMPLETE_BALANCED_WEEKS,
            "natural_missing_cells_min": 1,
            "panel_missing_fraction_max": MAX_PANEL_MISSING_FRACTION,
        },
        "target_boundary": {
            "target_rank": TARGET_RANK,
            "replicates": REPLICATES,
            "between_bulk_variance": 0.0,
            "residual_bulk_variance": 1.0,
            "population_spike_multiplier": BOUNDARY_MULTIPLIER,
            "between_aspect_ratio_must_be_below_one": True,
        },
        "input_boundary": {
            "warmup_start": WARMUP_START.date().isoformat(),
            "development_start": DEVELOPMENT_START.date().isoformat(),
            "development_end": DEVELOPMENT_END.date().isoformat(),
            "final_holdout_start": FINAL_HOLDOUT_START.date().isoformat(),
            "return_values_persisted": False,
            "return_presence_and_numeric_validity_read": True,
            "detector_outcomes_present": False,
            "aws_execution_authorized": False,
        },
        "headline_calibration_claim": headline_calibration_claim(),
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def week_start(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value).normalize()
    return timestamp - pd.Timedelta(days=int(timestamp.weekday()))


def rolling_window_start(formation_date: pd.Timestamp) -> pd.Timestamp:
    return week_start(formation_date) - pd.Timedelta(weeks=WINDOW_WEEKS - 1)


def source_months_for_window(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts < WARMUP_START or end_ts > DEVELOPMENT_END or start_ts > end_ts:
        raise ValueError("V5 source months must remain inside 2010-2018.")
    return [
        period.strftime("%Y-%m")
        for period in pd.period_range(start_ts, end_ts, freq="M")
    ]


def _validate_source_binding_set(
    bindings: Sequence[SourcePartitionBinding], *, start: str, end: str
) -> list[str]:
    expected = source_months_for_window(pd.Timestamp(start), pd.Timestamp(end))
    observed = [binding.partition.removeprefix("month=") for binding in bindings]
    if observed != expected or len(observed) != len(set(observed)):
        raise ValueError(
            "V5 source bindings must be the exact unique ordered rolling-window "
            f"month set: expected={expected}, observed={observed}."
        )
    for month, binding in zip(expected, bindings, strict=True):
        if Path(binding.path).parent.name != f"month={month}":
            raise ValueError(f"V5 source path/partition mismatch for {month}.")
    return expected


@dataclass(frozen=True)
class RollingGeometrySpec:
    endpoint_month: str
    formation_date: str
    window_start: str
    window_end: str
    proof_only: bool = True

    def validate(self) -> None:
        if self.endpoint_month not in development_endpoint_months():
            raise ValueError("V5 endpoint month must be in 2013-2018 development.")
        formation = pd.Timestamp(self.formation_date)
        start = pd.Timestamp(self.window_start)
        end = pd.Timestamp(self.window_end)
        if formation.to_period("M").strftime("%Y-%m") != self.endpoint_month:
            raise ValueError("Formation date does not match its endpoint month.")
        if formation < DEVELOPMENT_START or formation > DEVELOPMENT_END:
            raise ValueError("Formation date must remain in development.")
        if start != rolling_window_start(formation) or end != formation:
            raise ValueError("V5 window must be the exact rolling 156-week window.")
        if start < WARMUP_START:
            raise ValueError("V5 window may not precede the frozen warm-up start.")
        if max(formation, start, end) >= FINAL_HOLDOUT_START:
            raise ValueError("The 2025 final holdout must remain unopened.")
        if self.proof_only and self.endpoint_month != BOUNDED_PROOF_ENDPOINT_MONTH:
            raise ValueError("The bounded real proof endpoint is frozen to 2013-01.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def resolve_spec(
    endpoint_month: str, factor_dates: Sequence[pd.Timestamp]
) -> RollingGeometrySpec:
    if endpoint_month not in development_endpoint_months():
        raise ValueError("Endpoint month is outside frozen development.")
    dates = pd.DatetimeIndex(pd.to_datetime(list(factor_dates))).sort_values().unique()
    candidates = dates[dates.to_period("M") == pd.Period(endpoint_month, freq="M")]
    if len(candidates) == 0:
        raise ValueError(f"No factor calendar date exists for {endpoint_month}.")
    formation = pd.Timestamp(candidates[-1])
    start = rolling_window_start(formation)
    return RollingGeometrySpec(
        endpoint_month=endpoint_month,
        formation_date=formation.date().isoformat(),
        window_start=start.date().isoformat(),
        window_end=formation.date().isoformat(),
        proof_only=True,
    )


def load_bound_factor_calendar(
    binding: FactorBinding, *, start: str, end: str
) -> pd.DatetimeIndex:
    path = Path(binding.path)
    if file_sha256(path) != binding.sha256:
        raise ValueError("Factor file changed after binding.")
    frame = pd.read_csv(path, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values()
    if bool(dates.duplicated().any()):
        raise ValueError("Factor calendar dates must be unique.")
    subset = dates.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))]
    if subset.empty:
        raise ValueError("Factor calendar does not cover the requested window.")
    return pd.DatetimeIndex(subset)


def _deduplicate_partition(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicated = frame.duplicated(subset=["dlycaldt", "permno"], keep=False)
    if not bool(duplicated.any()):
        return frame, 0
    comparison = [name for name in SOURCE_COLUMNS if name not in {"dlycaldt", "permno"}]
    conflicts = []
    for (date_value, permno), group in frame.loc[duplicated].groupby(
        ["dlycaldt", "permno"], sort=True, dropna=False
    ):
        varying = [
            name for name in comparison if int(group[name].nunique(dropna=False)) > 1
        ]
        if varying:
            conflicts.append(
                {
                    "date": pd.Timestamp(date_value).date().isoformat(),
                    "permno": int(permno),
                    "varying_required_fields": varying,
                }
            )
    if conflicts:
        raise ValueError(f"Conflicting V5 date/PERMNO duplicates: {conflicts[:5]}")
    before = len(frame)
    deduplicated = frame.drop_duplicates(subset=list(SOURCE_COLUMNS), keep="first")
    if bool(deduplicated.duplicated(subset=["dlycaldt", "permno"]).any()):
        raise ValueError("Exact duplicate collapse did not restore V5 identity.")
    return deduplicated, before - len(deduplicated)


def geometry_logical_sha256(frame: pd.DataFrame) -> str:
    if tuple(frame.columns) != GEOMETRY_COLUMNS:
        raise ValueError("Logical V5 geometry hash requires exact geometry columns.")
    ordered = frame.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)
    digest = hashlib.sha256()
    row_dtype = np.dtype([("dlycaldt", "<i8"), ("permno", "<i8"), ("dlycap", "<f8")])
    for start in range(0, len(ordered), 100_000):
        chunk = ordered.iloc[start : start + 100_000]
        values = np.empty(len(chunk), dtype=row_dtype)
        values["dlycaldt"] = chunk["dlycaldt"].astype("int64").to_numpy()
        values["permno"] = chunk["permno"].astype("int64").to_numpy()
        values["dlycap"] = chunk["dlycap"].astype("float64").to_numpy()
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def load_geometry_only_sources(
    bindings: Sequence[SourcePartitionBinding],
    *,
    start: str,
    end: str,
    chunksize: int = 25_000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts < WARMUP_START or end_ts > DEVELOPMENT_END or start_ts > end_ts:
        raise ValueError("V5 geometry loading is restricted to 2010-2018.")
    if not bindings or chunksize <= 0:
        raise ValueError("V5 geometry loading needs bindings and positive chunksize.")
    expected_months = _validate_source_binding_set(
        bindings, start=start_ts.date().isoformat(), end=end_ts.date().isoformat()
    )

    geometry_parts: list[pd.DataFrame] = []
    scans: list[dict[str, Any]] = []
    duplicate_total = 0
    for binding in bindings:
        source = Path(binding.path)
        if file_sha256(source) != binding.sha256:
            raise ValueError(f"CRSP source changed after V5 binding: {source}")
        scanned = 0
        monthly_clean: list[pd.DataFrame] = []
        for raw in pd.read_csv(
            source,
            usecols=list(SOURCE_COLUMNS),
            chunksize=chunksize,
            low_memory=False,
        ):
            scanned += len(raw)
            clean = _normalize_source_frame(raw)
            clean = clean.loc[clean["dlycaldt"].between(start_ts, end_ts)]
            if not clean.empty:
                monthly_clean.append(clean)
        if scanned != binding.receipt_rows:
            raise ValueError(
                f"V5 source scan did not consume the full receipted partition "
                f"{binding.partition}: scanned={scanned}, "
                f"receipt={binding.receipt_rows}."
            )
        if monthly_clean:
            month_frame = pd.concat(monthly_clean, ignore_index=True)
            month_frame, collapsed = _deduplicate_partition(month_frame)
            duplicate_total += collapsed
            geometry_parts.append(month_frame.loc[:, list(GEOMETRY_COLUMNS)].copy())
            retained = len(month_frame)
        else:
            retained = 0
        scans.append(
            {
                "binding_sha256": binding.binding_sha256,
                "rows_scanned": scanned,
                "rows_receipted": binding.receipt_rows,
                "rows_after_frozen_filters_and_date_bounds": retained,
                "scan_truncated": False,
            }
        )
    if not geometry_parts:
        raise ValueError("No observations survive the frozen V5 filters.")
    geometry = pd.concat(geometry_parts, ignore_index=True)
    geometry = geometry.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)
    if bool(geometry.duplicated(subset=["dlycaldt", "permno"]).any()):
        raise ValueError("V5 source partitions overlap or retain duplicate identities.")
    scan: dict[str, Any] = {
        "requested_start": start_ts.date().isoformat(),
        "requested_end": end_ts.date().isoformat(),
        "chunksize": chunksize,
        "partitions": scans,
        "expected_source_months": expected_months,
        "source_binding_set_digest": stable_sha256(
            [binding.binding_sha256 for binding in bindings]
        ),
        "rows_after_all_filters": len(geometry),
        "exact_duplicate_rows_collapsed": duplicate_total,
        "return_values_persisted": False,
        "return_presence_and_validity_read": True,
        "logical_geometry_sha256": geometry_logical_sha256(geometry),
    }
    scan["sha256"] = stable_sha256(scan)
    return geometry, scan


def _target_boundary(complete_groups: int) -> dict[str, Any]:
    bulk_dimension = UNIVERSE_SIZE - TARGET_RANK
    between_dof = complete_groups - 1
    within_dof = complete_groups * (REPLICATES - 1)
    if between_dof <= 0 or within_dof <= 0:
        raise ValueError("At least two complete balanced weeks are required.")
    y_between = bulk_dimension / float(between_dof)
    y_within = bulk_dimension / float(within_dof)
    population_boundary = math.sqrt(y_between + y_within) / REPLICATES
    payload: dict[str, Any] = {
        "boundary_kind": "population_between_covariance_eigenvalue",
        "p_assets": UNIVERSE_SIZE,
        "target_rank": TARGET_RANK,
        "complete_balanced_groups": complete_groups,
        "replicates": REPLICATES,
        "bulk_dimension": bulk_dimension,
        "between_degrees_of_freedom": between_dof,
        "within_degrees_of_freedom": within_dof,
        "between_aspect_ratio": y_between,
        "within_aspect_ratio": y_within,
        "population_eigenvalue_boundary": population_boundary,
        "predeclared_power_spike": BOUNDARY_MULTIPLIER * population_boundary,
        "finite_positive": math.isfinite(population_boundary)
        and population_boundary > 0.0,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def _coverage_gate_results(
    metrics: Mapping[str, Any], boundary: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "calendar_week_count_exact": int(metrics["n_groups"]) == WINDOW_WEEKS,
        "calendar_dates_min": int(metrics["n_dates"]) >= MIN_CALENDAR_DATES,
        "selected_assets_exact": int(metrics["p_assets"]) == UNIVERSE_SIZE,
        "observations_per_asset_min": int(metrics["observations_per_asset_min"])
        >= MIN_ASSET_OBSERVATIONS,
        "observed_assets_per_date_min": int(metrics["observed_assets_per_date_min"])
        >= MIN_OBSERVED_ASSETS_PER_DATE,
        "pairwise_observations_min": int(metrics["pairwise_observations_min"])
        >= MIN_PAIRWISE_OBSERVATIONS,
        "complete_balanced_weeks_min": int(metrics["complete_balanced_groups"])
        >= MIN_COMPLETE_BALANCED_WEEKS,
        "natural_missing_cells_min": int(metrics["missing_cells"]) >= 1,
        "panel_missing_fraction_max": float(metrics["missing_fraction"])
        <= MAX_PANEL_MISSING_FRACTION,
        "target_boundary_finite_positive": bool(boundary["finite_positive"]),
        "target_between_aspect_ratio_below_one": float(boundary["between_aspect_ratio"])
        < 1.0,
    }


def _quantiles(values: Sequence[int | float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _geometry_metrics_from_mask(
    mask: np.ndarray,
    labels: pd.DatetimeIndex,
    *,
    eligible_candidate_count: int,
) -> tuple[dict[str, Any], np.ndarray, list[int]]:
    if mask.ndim != 2 or mask.shape[1] != UNIVERSE_SIZE:
        raise ValueError("V5 observed mask must be dates by exactly 60 assets.")
    unique_labels = labels.unique().sort_values()
    group_sizes = [int(np.sum(labels == label)) for label in unique_labels]
    observed_per_asset = mask.sum(axis=0).astype(int)
    observed_per_date = mask.sum(axis=1).astype(int)
    pairwise = mask.astype(np.int32).T @ mask.astype(np.int32)
    missing_cells = int(mask.size - int(mask.sum()))
    complete_groups = 0
    for label in unique_labels:
        rows = np.flatnonzero(labels == label)
        if len(rows) == REPLICATES and bool(mask[rows, :].all()):
            complete_groups += 1
    between_dof = len(unique_labels) - 1
    within_dof = sum(max(value - 1, 0) for value in group_sizes)
    metrics: dict[str, Any] = {
        "p_assets": UNIVERSE_SIZE,
        "n_dates": mask.shape[0],
        "n_groups": len(unique_labels),
        "group_size_histogram": {
            str(key): value for key, value in sorted(Counter(group_sizes).items())
        },
        "complete_balanced_groups": complete_groups,
        "between_degrees_of_freedom": between_dof,
        "within_degrees_of_freedom": within_dof,
        "between_aspect_ratio": (UNIVERSE_SIZE - TARGET_RANK) / between_dof,
        "within_aspect_ratio": (UNIVERSE_SIZE - TARGET_RANK) / within_dof,
        "eligible_candidate_count": int(eligible_candidate_count),
        "observations_per_asset_min": int(observed_per_asset.min()),
        "observations_per_asset_distribution": _quantiles(observed_per_asset),
        "observed_assets_per_date_min": int(observed_per_date.min()),
        "observed_assets_per_date_distribution": _quantiles(observed_per_date),
        "pairwise_observations_min": int(pairwise.min()),
        "pairwise_observations_distribution": _quantiles(pairwise.reshape(-1)),
        "missing_cells": missing_cells,
        "missing_fraction": float(missing_cells / mask.size),
    }
    return metrics, pairwise, group_sizes


def _decode_observed_mask(payload: Mapping[str, Any]) -> np.ndarray:
    if payload.get("encoding") != "packbits-little-hex":
        raise ValueError("V5 observed mask encoding mismatch.")
    raw = bytes.fromhex(str(payload["data"]))
    if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
        raise ValueError("V5 observed mask hash mismatch.")
    shape = tuple(int(value) for value in payload["shape"])
    count = int(np.prod(shape))
    values = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8), bitorder="little", count=count
    )
    return values.astype(bool).reshape(shape)


def _decode_pairwise_counts(payload: Mapping[str, Any]) -> np.ndarray:
    if payload.get("encoding") != "base64" or payload.get("dtype") != "int32-le":
        raise ValueError("V5 pairwise-count encoding mismatch.")
    raw = base64.b64decode(str(payload["data"]), validate=True)
    if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
        raise ValueError("V5 pairwise-count hash mismatch.")
    shape = tuple(int(value) for value in payload["shape"])
    values = np.frombuffer(raw, dtype="<i4")
    if values.size != int(np.prod(shape)):
        raise ValueError("V5 pairwise-count shape mismatch.")
    return values.reshape(shape)


def _factor_binding_from_mapping(payload: Mapping[str, Any]) -> FactorBinding:
    values = dict(payload)
    values["columns"] = tuple(str(value) for value in values["columns"])
    return FactorBinding(**values)


def build_rolling_geometry_proof(
    source_frame: pd.DataFrame,
    *,
    spec: RollingGeometrySpec,
    source_bindings: Sequence[SourcePartitionBinding],
    factor_binding: FactorBinding,
    scan_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    spec.validate()
    if tuple(source_frame.columns) != GEOMETRY_COLUMNS:
        raise ValueError("V5 source frame must contain only geometry columns.")
    formation = pd.Timestamp(spec.formation_date)
    start = pd.Timestamp(spec.window_start)
    expected_months = _validate_source_binding_set(
        source_bindings, start=spec.window_start, end=spec.window_end
    )
    if scan_receipt.get("expected_source_months") != expected_months:
        raise ValueError("V5 scan receipt source-month set mismatch.")
    expected_binding_digest = stable_sha256(
        [binding.binding_sha256 for binding in source_bindings]
    )
    if scan_receipt.get("source_binding_set_digest") != expected_binding_digest:
        raise ValueError("V5 scan receipt source-binding digest mismatch.")
    scan_parts = scan_receipt.get("partitions")
    if not isinstance(scan_parts, list) or len(scan_parts) != len(source_bindings):
        raise ValueError("V5 scan receipt partition count mismatch.")
    for scan, binding in zip(scan_parts, source_bindings, strict=True):
        if (
            scan.get("binding_sha256") != binding.binding_sha256
            or int(scan.get("rows_scanned", -1)) != binding.receipt_rows
            or bool(scan.get("scan_truncated"))
        ):
            raise ValueError("V5 scan receipt does not prove a full partition scan.")
    logical_geometry_sha256 = geometry_logical_sha256(source_frame)
    if scan_receipt.get("logical_geometry_sha256") != logical_geometry_sha256 or int(
        scan_receipt.get("rows_after_all_filters", -1)
    ) != len(source_frame):
        raise ValueError("V5 source frame does not match its logical scan binding.")

    calendar = load_bound_factor_calendar(
        factor_binding, start=spec.window_start, end=spec.window_end
    )
    calendar = calendar[(calendar >= start) & (calendar <= formation)]
    if len(calendar) == 0:
        raise ValueError("V5 factor calendar is empty.")
    labels = pd.DatetimeIndex([week_start(value) for value in calendar])
    unique_labels = labels.unique().sort_values()
    if len(unique_labels) != WINDOW_WEEKS:
        raise ValueError(
            f"V5 window has {len(unique_labels)} calendar weeks; "
            f"{WINDOW_WEEKS} required."
        )

    frame = source_frame.loc[source_frame["dlycaldt"].between(start, formation)].copy()
    histories = frame.groupby("permno")["dlycaldt"].nunique()
    eligible = histories.loc[histories.ge(MIN_ASSET_OBSERVATIONS)].index
    ranking = frame.loc[
        frame["permno"].isin(eligible) & frame["dlycaldt"].le(formation)
    ].copy()
    ranking = ranking.sort_values(["permno", "dlycaldt"])
    ranking = ranking.groupby("permno", as_index=False).tail(1)
    ranking = ranking.loc[
        (formation - ranking["dlycaldt"]).dt.days.le(MAX_CAP_STALENESS_DAYS)
    ].copy()
    ranking = ranking.sort_values(
        ["dlycap", "permno"], ascending=[False, True], kind="mergesort"
    )
    if len(ranking) < UNIVERSE_SIZE:
        raise ValueError(
            f"Only {len(ranking)} V5 candidates satisfy past-only eligibility."
        )
    selected = ranking.head(UNIVERSE_SIZE).copy()
    permnos = selected["permno"].astype(int).tolist()

    observations = frame.loc[
        frame["permno"].isin(permnos) & frame["dlycaldt"].isin(calendar),
        ["dlycaldt", "permno"],
    ].drop_duplicates()
    presence = pd.crosstab(observations["dlycaldt"], observations["permno"])
    presence = presence.reindex(index=calendar, columns=permnos, fill_value=0)
    mask = presence.to_numpy(dtype=bool)
    observed_per_asset = mask.sum(axis=0).astype(int)
    observed_per_date = mask.sum(axis=1).astype(int)
    metrics, pairwise, group_sizes = _geometry_metrics_from_mask(
        mask, labels, eligible_candidate_count=len(ranking)
    )
    boundary = _target_boundary(int(metrics["complete_balanced_groups"]))
    gates = _coverage_gate_results(metrics, boundary)

    members = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        index = permnos.index(int(row.permno))
        members.append(
            {
                "rank": rank,
                "permno": int(row.permno),
                "lagged_market_cap": float(row.dlycap),
                "cap_observation_date": pd.Timestamp(row.dlycaldt).date().isoformat(),
                "window_observations": int(observed_per_asset[index]),
            }
        )

    proof: dict[str, Any] = {
        "schema": ROLLING_GEOMETRY_SCHEMA,
        "cell_id": f"fjs-rolling-geometry-{spec.endpoint_month}-v5",
        "purpose": "Geometry-only bounded proof of frozen rolling FJS design.",
        "claim_boundary": {
            "development_only": True,
            "geometry_only": True,
            "return_values_persisted": False,
            "detector_outcomes_present": False,
            "empirical_claims_allowed": False,
            "aws_execution_authorized": False,
            "holdout_2025_opened": False,
        },
        "contract": rolling_geometry_contract(),
        "headline_calibration_claim": headline_calibration_claim(),
        "spec": spec.to_dict(),
        "source_bindings": [binding.to_dict() for binding in source_bindings],
        "source_geometry_binding": {
            "logical_geometry_sha256": logical_geometry_sha256,
            "row_count": len(source_frame),
            "source_binding_set_digest": expected_binding_digest,
            "source_months": expected_months,
        },
        "factor_binding": factor_binding.to_dict(),
        "scan_receipt": copy.deepcopy(dict(scan_receipt)),
        "calendar": {
            "dates": [pd.Timestamp(value).date().isoformat() for value in calendar],
            "week_labels": [pd.Timestamp(value).date().isoformat() for value in labels],
            "group_sizes": group_sizes,
            "sha256": stable_sha256(
                [pd.Timestamp(value).date().isoformat() for value in calendar]
            ),
        },
        "universe": {
            "members": members,
            "member_set_sha256": stable_sha256(members),
        },
        "missingness": {
            "observed_mask": _serialize_missingness(mask),
            "observed_per_asset": observed_per_asset.tolist(),
            "observed_per_date": observed_per_date.tolist(),
            "pairwise_observation_counts": _serialize_int_matrix(pairwise),
        },
        "geometry_metrics": metrics,
        "target_boundary_feasibility": boundary,
        "coverage_gates": gates,
        "coverage_proof_passed": all(gates.values()),
    }
    proof["proof_digest"] = stable_sha256(proof)
    validate_rolling_geometry_proof(proof, source_frame=source_frame)
    return proof


def _find_forbidden_keys(payload: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key)
            locator = f"{prefix}.{name}" if prefix else name
            if name.lower() in FORBIDDEN_PROOF_KEYS:
                found.append(locator)
            found.extend(_find_forbidden_keys(value, locator))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_find_forbidden_keys(value, f"{prefix}[{index}]"))
    return sorted(found)


def validate_rolling_geometry_proof(
    proof: Mapping[str, Any],
    *,
    source_frame: pd.DataFrame | None = None,
    revalidate_external: bool = False,
) -> None:
    if proof.get("schema") != ROLLING_GEOMETRY_SCHEMA:
        raise ValueError("V5 rolling geometry schema mismatch.")
    if _find_forbidden_keys(proof):
        raise ValueError(
            "V5 rolling geometry proof contains forbidden outcome/return keys."
        )
    if proof.get("contract") != rolling_geometry_contract():
        raise ValueError("V5 rolling geometry contract mismatch.")
    if proof.get("headline_calibration_claim") != headline_calibration_claim():
        raise ValueError("V5 headline calibration claim mismatch.")
    spec = RollingGeometrySpec(**dict(proof["spec"]))
    spec.validate()
    expected_cell = f"fjs-rolling-geometry-{spec.endpoint_month}-v5"
    if proof.get("cell_id") != expected_cell:
        raise ValueError("V5 rolling geometry cell identity mismatch.")

    raw_bindings = proof.get("source_bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError("V5 source bindings are missing.")
    bindings = [SourcePartitionBinding(**dict(raw)) for raw in raw_bindings]
    expected_months = _validate_source_binding_set(
        bindings, start=spec.window_start, end=spec.window_end
    )
    binding_set_digest = stable_sha256([binding.binding_sha256 for binding in bindings])
    geometry_binding = proof.get("source_geometry_binding")
    if not isinstance(geometry_binding, Mapping) or geometry_binding != {
        "logical_geometry_sha256": proof["scan_receipt"]["logical_geometry_sha256"],
        "row_count": proof["scan_receipt"]["rows_after_all_filters"],
        "source_binding_set_digest": binding_set_digest,
        "source_months": expected_months,
    }:
        raise ValueError("V5 logical source geometry binding mismatch.")
    scan = proof["scan_receipt"]
    if (
        scan.get("sha256")
        != stable_sha256({key: value for key, value in scan.items() if key != "sha256"})
        or scan.get("expected_source_months") != expected_months
        or scan.get("source_binding_set_digest") != binding_set_digest
        or scan.get("return_values_persisted") is not False
        or scan.get("return_presence_and_validity_read") is not True
    ):
        raise ValueError("V5 scan receipt boundary mismatch.")
    scan_parts = scan.get("partitions")
    if not isinstance(scan_parts, list) or len(scan_parts) != len(bindings):
        raise ValueError("V5 scan receipt partition count mismatch.")
    for item, binding in zip(scan_parts, bindings, strict=True):
        if (
            item.get("binding_sha256") != binding.binding_sha256
            or int(item.get("rows_scanned", -1)) != binding.receipt_rows
            or bool(item.get("scan_truncated"))
        ):
            raise ValueError("V5 proof does not bind a full source partition scan.")
        if revalidate_external and file_sha256(Path(binding.path)) != binding.sha256:
            raise ValueError(f"V5 source changed after proof: {binding.path}")
    if source_frame is not None:
        if geometry_logical_sha256(source_frame) != geometry_binding[
            "logical_geometry_sha256"
        ] or len(source_frame) != int(geometry_binding["row_count"]):
            raise ValueError("V5 readback source frame changed after the scan receipt.")

    factor_binding = _factor_binding_from_mapping(proof["factor_binding"])
    calendar = load_bound_factor_calendar(
        factor_binding, start=spec.window_start, end=spec.window_end
    )
    expected_dates = [pd.Timestamp(value).date().isoformat() for value in calendar]
    observed_calendar = proof["calendar"]
    if observed_calendar.get("dates") != expected_dates or observed_calendar.get(
        "sha256"
    ) != stable_sha256(expected_dates):
        raise ValueError("V5 proof calendar is not bound to the factor file.")
    labels = pd.DatetimeIndex([week_start(value) for value in calendar])
    expected_labels = [pd.Timestamp(value).date().isoformat() for value in labels]
    if observed_calendar.get("week_labels") != expected_labels:
        raise ValueError("V5 proof week labels do not match the factor calendar.")

    missingness = proof["missingness"]
    mask = _decode_observed_mask(missingness["observed_mask"])
    if mask.shape != (len(calendar), UNIVERSE_SIZE):
        raise ValueError("V5 observed mask shape does not match its calendar.")
    metrics = proof["geometry_metrics"]
    expected_metrics, pairwise, group_sizes = _geometry_metrics_from_mask(
        mask,
        labels,
        eligible_candidate_count=int(metrics["eligible_candidate_count"]),
    )
    if expected_metrics != metrics:
        raise ValueError("V5 geometry metrics do not recompute from the bound mask.")
    if observed_calendar.get("group_sizes") != group_sizes:
        raise ValueError("V5 group sizes do not recompute from the bound calendar.")
    if missingness.get("observed_per_asset") != mask.sum(axis=0).astype(int).tolist():
        raise ValueError("V5 per-asset counts do not recompute from the mask.")
    if missingness.get("observed_per_date") != mask.sum(axis=1).astype(int).tolist():
        raise ValueError("V5 per-date counts do not recompute from the mask.")
    if not np.array_equal(
        _decode_pairwise_counts(missingness["pairwise_observation_counts"]),
        pairwise,
    ):
        raise ValueError("V5 pairwise counts do not recompute from the mask.")

    boundary = proof["target_boundary_feasibility"]
    expected_boundary = _target_boundary(int(metrics["complete_balanced_groups"]))
    if boundary != expected_boundary:
        raise ValueError("V5 target boundary does not recompute from geometry.")
    expected_gates = _coverage_gate_results(expected_metrics, expected_boundary)
    if proof.get("coverage_gates") != expected_gates:
        raise ValueError("V5 coverage gate result mismatch.")
    if proof.get("coverage_proof_passed") is not all(expected_gates.values()):
        raise ValueError("V5 coverage proof aggregate mismatch.")
    members = proof["universe"]["members"]
    if len(members) != UNIVERSE_SIZE or proof["universe"][
        "member_set_sha256"
    ] != stable_sha256(members):
        raise ValueError("V5 universe membership mismatch.")
    if [int(member["window_observations"]) for member in members] != mask.sum(
        axis=0
    ).astype(int).tolist():
        raise ValueError("V5 member observation counts do not match the mask.")
    if proof.get("proof_digest") != stable_sha256(
        {key: value for key, value in proof.items() if key != "proof_digest"}
    ):
        raise ValueError("V5 rolling geometry proof digest mismatch.")


def build_rolling_geometry_manifest(
    proofs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(proofs) != 1:
        raise ValueError(
            "The bounded v5 lifecycle permits exactly one real proof cell."
        )
    proof = copy.deepcopy(dict(proofs[0]))
    validate_rolling_geometry_proof(proof)
    metrics = proof["geometry_metrics"]
    manifest: dict[str, Any] = {
        "schema": ROLLING_MANIFEST_SCHEMA,
        "contract": rolling_geometry_contract(),
        "headline_calibration_claim": headline_calibration_claim(),
        "proof_cells": [
            {
                "cell_id": proof["cell_id"],
                "proof_digest": proof["proof_digest"],
                "endpoint_month": proof["spec"]["endpoint_month"],
                "coverage_proof_passed": proof["coverage_proof_passed"],
            }
        ],
        "within_window_geometry_distribution": {
            "group_size_histogram": metrics["group_size_histogram"],
            "asset_observation_counts": metrics["observations_per_asset_distribution"],
            "date_observed_asset_counts": metrics[
                "observed_assets_per_date_distribution"
            ],
            "pairwise_observation_counts": metrics[
                "pairwise_observations_distribution"
            ],
            "missing_fraction": metrics["missing_fraction"],
            "between_degrees_of_freedom": metrics["between_degrees_of_freedom"],
            "within_degrees_of_freedom": metrics["within_degrees_of_freedom"],
            "target_boundary_feasibility": proof["target_boundary_feasibility"],
        },
        "coverage_proof_passed": proof["coverage_proof_passed"],
        "full_72_endpoint_derivation_run": False,
        "detector_outcomes_present": False,
        "aws_execution_authorized": False,
        "holdout_2025_opened": False,
    }
    manifest["manifest_digest"] = stable_sha256(manifest)
    return manifest
