from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CELL_SCHEMA = "fjs-real-design-cell/v1"
SOURCE_CONTRACT_ID = "fjs-crsp-ciz-development-source-v1"
UNIVERSE_CONTRACT_ID = "fjs-lagged-cap-permno-universe-v1"
RESIDUALIZATION_CONTRACT_ID = "fjs-past-only-ff6-residualization-v1"
COVARIANCE_CONTRACT_ID = "fjs-pairwise-psd-residual-covariance-v1"

DEVELOPMENT_START = pd.Timestamp("2013-01-01")
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
FINAL_HOLDOUT_START = pd.Timestamp("2025-01-01")

FACTOR_COLUMNS = ("MKT", "SMB", "HML", "RMW", "CMA", "MOM")
FACTOR_FILE_COLUMNS = ("date", *FACTOR_COLUMNS, "RF")
SOURCE_COLUMNS = (
    "permno",
    "dlycaldt",
    "securitytype",
    "securitysubtype",
    "sharetype",
    "usincflg",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "dlyprc",
    "dlycap",
    "dlyret",
    "dlyretmissflg",
    "dlydelflg",
)


def stable_json_dumps(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def stable_sha256(payload: Mapping[str, Any] | Sequence[Any] | str | bytes) -> str:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_digest(payload: Mapping[str, Any]) -> str:
    return stable_sha256(
        {key: value for key, value in payload.items() if key != "sha256"}
    )


@dataclass(frozen=True)
class SourcePartitionBinding:
    path: str
    partition: str
    sha256: str
    size_bytes: int
    receipt_manifest_path: str
    receipt_manifest_sha256: str
    receipt_status: str
    receipt_rows: int
    receipt_size_bytes: int
    receipt_schema: str
    receipt_table: str
    receipt_date_column: str
    binding_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorBinding:
    path: str
    sha256: str
    size_bytes: int
    registry_path: str
    registry_sha256: str
    registry_key: str
    source: str
    start_date: str
    end_date: str
    columns: tuple[str, ...]
    binding_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        return payload


@dataclass(frozen=True)
class RealDesignCellSpec:
    cell_id: str
    factor_fit_start: str
    factor_fit_end: str
    formation_date: str
    window_start: str
    window_end: str
    universe_size: int = 60
    min_factor_observations: int = 252
    min_window_observations: int = 100
    min_pairwise_observations: int = 60
    max_cap_staleness_days: int = 10
    proof_only: bool = False

    def validate(self) -> None:
        dates = {
            "factor_fit_start": pd.Timestamp(self.factor_fit_start),
            "factor_fit_end": pd.Timestamp(self.factor_fit_end),
            "formation_date": pd.Timestamp(self.formation_date),
            "window_start": pd.Timestamp(self.window_start),
            "window_end": pd.Timestamp(self.window_end),
        }
        if dates["factor_fit_start"] < DEVELOPMENT_START:
            raise ValueError(
                "Factor fitting may not precede the frozen development start."
            )
        if dates["window_end"] > DEVELOPMENT_END:
            raise ValueError(
                "Real-design cells must remain inside 2013-2018 development."
            )
        if max(dates.values()) >= FINAL_HOLDOUT_START:
            raise ValueError("The 2025 final holdout must remain unopened.")
        if not (
            dates["factor_fit_start"]
            <= dates["factor_fit_end"]
            <= dates["formation_date"]
            < dates["window_start"]
            <= dates["window_end"]
        ):
            raise ValueError(
                "Cell dates must satisfy fit_start <= fit_end <= formation "
                "< window_start <= window_end."
            )
        if self.universe_size < 2:
            raise ValueError("universe_size must be at least two.")
        if self.min_factor_observations < len(FACTOR_COLUMNS) + 1:
            raise ValueError("Factor fitting requires at least seven observations.")
        if self.min_window_observations < 2:
            raise ValueError("min_window_observations must be at least two.")
        if self.min_pairwise_observations < 2:
            raise ValueError("min_pairwise_observations must be at least two.")
        if self.max_cap_staleness_days < 0:
            raise ValueError("max_cap_staleness_days must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def source_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": SOURCE_CONTRACT_ID,
        "identity": "PERMNO",
        "development_start": DEVELOPMENT_START.date().isoformat(),
        "development_end": DEVELOPMENT_END.date().isoformat(),
        "final_holdout_start": FINAL_HOLDOUT_START.date().isoformat(),
        "source_table": "crsp.wrds_dsfv2_query",
        "required_columns": list(SOURCE_COLUMNS),
        "filters": {
            "securitytype": ["EQTY"],
            "securitysubtype": ["COM"],
            "sharetype": ["NS"],
            "usincflg": ["Y"],
            "primaryexch": ["A", "N", "Q"],
            "conditionaltype": ["RW"],
            "tradingstatusflg": ["A"],
            "absolute_price_min": 5.0,
            "market_cap_positive": True,
            "return_numeric_and_present": True,
        },
        "duplicate_policy": (
            "collapse_only_exact_required_field_duplicates_then_fail_on_conflict"
        ),
        "legacy_ticker_csv_is_not_a_source": True,
        "raw_inputs_must_remain_outside_git": True,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def universe_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": UNIVERSE_CONTRACT_ID,
        "identity": "PERMNO",
        "ranking_field": "dlycap",
        "ranking_direction": "descending",
        "ranking_tie_break": "permno_ascending",
        "ranking_observation": "last_eligible_on_or_before_formation_date",
        "membership_fixed_before_window_start": True,
        "future_backfill_forbidden": True,
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def residualization_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": RESIDUALIZATION_CONTRACT_ID,
        "target": "dlyret_minus_RF",
        "intercept": True,
        "factors": list(FACTOR_COLUMNS),
        "fit_method": "numpy_lstsq_rcond_none",
        "fit_observations": "strictly_before_window_start",
        "factor_date_join": "exact_trading_date_inner_join",
        "coefficient_order": ["intercept", *FACTOR_COLUMNS],
        "missing_return_policy": "preserve_as_missing",
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def covariance_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract_id": COVARIANCE_CONTRACT_ID,
        "centering": "per_asset_observed_residual_mean",
        "estimator": "pairwise_overlap_sample_covariance",
        "pairwise_denominator": "n_overlap_minus_one",
        "psd_projection": "symmetric_eigendecomposition_floor",
        "eigenvalue_floor": "max(max_eigenvalue*1e-10,1e-12)",
        "matrix_serialization": "little_endian_float64_row_major_base64",
        "missingness_serialization": "row_major_packbits_little_hex",
    }
    payload["sha256"] = stable_sha256(payload)
    return payload


def bind_source_partition(
    source_path: Path,
    receipt_manifest_path: Path,
) -> SourcePartitionBinding:
    source = source_path.expanduser().resolve()
    receipt = receipt_manifest_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CRSP source partition is missing: {source}")
    if not receipt.is_file():
        raise FileNotFoundError(f"CRSP receipt manifest is missing: {receipt}")
    manifest = json.loads(receipt.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise ValueError(f"Receipt manifest is not status=ok: {receipt}")
    matching = []
    for item in manifest.get("items", []):
        if not isinstance(item, Mapping) or not item.get("path"):
            continue
        if Path(str(item["path"])).expanduser().resolve() == source:
            matching.append(item)
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one status receipt for {source}; found {len(matching)}."
        )
    item = matching[0]
    if item.get("status") != "ok":
        raise ValueError(f"CRSP source receipt is not status=ok: {source}")
    actual_size = source.stat().st_size
    receipt_size = int(item.get("size_bytes", -1))
    if receipt_size != actual_size:
        raise ValueError(
            f"CRSP source size mismatch for {source}: receipt={receipt_size}, "
            f"actual={actual_size}."
        )
    base = {
        "path": str(source),
        "partition": str(item.get("partition", "")),
        "sha256": file_sha256(source),
        "size_bytes": actual_size,
        "receipt_manifest_path": str(receipt),
        "receipt_manifest_sha256": file_sha256(receipt),
        "receipt_status": str(item.get("status")),
        "receipt_rows": int(item.get("rows", -1)),
        "receipt_size_bytes": receipt_size,
        "receipt_schema": str(item.get("schema", "")),
        "receipt_table": str(item.get("table", "")),
        "receipt_date_column": str(item.get("date_column", "")),
    }
    if base["receipt_rows"] <= 0:
        raise ValueError(f"CRSP source receipt has no positive row count: {source}")
    if base["receipt_schema"] != "crsp":
        raise ValueError(f"Unexpected CRSP receipt schema: {base['receipt_schema']!r}")
    if base["receipt_table"] != "wrds_dsfv2_query":
        raise ValueError(f"Unexpected CRSP receipt table: {base['receipt_table']!r}")
    if base["receipt_date_column"] != "dlycaldt":
        raise ValueError(
            f"Unexpected CRSP receipt date column: {base['receipt_date_column']!r}"
        )
    base["binding_sha256"] = stable_sha256(base)
    return SourcePartitionBinding(**base)


def bind_factor_source(factor_path: Path, registry_path: Path) -> FactorBinding:
    factors = factor_path.expanduser().resolve()
    registry = registry_path.expanduser().resolve()
    if not factors.is_file():
        raise FileNotFoundError(f"FF6 factor file is missing: {factors}")
    if not registry.is_file():
        raise FileNotFoundError(f"Factor registry is missing: {registry}")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("Factor registry is missing its datasets mapping.")
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for key, raw in datasets.items():
        if not isinstance(raw, Mapping) or not raw.get("path"):
            continue
        candidate = Path(str(raw["path"]))
        if not candidate.is_absolute():
            candidate = (registry.parents[2] / candidate).resolve()
        if candidate == factors:
            matches.append((str(key), raw))
    if not matches:
        raise ValueError(f"Factor file is not registered: {factors}")
    preferred = next(
        (match for match in matches if match[0] == "data/factors/ff5mom_daily.csv"),
        matches[0],
    )
    key, entry = preferred
    expected_hash = str(entry.get("sha256", ""))
    actual_hash = file_sha256(factors)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            f"Factor hash mismatch: registry={expected_hash!r}, actual={actual_hash}."
        )
    columns = tuple(str(value) for value in entry.get("columns", []))
    if tuple(FACTOR_COLUMNS) != columns[: len(FACTOR_COLUMNS)] or "RF" not in columns:
        raise ValueError("Registered factor columns do not contain exact FF6 plus RF.")
    base = {
        "path": str(factors),
        "sha256": actual_hash,
        "size_bytes": factors.stat().st_size,
        "registry_path": str(registry),
        "registry_sha256": file_sha256(registry),
        "registry_key": key,
        "source": str(entry.get("source", "")),
        "start_date": str(entry.get("start_date", "")),
        "end_date": str(entry.get("end_date", "")),
        "columns": columns,
    }
    digest_payload = {**base, "columns": list(columns)}
    base["binding_sha256"] = stable_sha256(digest_payload)
    return FactorBinding(**base)


def _normalize_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"CRSP partition is missing columns: {sorted(missing)}")
    result = frame.loc[:, list(SOURCE_COLUMNS)].copy()
    result["dlycaldt"] = pd.to_datetime(result["dlycaldt"], errors="coerce")
    for column in (
        "securitytype",
        "securitysubtype",
        "sharetype",
        "usincflg",
        "primaryexch",
        "conditionaltype",
        "tradingstatusflg",
        "dlyretmissflg",
        "dlydelflg",
    ):
        result[column] = result[column].astype("string").str.strip().str.upper()
    for column in ("permno", "dlyprc", "dlycap", "dlyret"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    mask = (
        result["dlycaldt"].notna()
        & result["permno"].notna()
        & result["securitytype"].eq("EQTY")
        & result["securitysubtype"].eq("COM")
        & result["sharetype"].eq("NS")
        & result["usincflg"].eq("Y")
        & result["primaryexch"].isin(("A", "N", "Q"))
        & result["conditionaltype"].eq("RW")
        & result["tradingstatusflg"].eq("A")
        & result["dlyprc"].abs().ge(5.0)
        & result["dlycap"].gt(0.0)
        & result["dlyret"].notna()
    )
    result = result.loc[mask].copy()
    result["permno"] = result["permno"].astype(np.int64)
    return result.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)


def load_filtered_sources(
    bindings: Sequence[SourcePartitionBinding],
    *,
    start: str,
    end: str,
    chunksize: int = 25_000,
    max_source_rows_per_partition: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts < DEVELOPMENT_START or end_ts > DEVELOPMENT_END:
        raise ValueError("Source loading is restricted to 2013-2018 development.")
    if start_ts > end_ts:
        raise ValueError("Source start date must not follow end date.")
    if not bindings:
        raise ValueError("At least one CRSP source partition binding is required.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    if max_source_rows_per_partition is not None and max_source_rows_per_partition <= 0:
        raise ValueError("max_source_rows_per_partition must be positive.")

    collected: list[pd.DataFrame] = []
    scans: list[dict[str, Any]] = []
    for binding in bindings:
        source = Path(binding.path)
        if file_sha256(source) != binding.sha256:
            raise ValueError(f"CRSP source changed after binding: {source}")
        scanned = 0
        filtered = 0
        reader = pd.read_csv(
            source,
            usecols=list(SOURCE_COLUMNS),
            chunksize=chunksize,
            low_memory=False,
        )
        for raw_chunk in reader:
            if max_source_rows_per_partition is not None:
                remaining = max_source_rows_per_partition - scanned
                if remaining <= 0:
                    break
                raw_chunk = raw_chunk.iloc[:remaining].copy()
            scanned += len(raw_chunk)
            clean = _normalize_source_frame(raw_chunk)
            clean = clean.loc[clean["dlycaldt"].between(start_ts, end_ts)]
            filtered += len(clean)
            if not clean.empty:
                collected.append(clean)
            if (
                max_source_rows_per_partition is not None
                and scanned >= max_source_rows_per_partition
            ):
                break
        scans.append(
            {
                "binding_sha256": binding.binding_sha256,
                "rows_scanned": scanned,
                "rows_receipted": binding.receipt_rows,
                "rows_after_frozen_filters_and_date_bounds": filtered,
                "scan_truncated": scanned < binding.receipt_rows,
            }
        )
    if not collected:
        raise ValueError("No CRSP observations survive the frozen filters and dates.")
    combined = pd.concat(collected, ignore_index=True)
    combined = combined.sort_values(["dlycaldt", "permno"]).reset_index(drop=True)
    duplicated = combined.duplicated(subset=["dlycaldt", "permno"], keep=False)
    exact_duplicate_rows_collapsed = 0
    if bool(duplicated.any()):
        conflicts = []
        duplicate_groups = combined.loc[duplicated].groupby(
            ["dlycaldt", "permno"], sort=True, dropna=False
        )
        comparison_columns = [
            column for column in SOURCE_COLUMNS if column not in {"dlycaldt", "permno"}
        ]
        for (date_value, permno), group in duplicate_groups:
            varying = [
                column
                for column in comparison_columns
                if int(group[column].nunique(dropna=False)) > 1
            ]
            if varying:
                conflicts.append(
                    {
                        "dlycaldt": pd.Timestamp(date_value).date().isoformat(),
                        "permno": int(permno),
                        "varying_required_fields": varying,
                    }
                )
        if conflicts:
            raise ValueError(
                "Conflicting duplicate CRSP date/PERMNO rows violate the v4 "
                f"identity contract: {conflicts[:5]}"
            )
        before = len(combined)
        combined = combined.drop_duplicates(
            subset=list(SOURCE_COLUMNS), keep="first"
        ).reset_index(drop=True)
        exact_duplicate_rows_collapsed = before - len(combined)
        if bool(combined.duplicated(subset=["dlycaldt", "permno"], keep=False).any()):
            raise ValueError(
                "Exact duplicate collapse did not restore PERMNO identity."
            )
    scan_payload = {
        "requested_start": start_ts.date().isoformat(),
        "requested_end": end_ts.date().isoformat(),
        "chunksize": chunksize,
        "max_source_rows_per_partition": max_source_rows_per_partition,
        "partitions": scans,
        "rows_after_all_filters": len(combined),
        "exact_duplicate_rows_collapsed": exact_duplicate_rows_collapsed,
        "source_contract_sha256": source_contract()["sha256"],
    }
    scan_payload["sha256"] = stable_sha256(scan_payload)
    return combined, scan_payload


def load_bound_factors(
    binding: FactorBinding,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    factor_path = Path(binding.path)
    if file_sha256(factor_path) != binding.sha256:
        raise ValueError("FF6 factor file changed after binding.")
    frame = pd.read_csv(factor_path, usecols=list(FACTOR_FILE_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in FACTOR_FILE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(FACTOR_FILE_COLUMNS))
    frame = frame.sort_values("date")
    if bool(frame["date"].duplicated().any()):
        raise ValueError("FF6 factor dates must be unique.")
    subset = frame.loc[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if subset.empty:
        raise ValueError("No FF6 factors cover the requested cell interval.")
    return subset.set_index("date")


def _serialize_float_matrix(matrix: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(matrix, dtype="<f8")
    return {
        "shape": list(values.shape),
        "dtype": "float64-le",
        "order": "C",
        "encoding": "base64",
        "data": base64.b64encode(values.tobytes(order="C")).decode("ascii"),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def _serialize_int_matrix(matrix: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(matrix, dtype="<i4")
    return {
        "shape": list(values.shape),
        "dtype": "int32-le",
        "order": "C",
        "encoding": "base64",
        "data": base64.b64encode(values.tobytes(order="C")).decode("ascii"),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def _serialize_missingness(mask: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(mask, dtype=np.uint8)
    packed = np.packbits(values.reshape(-1), bitorder="little").tobytes()
    return {
        "shape": list(values.shape),
        "dtype": "bool",
        "order": "C",
        "encoding": "packbits-little-hex",
        "data": packed.hex(),
        "sha256": hashlib.sha256(packed).hexdigest(),
    }


def _pairwise_covariance(
    residuals: np.ndarray,
    *,
    min_pairwise_observations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if residuals.ndim != 2:
        raise ValueError("Residual matrix must be two-dimensional.")
    p_assets = residuals.shape[1]
    centered = residuals - np.nanmean(residuals, axis=0, keepdims=True)
    covariance = np.empty((p_assets, p_assets), dtype=np.float64)
    counts = np.empty((p_assets, p_assets), dtype=np.int32)
    for left in range(p_assets):
        for right in range(left, p_assets):
            valid = np.isfinite(centered[:, left]) & np.isfinite(centered[:, right])
            count = int(valid.sum())
            if count < min_pairwise_observations:
                raise ValueError(
                    "Insufficient pairwise residual overlap for assets "
                    f"{left}/{right}: {count} < {min_pairwise_observations}."
                )
            value = float(
                np.dot(centered[valid, left], centered[valid, right]) / (count - 1)
            )
            covariance[left, right] = value
            covariance[right, left] = value
            counts[left, right] = count
            counts[right, left] = count
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    maximum = float(np.max(eigenvalues))
    floor = max(maximum * 1e-10, 1e-12)
    projected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    projected = 0.5 * (projected + projected.T)
    diagnostics = {
        "raw_min_eigenvalue": float(np.min(eigenvalues)),
        "raw_max_eigenvalue": maximum,
        "projection_floor": floor,
        "projected_min_eigenvalue": float(np.min(np.linalg.eigvalsh(projected))),
    }
    return projected, counts, diagnostics


def derive_real_design_cell(
    source_frame: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    spec: RealDesignCellSpec,
    source_bindings: Sequence[SourcePartitionBinding],
    factor_binding: FactorBinding,
    scan_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    spec.validate()
    fit_start = pd.Timestamp(spec.factor_fit_start)
    fit_end = pd.Timestamp(spec.factor_fit_end)
    formation = pd.Timestamp(spec.formation_date)
    window_start = pd.Timestamp(spec.window_start)
    window_end = pd.Timestamp(spec.window_end)
    if bool(scan_receipt.get("partitions")) is False:
        raise ValueError("A source scan receipt is required.")
    if not source_bindings:
        raise ValueError("At least one source binding is required.")

    factor_dates = factors.loc[factors.index.to_series().between(fit_start, window_end)]
    if factor_dates.empty:
        raise ValueError("FF6 factors do not cover the cell interval.")
    fit_date_index = factor_dates.loc[
        factor_dates.index.to_series().between(fit_start, fit_end)
    ].index
    date_index = factor_dates.loc[
        factor_dates.index.to_series().between(window_start, window_end)
    ].index
    if len(date_index) < spec.min_window_observations:
        raise ValueError(
            f"Only {len(date_index)} factor dates exist in the window; "
            f"{spec.min_window_observations} required."
        )

    fit_observation_counts = (
        source_frame.loc[source_frame["dlycaldt"].isin(fit_date_index)]
        .groupby("permno")["dlycaldt"]
        .nunique()
    )
    window_observation_counts = (
        source_frame.loc[source_frame["dlycaldt"].isin(date_index)]
        .groupby("permno")["dlycaldt"]
        .nunique()
    )
    eligible_permnos = fit_observation_counts.loc[
        fit_observation_counts.ge(spec.min_factor_observations)
    ].index.intersection(
        window_observation_counts.loc[
            window_observation_counts.ge(spec.min_window_observations)
        ].index,
        sort=False,
    )

    ranking = source_frame.loc[source_frame["dlycaldt"].le(formation)].copy()
    ranking = ranking.loc[ranking["permno"].isin(eligible_permnos)].copy()
    ranking = ranking.sort_values(["permno", "dlycaldt"])
    ranking = ranking.groupby("permno", as_index=False).tail(1)
    age_days = (formation - ranking["dlycaldt"]).dt.days
    ranking = ranking.loc[age_days.le(spec.max_cap_staleness_days)].copy()
    ranking = ranking.sort_values(
        ["dlycap", "permno"], ascending=[False, True], kind="mergesort"
    )
    if len(ranking) < spec.universe_size:
        raise ValueError(
            f"Only {len(ranking)} eligible PERMNOs exist at formation; "
            f"{spec.universe_size} required."
        )
    selected = ranking.head(spec.universe_size).copy()
    selected_permnos = selected["permno"].astype(int).tolist()
    member_rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        member_rows.append(
            {
                "rank": rank,
                "permno": int(row.permno),
                "lagged_market_cap": float(row.dlycap),
                "cap_observation_date": pd.Timestamp(row.dlycaldt).date().isoformat(),
            }
        )

    relevant = source_frame.loc[
        source_frame["permno"].isin(selected_permnos)
        & source_frame["dlycaldt"].between(fit_start, window_end)
    ].copy()
    residuals = np.full((len(date_index), len(selected_permnos)), np.nan)
    coefficients = np.full((len(selected_permnos), len(FACTOR_COLUMNS) + 1), np.nan)
    fit_counts: list[int] = []
    date_positions = {
        pd.Timestamp(value): index for index, value in enumerate(date_index)
    }
    for column_index, permno in enumerate(selected_permnos):
        returns = relevant.loc[relevant["permno"].eq(permno), ["dlycaldt", "dlyret"]]
        joined = returns.merge(
            factor_dates.reset_index(),
            left_on="dlycaldt",
            right_on="date",
            how="inner",
            validate="one_to_one",
        )
        training = joined.loc[joined["date"].between(fit_start, fit_end)].copy()
        if len(training) < spec.min_factor_observations:
            raise ValueError(
                f"PERMNO {permno} has {len(training)} factor-fit observations; "
                f"{spec.min_factor_observations} required."
            )
        design = np.column_stack(
            [
                np.ones(len(training), dtype=np.float64),
                training.loc[:, FACTOR_COLUMNS].to_numpy(dtype=np.float64),
            ]
        )
        target = training["dlyret"].to_numpy(dtype=np.float64) - training[
            "RF"
        ].to_numpy(dtype=np.float64)
        if int(np.linalg.matrix_rank(design)) < design.shape[1]:
            raise ValueError(f"PERMNO {permno} factor design is rank deficient.")
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        coefficients[column_index, :] = beta
        fit_counts.append(len(training))
        window = joined.loc[joined["date"].between(window_start, window_end)].copy()
        window_design = np.column_stack(
            [
                np.ones(len(window), dtype=np.float64),
                window.loc[:, FACTOR_COLUMNS].to_numpy(dtype=np.float64),
            ]
        )
        window_target = window["dlyret"].to_numpy(dtype=np.float64) - window[
            "RF"
        ].to_numpy(dtype=np.float64)
        observed_residuals = window_target - window_design @ beta
        for observed_date, residual in zip(
            window["date"], observed_residuals, strict=True
        ):
            residuals[date_positions[pd.Timestamp(observed_date)], column_index] = (
                residual
            )

    observed_per_asset = np.isfinite(residuals).sum(axis=0)
    if bool((observed_per_asset < spec.min_window_observations).any()):
        failing = [
            {
                "permno": selected_permnos[index],
                "observations": int(count),
            }
            for index, count in enumerate(observed_per_asset)
            if count < spec.min_window_observations
        ]
        raise ValueError(f"Selected assets fail window coverage: {failing}")

    missing_mask = np.isfinite(residuals)
    covariance, pairwise_counts, covariance_diagnostics = _pairwise_covariance(
        residuals,
        min_pairwise_observations=spec.min_pairwise_observations,
    )
    week_labels = [
        (pd.Timestamp(value) - pd.Timedelta(days=pd.Timestamp(value).weekday()))
        .date()
        .isoformat()
        for value in date_index
    ]
    weekday_slots = [int(pd.Timestamp(value).weekday()) for value in date_index]
    week_counts = pd.Series(week_labels, dtype="string").value_counts(sort=False)
    group_sizes = [int(week_counts[label]) for label in sorted(week_counts.index)]
    p_assets = len(selected_permnos)
    n_groups = len(group_sizes)
    between_dof = max(n_groups - 1, 0)
    within_dof = sum(max(value - 1, 0) for value in group_sizes)
    complete_group_count = 0
    for label in sorted(set(week_labels)):
        rows = [index for index, value in enumerate(week_labels) if value == label]
        if len(rows) == 5 and bool(missing_mask[rows, :].all()):
            complete_group_count += 1

    universe_payload = {
        "contract": universe_contract(),
        "formation_date": formation.date().isoformat(),
        "members": member_rows,
    }
    universe_payload["member_set_sha256"] = stable_sha256(member_rows)
    cell: dict[str, Any] = {
        "schema": CELL_SCHEMA,
        "cell_id": spec.cell_id,
        "purpose": (
            "Development-only realistic FJS detector calibration input; "
            "not an empirical result or promotion artifact."
        ),
        "claim_boundary": {
            "development_only": True,
            "mechanism_calibration_only": True,
            "empirical_claims_forbidden": True,
            "promotion_allowed": False,
            "proof_only": spec.proof_only,
            "legacy_ticker_csv_used": False,
            "holdout_2025_opened": False,
        },
        "spec": spec.to_dict(),
        "source_contract": source_contract(),
        "source_partitions": [binding.to_dict() for binding in source_bindings],
        "source_scan_receipt": dict(scan_receipt),
        "factor_source": factor_binding.to_dict(),
        "residualization_contract": residualization_contract(),
        "covariance_contract": covariance_contract(),
        "universe": universe_payload,
        "factor_fit": {
            "start": fit_start.date().isoformat(),
            "end": fit_end.date().isoformat(),
            "window_start": window_start.date().isoformat(),
            "observations_per_asset": fit_counts,
            "coefficient_order": ["intercept", *FACTOR_COLUMNS],
            "coefficients": _serialize_float_matrix(coefficients),
        },
        "window_geometry": {
            "dates": [pd.Timestamp(value).date().isoformat() for value in date_index],
            "week_labels": week_labels,
            "weekday_slots": weekday_slots,
            "group_sizes": group_sizes,
            "p_assets": p_assets,
            "n_dates": len(date_index),
            "n_groups": n_groups,
            "replicate_slots": 5,
            "complete_balanced_groups": complete_group_count,
            "between_degrees_of_freedom": between_dof,
            "within_degrees_of_freedom": within_dof,
            "between_aspect_ratio": (
                p_assets / float(between_dof) if between_dof else None
            ),
            "within_aspect_ratio": p_assets / float(within_dof) if within_dof else None,
        },
        "missingness": {
            "observed_mask": _serialize_missingness(missing_mask),
            "observed_per_asset": observed_per_asset.astype(int).tolist(),
            "observed_per_date": missing_mask.sum(axis=1).astype(int).tolist(),
            "missing_fraction": float(1.0 - missing_mask.mean()),
        },
        "residual_covariance": {
            "matrix": _serialize_float_matrix(covariance),
            "pairwise_observation_counts": _serialize_int_matrix(pairwise_counts),
            "diagnostics": covariance_diagnostics,
        },
    }
    cell["cell_digest"] = stable_sha256(cell)
    return cell


def _decode_matrix(payload: Mapping[str, Any], dtype: str) -> np.ndarray:
    if payload.get("encoding") != "base64":
        raise ValueError("Matrix encoding must be base64.")
    raw = base64.b64decode(str(payload["data"]), validate=True)
    if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
        raise ValueError("Serialized matrix hash mismatch.")
    shape = tuple(int(value) for value in payload["shape"])
    values = np.frombuffer(raw, dtype=dtype)
    if values.size != int(np.prod(shape)):
        raise ValueError("Serialized matrix shape does not match its bytes.")
    return values.reshape(shape)


def validate_real_design_cell(cell: Mapping[str, Any]) -> None:
    if cell.get("schema") != CELL_SCHEMA:
        raise ValueError("Unknown real-design cell schema.")
    observed_digest = cell.get("cell_digest")
    expected_digest = stable_sha256(
        {key: value for key, value in cell.items() if key != "cell_digest"}
    )
    if observed_digest != expected_digest:
        raise ValueError("Real-design cell digest mismatch.")
    if cell.get("source_contract") != source_contract():
        raise ValueError("Real-design source contract mismatch.")
    if cell.get("residualization_contract") != residualization_contract():
        raise ValueError("Real-design residualization contract mismatch.")
    if cell.get("covariance_contract") != covariance_contract():
        raise ValueError("Real-design covariance contract mismatch.")
    claim_boundary = cell.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        raise ValueError("Real-design claim boundary is missing.")
    if claim_boundary.get("legacy_ticker_csv_used") is not False:
        raise ValueError("The legacy ticker CSV cannot source v4 real-design cells.")
    if claim_boundary.get("holdout_2025_opened") is not False:
        raise ValueError("The 2025 holdout must remain unopened.")
    spec = RealDesignCellSpec(**dict(cell["spec"]))
    spec.validate()

    members = cell["universe"]["members"]
    if len(members) != spec.universe_size:
        raise ValueError("Real-design universe size mismatch.")
    if [int(item["rank"]) for item in members] != list(
        range(1, spec.universe_size + 1)
    ):
        raise ValueError("Real-design universe ranks are not contiguous.")
    permnos = [int(item["permno"]) for item in members]
    if len(set(permnos)) != len(permnos):
        raise ValueError("Real-design universe PERMNO identities are not unique.")
    if cell["universe"].get("member_set_sha256") != stable_sha256(members):
        raise ValueError("Real-design universe member digest mismatch.")

    covariance = _decode_matrix(cell["residual_covariance"]["matrix"], "<f8")
    counts = _decode_matrix(
        cell["residual_covariance"]["pairwise_observation_counts"], "<i4"
    )
    expected_shape = (spec.universe_size, spec.universe_size)
    if covariance.shape != expected_shape or counts.shape != expected_shape:
        raise ValueError("Real-design covariance dimensions do not match universe.")
    if not np.isfinite(covariance).all():
        raise ValueError("Real-design covariance contains non-finite values.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("Real-design covariance is not symmetric.")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-12:
        raise ValueError("Real-design covariance is not positive semidefinite.")
    if int(np.min(counts)) < spec.min_pairwise_observations:
        raise ValueError("Real-design pairwise counts violate the frozen minimum.")

    mask_payload = cell["missingness"]["observed_mask"]
    packed = bytes.fromhex(str(mask_payload["data"]))
    if hashlib.sha256(packed).hexdigest() != mask_payload.get("sha256"):
        raise ValueError("Real-design missingness hash mismatch.")
    shape = tuple(int(value) for value in mask_payload["shape"])
    if shape != (cell["window_geometry"]["n_dates"], spec.universe_size):
        raise ValueError("Real-design missingness dimensions do not match geometry.")
    required_bits = int(np.prod(shape))
    unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[
        :required_bits
    ]
    if unpacked.size != required_bits:
        raise ValueError("Real-design missingness bytes are truncated.")


def write_real_design_cell(cell: Mapping[str, Any], path: Path) -> dict[str, Any]:
    validate_real_design_cell(cell)
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stable_json_dumps(cell) + "\n", encoding="utf-8")
    return {
        "cell_id": str(cell["cell_id"]),
        "path": str(out),
        "sha256": file_sha256(out),
        "size_bytes": out.stat().st_size,
        "cell_digest": str(cell["cell_digest"]),
        "proof_only": bool(cell["claim_boundary"]["proof_only"]),
        "member_set_sha256": str(cell["universe"]["member_set_sha256"]),
        "source_binding_sha256": [
            str(item["binding_sha256"]) for item in cell["source_partitions"]
        ],
        "factor_binding_sha256": str(cell["factor_source"]["binding_sha256"]),
    }
