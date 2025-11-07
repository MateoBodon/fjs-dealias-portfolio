from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


__all__ = [
    "FactorRegistryError",
    "FactorRegistryEntry",
    "assert_registered_factor",
    "get_factor_registry",
    "load_registered_factors",
]


class FactorRegistryError(RuntimeError):
    """Raised when observed-factor datasets fail validation."""


@dataclass(frozen=True)
class FactorRegistryEntry:
    """Metadata describing an observed-factor dataset."""

    key: str
    path: Path
    sha256: str
    columns: tuple[str, ...]
    start_date: str | None = None
    end_date: str | None = None
    frequency: str | None = None
    source: str | None = None
    note: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registry_path() -> Path:
    override = os.environ.get("FACTOR_REGISTRY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return _repo_root() / "data" / "factors" / "registry.json"


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=1)
def get_factor_registry() -> Mapping[str, Any]:
    """Load the observed-factor registry payload."""

    path = _registry_path()
    if not path.exists():
        raise FactorRegistryError(f"Factor registry missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise FactorRegistryError(f"Failed to parse factor registry: {exc}") from exc
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping):
        raise FactorRegistryError("Factor registry missing 'datasets' mapping.")
    return payload


def _resolve_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = (_repo_root() / candidate).resolve()
    return candidate


@lru_cache(maxsize=None)
def assert_registered_factor(key: str) -> FactorRegistryEntry:
    """Validate dataset integrity against ``data/factors/registry.json``."""

    registry = get_factor_registry()
    datasets = registry.get("datasets", {})
    if not isinstance(datasets, Mapping):  # pragma: no cover - already checked
        raise FactorRegistryError("Factor registry missing 'datasets' mapping.")
    entry = datasets.get(key)
    if not isinstance(entry, Mapping):
        raise FactorRegistryError(f"Factor dataset '{key}' not registered.")
    path_value = entry.get("path")
    if not path_value:
        raise FactorRegistryError(f"Factor dataset '{key}' missing 'path'.")
    dataset_path = _resolve_path(str(path_value))
    if not dataset_path.exists():
        raise FactorRegistryError(
            f"Factor dataset '{key}' expected at {dataset_path}, but file is missing. "
            "Run the ingestion script or sync WRDS exports before enabling factor prewhitening."
        )
    expected_hash = str(entry.get("sha256", "")).strip()
    if not expected_hash:
        raise FactorRegistryError(
            f"Factor dataset '{key}' missing sha256 in registry. Update data/factors/registry.json."
        )
    actual_hash = _compute_sha256(dataset_path)
    if actual_hash != expected_hash:
        raise FactorRegistryError(
            f"Factor dataset '{key}' hash mismatch. Expected {expected_hash}, observed {actual_hash}."
        )
    columns = tuple(str(col).strip() for col in entry.get("columns", []) if str(col).strip())
    return FactorRegistryEntry(
        key=key,
        path=dataset_path,
        sha256=expected_hash,
        columns=columns,
        start_date=str(entry.get("start_date")) if entry.get("start_date") else None,
        end_date=str(entry.get("end_date")) if entry.get("end_date") else None,
        frequency=str(entry.get("frequency")) if entry.get("frequency") else None,
        source=str(entry.get("source")) if entry.get("source") else None,
        note=str(entry.get("note")) if entry.get("note") else None,
    )


def _to_timestamp(value: object | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    if isinstance(ts, pd.Series):
        return pd.to_datetime(ts.iloc[0])
    return pd.to_datetime(ts)


def load_registered_factors(
    key: str,
    *,
    start: object | None = None,
    end: object | None = None,
    required: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, FactorRegistryEntry]:
    """Load and validate a registered observed-factor dataset."""

    entry = assert_registered_factor(key)
    required_cols = tuple(required) if required else entry.columns
    from baselines import load_observed_factors  # Local import to avoid circular dependency
    factors = load_observed_factors(path=entry.path, required=required_cols or None)
    factors = factors.sort_index().tz_localize(None)
    start_ts = _to_timestamp(start)
    end_ts = _to_timestamp(end)
    if start_ts is not None and start_ts < factors.index.min():
        raise FactorRegistryError(
            f"Factor dataset '{key}' begins at {factors.index.min().date()}, earlier than requested start {start_ts.date()}."
        )
    if end_ts is not None and end_ts > factors.index.max():
        raise FactorRegistryError(
            f"Factor dataset '{key}' ends at {factors.index.max().date()}, earlier than requested end {end_ts.date()}."
        )
    return factors, entry
