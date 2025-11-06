from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "DatasetRegistryError",
    "RegistryEntry",
    "assert_registered_dataset",
    "get_registry",
]


class DatasetRegistryError(RuntimeError):
    """Raised when a dataset fails integrity validation."""


@dataclass(frozen=True)
class RegistryEntry:
    """Entry describing a WRDS-derived dataset."""

    path: str
    sha256: str
    rows: int | None = None
    columns: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    source: str | None = None
    note: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registry_path() -> Path:
    return _repo_root() / "data" / "registry.json"


@lru_cache(maxsize=1)
def get_registry() -> dict[str, Any]:
    """Load the dataset registry from disk."""

    path = _registry_path()
    if not path.exists():
        raise DatasetRegistryError(f"Dataset registry missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise DatasetRegistryError(f"Failed to parse dataset registry: {exc}") from exc
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping):
        raise DatasetRegistryError("Registry is missing 'datasets' mapping.")
    return payload


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalise_key(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(_repo_root())
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()


@lru_cache(maxsize=None)
def _assert_registered_dataset(path_key: str) -> RegistryEntry:
    abs_path = Path(path_key)
    if not abs_path.exists():
        raise DatasetRegistryError(f"Dataset not found: {abs_path}")

    registry = get_registry()
    datasets = registry.get("datasets", {})
    if not isinstance(datasets, Mapping):
        raise DatasetRegistryError("Registry is missing 'datasets' mapping.")

    key = _normalise_key(abs_path)
    entry = datasets.get(key)
    repo_data_root = _repo_root() / "data"

    if entry is None:
        if abs_path.is_relative_to(repo_data_root):
            raise DatasetRegistryError(
                f"Dataset {key} is not registered. Refresh WRDS data and update "
                "data/registry.json."
            )
        # Allow non-repo paths without forcing registration.
        return RegistryEntry(path=key, sha256="", rows=None, columns=None)

    expected_hash = str(entry.get("sha256", "")).strip()
    if not expected_hash:
        raise DatasetRegistryError(f"Registry entry for {key} missing sha256.")

    actual_hash = _compute_sha256(abs_path)
    if actual_hash != expected_hash:
        raise DatasetRegistryError(
            f"Dataset {key} hash mismatch. Expected {expected_hash}, got {actual_hash}. "
            "Re-ingest from WRDS and update data/registry.json."
        )

    return RegistryEntry(
        path=key,
        sha256=expected_hash,
        rows=int(entry["rows"]) if "rows" in entry else None,
        columns=int(entry["columns"]) if "columns" in entry else None,
        start_date=str(entry["start_date"]) if entry.get("start_date") else None,
        end_date=str(entry["end_date"]) if entry.get("end_date") else None,
        source=str(entry["wrds_source"]) if entry.get("wrds_source") else None,
        note=str(entry["note"]) if entry.get("note") else None,
    )


def assert_registered_dataset(dataset_path: Path) -> RegistryEntry:
    """Validate dataset integrity against ``data/registry.json``."""

    return _assert_registered_dataset(dataset_path.resolve().as_posix())
