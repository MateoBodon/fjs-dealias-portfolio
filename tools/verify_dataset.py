#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise_key(path: Path) -> str:
    try:
        rel = path.relative_to(Path.cwd())
        return rel.as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"registry file '{path}' does not exist")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("registry missing 'datasets' mapping")
    return datasets


def verify_dataset(dataset_path: Path, registry_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset '{dataset_path}' is missing")
    registry = _load_registry(registry_path)
    key_options = {
        dataset_path.as_posix(),
        str(dataset_path),
        _normalise_key(dataset_path),
    }
    record = None
    for key in key_options:
        if key in registry:
            record = registry[key]
            break
    if record is None:
        raise KeyError(f"dataset '{dataset_path}' not found in registry '{registry_path}'")
    expected_hash = record.get("sha256")
    if not expected_hash:
        raise ValueError(f"registry entry for '{dataset_path}' missing 'sha256'")
    actual_hash = _sha256(dataset_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"sha256 mismatch for '{dataset_path}': expected {expected_hash}, got {actual_hash}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset hashes against data/registry.json")
    parser.add_argument("dataset", type=Path, help="Path to the dataset to verify.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/registry.json"),
        help="Registry JSON containing expected hashes (default: data/registry.json).",
    )
    args = parser.parse_args()
    verify_dataset(args.dataset, args.registry)


if __name__ == "__main__":
    main()
