#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_file(path: Path, expected_sha: str) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f"missing file: {path}")
        return errors
    actual = sha256_file(path)
    if actual != expected_sha:
        errors.append(f"sha256 mismatch for {path} (expected {expected_sha}, got {actual})")
    return errors


def check_data_registry(registry_path: Path) -> list[str]:
    errors: list[str] = []
    if not registry_path.exists():
        return [f"missing registry: {registry_path}"]
    with registry_path.open() as handle:
        registry = json.load(handle)
    datasets = registry.get("datasets", {})
    for rel_path, meta in datasets.items():
        expected = meta.get("sha256")
        if not expected:
            errors.append(f"missing sha256 for registry entry: {rel_path}")
            continue
        errors.extend(check_file(Path(rel_path), expected))
    return errors


def check_factor_registry(registry_path: Path) -> list[str]:
    errors: list[str] = []
    if not registry_path.exists():
        return [f"missing registry: {registry_path}"]
    with registry_path.open() as handle:
        registry = json.load(handle)
    datasets = registry.get("datasets", {})
    seen_paths: set[str] = set()
    for _, meta in datasets.items():
        path = meta.get("path")
        expected = meta.get("sha256")
        if not path:
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        if not expected:
            errors.append(f"missing sha256 for registry entry: {path}")
            continue
        errors.extend(check_file(Path(path), expected))
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_data_registry(Path("data/registry.json")))
    errors.extend(check_factor_registry(Path("data/factors/registry.json")))
    if errors:
        print("check_data_policy: FAILED")
        for err in errors:
            print(f"- {err}")
        return 1
    print("check_data_policy: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
