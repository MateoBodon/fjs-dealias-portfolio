from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data import registry as reg


def _write_registry(path: Path, dataset_key: str, sha: str) -> None:
    payload = {
        "generated_at": "2025-11-06T00:00:00Z",
        "datasets": {
            dataset_key: {
                "sha256": sha,
                "rows": 1,
                "columns": 1,
                "start_date": "2020-01-01",
                "end_date": "2020-01-01",
                "wrds_source": "crsp.dsf",
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_assert_registered_dataset_valid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True)
    dataset = data_dir / "returns_daily.csv"
    dataset.write_text("date,ticker,ret\n2020-01-01,A,0.1\n", encoding="utf-8")

    sha = _sha256(dataset)
    registry_path = data_dir / "registry.json"
    _write_registry(registry_path, "data/returns_daily.csv", sha)

    monkeypatch.setattr(reg, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(reg, "_registry_path", lambda: registry_path)
    reg.get_registry.cache_clear()
    reg._assert_registered_dataset.cache_clear()

    entry = reg.assert_registered_dataset(dataset)
    assert entry.sha256 == sha
    assert entry.path == "data/returns_daily.csv"


def test_assert_registered_dataset_hash_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True)
    dataset = data_dir / "returns_daily.csv"
    dataset.write_text("date,ticker,ret\n2020-01-01,A,0.1\n", encoding="utf-8")

    sha = _sha256(dataset)
    registry_path = data_dir / "registry.json"
    _write_registry(registry_path, "data/returns_daily.csv", sha)

    monkeypatch.setattr(reg, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(reg, "_registry_path", lambda: registry_path)
    reg.get_registry.cache_clear()
    reg._assert_registered_dataset.cache_clear()

    # mutate dataset to trigger mismatch
    dataset.write_text("date,ticker,ret\n2020-01-01,A,0.2\n", encoding="utf-8")
    reg._assert_registered_dataset.cache_clear()

    with pytest.raises(reg.DatasetRegistryError):
        reg.assert_registered_dataset(dataset)
