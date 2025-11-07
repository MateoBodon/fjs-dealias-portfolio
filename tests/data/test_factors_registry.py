from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from data import factors as factor_mod


def _write_sample_csv(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "MKT": [0.001, -0.002],
            "SMB": [0.0, 0.0005],
            "HML": [0.0001, -0.0002],
            "RMW": [0.0003, 0.0004],
            "CMA": [-0.0001, 0.0002],
            "RF": [0.0, 0.0],
            "MOM": [0.0009, -0.0001],
        }
    )
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_load_registered_factors(tmp_path, monkeypatch):
    csv_path = tmp_path / "ff5mom.csv"
    _write_sample_csv(csv_path)
    registry_payload = {
        "generated_at": "2025-11-07T00:00:00Z",
        "maintainer": "tests",
        "datasets": {
            "ff5mom": {
                "path": csv_path.as_posix(),
                "sha256": _sha256(csv_path),
                "columns": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM", "RF"],
            }
        },
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")
    monkeypatch.setenv("FACTOR_REGISTRY_PATH", registry_path.as_posix())
    factor_mod.get_factor_registry.cache_clear()
    factor_mod.assert_registered_factor.cache_clear()

    frame, entry = factor_mod.load_registered_factors(
        "ff5mom",
        start="2024-01-02",
        end="2024-01-03",
    )

    assert entry.key == "ff5mom"
    assert entry.path == csv_path.resolve()
    assert not frame.empty
    assert pd.Timestamp("2024-01-02") in frame.index
    assert frame.loc[pd.Timestamp("2024-01-03"), "MKT"] == -0.002

    factor_mod.get_factor_registry.cache_clear()
    factor_mod.assert_registered_factor.cache_clear()
