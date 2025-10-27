from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data.panels import PanelManifest


def window_key(
    manifest: PanelManifest,
    week_list: Iterable[str],
    tickers: Iterable[str],
    replicates: int,
) -> str:
    """Stable hash identifying a per-window cache entry."""

    payload = {
        "data_hash": manifest.data_hash,
        "universe_hash": manifest.universe_hash,
        "partial_week_policy": manifest.partial_week_policy,
        "days_per_week": manifest.days_per_week,
        "weeks": list(week_list),
        "tickers": list(tickers),
        "replicates": int(replicates),
        "preprocess_flags": dict(manifest.preprocess_flags),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_window(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    """Persist cached per-window statistics."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / f"{key}.json"
    npz_path = cache_dir / f"{key}.npz"

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    for name, value in payload.items():
        if isinstance(value, np.ndarray):
            arrays[name] = value
        else:
            scalars[name] = value

    if arrays:
        np.savez_compressed(npz_path, **arrays)
        scalars["_arrays"] = sorted(arrays.keys())
    elif npz_path.exists():
        npz_path.unlink()

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(scalars, handle, sort_keys=True, indent=2)


def load_window(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Load cached per-window statistics if available."""

    json_path = cache_dir / f"{key}.json"
    if not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            scalars = json.load(handle)
    except Exception:
        return None

    arrays_list = scalars.pop("_arrays", [])
    result: dict[str, Any] = dict(scalars)

    if arrays_list:
        npz_path = cache_dir / f"{key}.npz"
        if not npz_path.exists():
            return None
        try:
            with np.load(npz_path) as data:
                for name in arrays_list:
                    if name in data.files:
                        result[name] = np.array(data[name])
                    else:
                        return None
        except Exception:
            return None

    return result
