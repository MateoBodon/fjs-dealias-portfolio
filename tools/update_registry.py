#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def summarise_returns(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, parse_dates=["date"])
    return {
        "rows": int(len(frame)),
        "columns": int(frame["ticker"].nunique()),
        "start_date": frame["date"].min().date().isoformat() if not frame.empty else None,
        "end_date": frame["date"].max().date().isoformat() if not frame.empty else None,
    }


def load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "generated_at": None,
        "datasets": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Update data/registry.json with refreshed dataset metadata.")
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the WRDS-derived dataset (e.g. data/returns_daily.csv).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "registry.json",
        help="Path to the registry JSON (default: data/registry.json).",
    )
    parser.add_argument(
        "--wrds-source",
        type=str,
        default=None,
        help="Optional WRDS table identifier (e.g. crsp.dsf).",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Free-form note saved alongside the dataset entry.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset does not exist: {dataset_path}")

    repo_root = Path(__file__).resolve().parents[1]
    try:
        key = dataset_path.relative_to(repo_root).as_posix()
    except ValueError:
        key = dataset_path.as_posix()

    registry_path = args.registry
    registry = load_registry(registry_path)
    datasets = registry.setdefault("datasets", {})

    sha = compute_sha256(dataset_path)
    summary = summarise_returns(dataset_path)

    entry = datasets.setdefault(key, {})
    entry.update(
        {
            "sha256": sha,
            "rows": summary["rows"],
            "columns": summary["columns"],
            "start_date": summary["start_date"],
            "end_date": summary["end_date"],
        }
    )
    if args.wrds_source:
        entry["wrds_source"] = args.wrds_source
    if args.note:
        entry["note"] = args.note

    registry["generated_at"] = datetime.now(timezone.utc).isoformat()

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)

    print(f"Updated {registry_path} entry for {key}")
    print(f"sha256={sha}")


if __name__ == "__main__":
    main()
