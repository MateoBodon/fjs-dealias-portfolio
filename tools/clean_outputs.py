#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

TAGGED_PATTERN = re.compile(r"^[^/]+_J\d+_solver-[^/]+_est-[^/]+_prep-[^/]+$")


def is_tagged_directory(path: Path) -> bool:
    return path.is_dir() and TAGGED_PATTERN.match(path.name) is not None


def unique_destination(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    counter = 1
    while True:
        candidate = base.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def collect_legacy(root: Path) -> list[Path]:
    legacy: list[Path] = []
    for child in root.iterdir():
        if child.name == "archived":
            continue
        if is_tagged_directory(child):
            continue
        legacy.append(child)
    return sorted(legacy)


def clean_outputs(root: Path, *, purge: bool, dry_run: bool) -> None:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist.")
    legacy_items = collect_legacy(root)
    if not legacy_items:
        print(f"No legacy outputs found in {root}.")
        return

    archived_dir = root / "archived"
    for item in legacy_items:
        if purge:
            action = f"Would delete {item}" if dry_run else f"Deleting {item}"
            print(action)
            if not dry_run:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        else:
            dest_dir = archived_dir
            dest_dir.mkdir(exist_ok=True)
            destination = unique_destination(dest_dir / item.name)
            action = f"Would move {item} -> {destination}" if dry_run else f"Moving {item} -> {destination}"
            print(action)
            if not dry_run:
                shutil.move(str(item), str(destination))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive or purge untagged outputs in the smoke directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/equity_panel/outputs_smoke"),
        help="Root directory containing smoke outputs.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete legacy outputs instead of moving them to an archived/ folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without modifying the filesystem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_outputs(
        args.root,
        purge=args.purge,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
