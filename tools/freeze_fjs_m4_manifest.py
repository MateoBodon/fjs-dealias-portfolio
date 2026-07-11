#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fjs_m4_contract import build_manifest, stable_json_dumps  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Freeze the FJS M4 calibration manifest.")
    parser.add_argument(
        "--profile",
        choices=["full", "smoke"],
        default="full",
        help="Manifest profile to freeze.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to calibration/manifests/<manifest_id>.json.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=710_000,
        help="Base seed assigned to the first frozen cell.",
    )
    parser.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help="Optional bounded cell cap for a smoke/debug manifest rewrite.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    manifest = build_manifest(
        profile_name=args.profile,
        seed_base=int(args.seed_base),
        limit_cells=args.limit_cells,
    )
    out_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else Path("calibration/manifests") / f"{manifest['manifest_id']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
