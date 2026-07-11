#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_path in (ROOT, SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.fjs_m4_contract import stable_json_dumps  # noqa: E402
from tools.fjs_m4_contract_v3 import build_manifest_v3  # noqa: E402


def default_manifest_path(manifest_id: str) -> Path:
    filename = f"{str(manifest_id).replace('-', '_')}.json"
    return ROOT / "calibration" / "manifests" / filename


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Freeze the FJS M4 v3 calibration manifest.")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed-base", type=int, default=730_000)
    parser.add_argument("--limit-cells", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    manifest = build_manifest_v3(
        profile_name=str(args.profile),
        seed_base=int(args.seed_base),
        limit_cells=args.limit_cells,
    )
    out_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else default_manifest_path(str(manifest["manifest_id"]))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    main()
