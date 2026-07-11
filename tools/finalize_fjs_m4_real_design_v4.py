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

from fjs.real_design_contract import stable_json_dumps  # noqa: E402
from fjs.real_design_finalizer import (  # noqa: E402
    build_cell_receipt,
    build_final_manifest,
    checkpoint_status,
    independent_readback,
    load_checkpoint,
    new_checkpoint,
    register_cell,
    write_checkpoint,
    write_final_manifest,
    write_readback,
)


def _print(payload: object) -> None:
    if isinstance(payload, (dict, list)):
        print(stable_json_dumps(payload))
    else:
        print(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Restart-safe FJS M4 v4 72-month finalizer and independent readback."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--checkpoint", type=Path, required=True)
    init.add_argument("--generation-id", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--month", required=True)
    register.add_argument("--cell", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--checkpoint", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--checkpoint", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)

    readback = subparsers.add_parser("readback")
    readback.add_argument("--manifest", type=Path, required=True)
    readback.add_argument("--receipt-out", type=Path, default=None)
    return parser.parse_args(argv)


def _init(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint["generation_id"] != args.generation_id:
            raise ValueError(
                "Existing checkpoint belongs to another generation: "
                f"{checkpoint['generation_id']!r}."
            )
    else:
        checkpoint = new_checkpoint(str(args.generation_id))
        write_checkpoint(checkpoint, checkpoint_path)
    _print(checkpoint_status(checkpoint))
    return checkpoint_path


def _register(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    receipt = build_cell_receipt(
        generation_id=str(checkpoint["generation_id"]),
        month=str(args.month),
        cell_path=args.cell,
    )
    updated = register_cell(checkpoint, receipt)
    write_checkpoint(updated, checkpoint_path)
    _print(checkpoint_status(updated))
    return checkpoint_path


def _status(args: argparse.Namespace) -> Path:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    _print(checkpoint_status(load_checkpoint(checkpoint_path)))
    return checkpoint_path


def _finalize(args: argparse.Namespace) -> Path:
    checkpoint = load_checkpoint(args.checkpoint, revalidate_artifacts=True)
    manifest = build_final_manifest(checkpoint)
    out = write_final_manifest(manifest, args.out)
    readback = independent_readback(out)
    _print(
        {
            "manifest_path": str(out),
            "manifest_file_sha256": readback["manifest_file_sha256"],
            "manifest_digest": readback["manifest_digest"],
            "source_set_digest": readback["source_set_digest"],
            "cell_set_digest": readback["cell_set_digest"],
            "independent_readback_passed": True,
            "full_execution_ready": False,
            "aws_execution_authorized": False,
            "outcomes_present": False,
            "holdout_2025_opened": False,
        }
    )
    return out


def _readback(args: argparse.Namespace) -> Path:
    manifest_path = args.manifest.expanduser().resolve()
    receipt = independent_readback(manifest_path)
    if args.receipt_out is not None:
        write_readback(receipt, args.receipt_out)
    _print(receipt)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.command == "init":
        return _init(args)
    if args.command == "register":
        return _register(args)
    if args.command == "status":
        return _status(args)
    if args.command == "finalize":
        return _finalize(args)
    if args.command == "readback":
        return _readback(args)
    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
