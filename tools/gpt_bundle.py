"""Helpers for GPT bundle packaging."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class DiffPatchError(RuntimeError):
    """Raised when diff patch generation fails."""


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_diff_patch(repo_root: Path, output_path: Path, rev: str = "HEAD") -> int:
    """Write a non-empty diff patch for a specific git revision."""
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()

    rev_check = _run_git(repo_root, ["rev-parse", "--verify", rev])
    if rev_check.returncode != 0:
        msg = rev_check.stderr.decode("utf-8", errors="replace").strip()
        raise DiffPatchError(f"git rev-parse failed for {rev}: {msg}")

    diff = _run_git(repo_root, ["show", "--patch", "--stat", "--binary", rev])
    if diff.returncode != 0:
        msg = diff.stderr.decode("utf-8", errors="replace").strip()
        raise DiffPatchError(f"git show failed for {rev}: {msg}")

    if not diff.stdout:
        raise DiffPatchError(f"DIFF.patch would be empty for {rev}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(diff.stdout)
    return len(diff.stdout)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT bundle helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_parser = subparsers.add_parser("diff", help="Write DIFF.patch output")
    diff_parser.add_argument("--repo", default=".", help="Git repo root")
    diff_parser.add_argument("--rev", default="HEAD", help="Git revision to show")
    diff_parser.add_argument("--output", required=True, help="Output path for DIFF.patch")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diff":
        try:
            write_diff_patch(Path(args.repo), Path(args.output), rev=args.rev)
        except DiffPatchError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
