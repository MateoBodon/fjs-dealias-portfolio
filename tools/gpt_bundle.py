"""Helpers for GPT bundle packaging."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class DiffPatchError(RuntimeError):
    """Raised when diff patch generation fails."""


BASE_REF_CANDIDATES = ("origin/main", "origin/master", "main", "master")


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _ref_exists(repo_root: Path, ref: str) -> bool:
    return _run_git(repo_root, ["rev-parse", "--verify", ref]).returncode == 0


def _rev_parse(repo_root: Path, ref: str) -> str:
    result = _run_git(repo_root, ["rev-parse", "--verify", ref])
    if result.returncode != 0:
        msg = result.stderr.decode("utf-8", errors="replace").strip()
        raise DiffPatchError(f"git rev-parse failed for {ref}: {msg}")
    return result.stdout.decode("utf-8", errors="replace").strip()


def resolve_base_ref(
    repo_root: Path,
    base_override: str | None = None,
    candidates: Iterable[str] = BASE_REF_CANDIDATES,
) -> str:
    """Resolve the base ref for bundle diffs."""
    if base_override:
        if not _ref_exists(repo_root, base_override):
            raise DiffPatchError(
                f"BUNDLE_BASE ref {base_override} could not be resolved; "
                "set BUNDLE_BASE to a valid ref or SHA."
            )
        return base_override

    for ref in candidates:
        if _ref_exists(repo_root, ref):
            return ref

    raise DiffPatchError(
        "Could not resolve base ref (tried origin/main, origin/master, main, master). "
        "Set BUNDLE_BASE to a valid ref or SHA."
    )


def _merge_base(repo_root: Path, base_ref: str, head_ref: str) -> str:
    result = _run_git(repo_root, ["merge-base", base_ref, head_ref])
    if result.returncode != 0:
        msg = result.stderr.decode("utf-8", errors="replace").strip()
        raise DiffPatchError(f"git merge-base failed for {base_ref}..{head_ref}: {msg}")
    base_sha = result.stdout.decode("utf-8", errors="replace").strip()
    if not base_sha:
        raise DiffPatchError(f"git merge-base returned empty for {base_ref}..{head_ref}")
    return base_sha


def write_range_diff(
    repo_root: Path,
    output_path: Path,
    base_ref: str,
    head_ref: str = "HEAD",
) -> dict[str, str]:
    """Write a non-empty diff patch for a git range."""
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()

    base_sha = _merge_base(repo_root, base_ref, head_ref)
    head_sha = _rev_parse(repo_root, head_ref)
    diff_command = f"git diff --binary {base_sha}..{head_sha}"

    diff = _run_git(repo_root, ["diff", "--binary", f"{base_sha}..{head_sha}"])
    if diff.returncode != 0:
        msg = diff.stderr.decode("utf-8", errors="replace").strip()
        raise DiffPatchError(f"git diff failed for {base_sha}..{head_sha}: {msg}")

    if not diff.stdout:
        raise DiffPatchError(f"DIFF.patch would be empty for {base_sha}..{head_sha}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(diff.stdout)
    return {
        "base_sha": base_sha,
        "head_sha": head_sha,
        "diff_command": diff_command,
        "size": str(len(diff.stdout)),
    }


def write_bundle_meta(
    output_path: Path,
    *,
    run_name: str,
    ticket: str,
    base_ref: str,
    base_sha: str,
    head_sha: str,
    diff_command: str,
    timestamp_utc: str | None = None,
) -> None:
    """Write bundle metadata to a markdown file."""
    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"run_name: {run_name}",
        f"ticket: {ticket}",
        f"base_ref: {base_ref}",
        f"base_sha: {base_sha}",
        f"head_sha: {head_sha}",
        f"diff_command: {diff_command}",
        f"timestamp_utc: {timestamp_utc}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT bundle helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_parser = subparsers.add_parser("diff", help="Write DIFF.patch output")
    diff_parser.add_argument("--repo", default=".", help="Git repo root")
    diff_parser.add_argument(
        "--base-ref",
        default=None,
        help="Base ref to diff from (optional; otherwise auto-detected or BUNDLE_BASE).",
    )
    diff_parser.add_argument("--head-ref", default="HEAD", help="Head ref to diff to")
    diff_parser.add_argument("--output", required=True, help="Output path for DIFF.patch")
    diff_parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional output path for BUNDLE_META.md",
    )
    diff_parser.add_argument("--run-name", default=None, help="Run name for metadata")
    diff_parser.add_argument("--ticket", default=None, help="Ticket id for metadata")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diff":
        try:
            repo_root = Path(args.repo)
            base_override = args.base_ref or os.environ.get("BUNDLE_BASE")
            base_ref = resolve_base_ref(repo_root, base_override=base_override)
            meta = write_range_diff(
                repo_root,
                Path(args.output),
                base_ref=base_ref,
                head_ref=args.head_ref,
            )
            if args.meta_output:
                if not args.run_name or not args.ticket:
                    raise DiffPatchError("--meta-output requires --run-name and --ticket")
                write_bundle_meta(
                    Path(args.meta_output),
                    run_name=args.run_name,
                    ticket=args.ticket,
                    base_ref=base_ref,
                    base_sha=meta["base_sha"],
                    head_sha=meta["head_sha"],
                    diff_command=meta["diff_command"],
                )
        except DiffPatchError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
