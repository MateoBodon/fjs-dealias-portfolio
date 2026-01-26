#!/usr/bin/env python3
"""Repo-specific GPT bundle wrapper.

Delegates to the Makefile gpt-bundle target to preserve auditability
requirements (merge-base diff + required files).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def get_git_status_porcelain(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or "git status failed"
        raise RuntimeError(msg)
    return result.stdout


def _set_meta_field(text: str, key: str, value: str) -> str:
    prefix = f"{key}:"
    lines = text.splitlines()
    updated = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            new_lines.append(f"{prefix} {value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{prefix} {value}")
    return "\n".join(new_lines) + "\n"


def update_bundle_meta_zip(zip_path: Path, git_dirty: bool) -> None:
    zip_path = zip_path.resolve()
    if not zip_path.exists():
        raise RuntimeError(f"Bundle zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as bundle:
        try:
            meta_bytes = bundle.read("BUNDLE_META.md")
        except KeyError as exc:
            raise RuntimeError("BUNDLE_META.md missing from bundle") from exc
        entries = bundle.infolist()
        payloads = {entry.filename: bundle.read(entry.filename) for entry in entries}

    meta_text = meta_bytes.decode("utf-8", errors="replace")
    updated_meta = _set_meta_field(meta_text, "git_dirty", "true" if git_dirty else "false")

    tmp_path = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp_path, "w") as out:
        for entry in entries:
            if entry.filename == "BUNDLE_META.md":
                continue
            out.writestr(entry, payloads[entry.filename])
        out.writestr("BUNDLE_META.md", updated_meta)
    tmp_path.replace(zip_path)


def _extract_bundle_path(stdout: str) -> Path | None:
    candidates = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(candidates):
        if line.endswith(".zip"):
            return Path(line)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true", help="Create GPT bundle (default).")
    parser.add_argument("--ticket", required=True, help="Ticket id to include.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name (or set RUN_NAME env var).",
    )
    args = parser.parse_args()

    run_name = args.run_name or os.environ.get("RUN_NAME")
    if not run_name:
        print("RUN_NAME is required (use --run-name or RUN_NAME env).", file=sys.stderr)
        return 1

    repo = Path.cwd()
    try:
        status = get_git_status_porcelain(repo)
    except RuntimeError as exc:
        print(f"Failed to check git status: {exc}", file=sys.stderr)
        return 1
    if status.strip():
        print(
            "Repository is dirty; gpt-bundle requires a clean working tree.",
            file=sys.stderr,
        )
        print("git status --porcelain output:", file=sys.stderr)
        print(status.rstrip(), file=sys.stderr)
        return 1
    cmd = [
        "make",
        "gpt-bundle",
        f"TICKET={args.ticket}",
        f"RUN_NAME={run_name}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        return proc.returncode

    bundle_path = _extract_bundle_path(stdout)
    if not bundle_path:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        print("Could not locate bundle path in gpt-bundle output.", file=sys.stderr)
        return 1
    if not bundle_path.exists():
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        print(f"Bundle path does not exist: {bundle_path}", file=sys.stderr)
        return 1
    try:
        update_bundle_meta_zip(bundle_path, git_dirty=False)
    except RuntimeError as exc:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        print(str(exc), file=sys.stderr)
        return 1

    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
