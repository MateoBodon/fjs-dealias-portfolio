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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class StashState:
    dirty: bool
    stash_used: bool
    status_before: str
    stash_ref: str | None


def _run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )


def get_git_status_porcelain(repo: Path) -> str:
    result = _run_git(repo, ["status", "--porcelain"])
    if result.returncode != 0:
        msg = result.stderr.strip() or "git status failed"
        raise RuntimeError(msg)
    return result.stdout


def _stash_push(repo: Path, message: str) -> str:
    result = _run_git(repo, ["stash", "push", "-u", "-m", message])
    if result.returncode != 0:
        msg = result.stderr.strip() or "git stash push failed"
        raise RuntimeError(msg)
    ref = _run_git(repo, ["stash", "list", "-1", "--format=%gd"])
    if ref.returncode != 0:
        msg = ref.stderr.strip() or "git stash list failed"
        raise RuntimeError(msg)
    return ref.stdout.strip() or "stash@{0}"


def _stash_apply(repo: Path, stash_ref: str) -> None:
    result = _run_git(repo, ["stash", "apply", stash_ref])
    if result.returncode != 0:
        msg = result.stderr.strip() or "git stash apply failed"
        raise RuntimeError(msg)


def _stash_drop(repo: Path, stash_ref: str) -> None:
    result = _run_git(repo, ["stash", "drop", stash_ref])
    if result.returncode != 0:
        msg = result.stderr.strip() or "git stash drop failed"
        raise RuntimeError(msg)


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


def _prepare_repo(repo: Path, ticket: str, allow_stash: bool) -> StashState:
    status_before = get_git_status_porcelain(repo)
    dirty = bool(status_before.strip())
    print(f"dirty_detected: {'yes' if dirty else 'no'}", file=sys.stderr)
    if not dirty:
        print("stash_used: no", file=sys.stderr)
        return StashState(dirty=False, stash_used=False, status_before=status_before, stash_ref=None)
    if not allow_stash:
        print("stash_used: no", file=sys.stderr)
        raise RuntimeError(
            "Repository is dirty and --no-stash was set. "
            "Clean the working tree or rerun without --no-stash."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stash_ref = _stash_push(repo, f"temp: gpt_bundle {ticket} {stamp}")
    status_after = get_git_status_porcelain(repo)
    if status_after.strip():
        raise RuntimeError(
            "Failed to stash cleanly; working tree is still dirty after stashing."
        )
    print("stash_used: yes", file=sys.stderr)
    return StashState(dirty=True, stash_used=True, status_before=status_before, stash_ref=stash_ref)


def _restore_repo(repo: Path, stash_state: StashState) -> None:
    if not stash_state.stash_used:
        return
    stash_ref = stash_state.stash_ref or "stash@{0}"
    _stash_apply(repo, stash_ref)
    status_after = get_git_status_porcelain(repo)
    if status_after != stash_state.status_before:
        raise RuntimeError(
            "Working tree mismatch after restoring stash.\n"
            "The stash has NOT been dropped. Resolve manually and then drop it:\n"
            f"  git stash list\n  git stash drop {stash_ref}"
        )
    _stash_drop(repo, stash_ref)


def _extract_bundle_path(stdout: str) -> Path | None:
    candidates = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(candidates):
        if line.endswith(".zip"):
            return Path(line)
    return None


def _run_make_bundle(repo: Path, ticket: str, run_name: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "make",
        "gpt-bundle",
        f"TICKET={ticket}",
        f"RUN_NAME={run_name}",
    ]
    return subprocess.run(
        cmd,
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true", help="Create GPT bundle (default).")
    parser.add_argument("--ticket", required=True, help="Ticket id to include.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name (or set RUN_NAME env var).",
    )
    parser.add_argument(
        "--no-stash",
        action="store_true",
        help="Disable automatic stashing when the repo is dirty.",
    )
    args = parser.parse_args()

    run_name = args.run_name or os.environ.get("RUN_NAME")
    if not run_name:
        print("RUN_NAME is required (use --run-name or RUN_NAME env).", file=sys.stderr)
        return 1

    repo = Path.cwd()
    stash_state: StashState | None = None
    stdout = ""
    stderr = ""
    bundle_path: Path | None = None
    exit_code = 0
    restore_error: RuntimeError | None = None

    try:
        stash_state = _prepare_repo(repo, args.ticket, allow_stash=not args.no_stash)
        proc = _run_make_bundle(repo, args.ticket, run_name)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if proc.returncode != 0:
            exit_code = proc.returncode
        else:
            bundle_path = _extract_bundle_path(stdout)
            if not bundle_path:
                raise RuntimeError("Could not locate bundle path in gpt-bundle output.")
            if not bundle_path.exists():
                raise RuntimeError(f"Bundle path does not exist: {bundle_path}")
            update_bundle_meta_zip(bundle_path, git_dirty=stash_state.dirty if stash_state else False)
            print(f"bundle_path: {bundle_path}", file=sys.stderr)
    except RuntimeError as exc:
        exit_code = 1
        stderr = (stderr or "") + (str(exc) + "\n")
    finally:
        if stash_state and stash_state.stash_used:
            try:
                _restore_repo(repo, stash_state)
            except RuntimeError as exc:
                restore_error = exc

    if restore_error:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        print(str(restore_error), file=sys.stderr)
        return 1
    if exit_code != 0:
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
        return exit_code

    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
