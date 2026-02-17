#!/usr/bin/env python3
"""tools/agentic/validate_runlog.py

Validate that a run log folder exists and contains the minimum required files.

This is intentionally lightweight (no repo-specific heuristics).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

REQUIRED = ["PROMPT.md", "COMMANDS.md", "RESULTS.md", "TESTS.md"]
META_CANONICAL = "META.json"
META_LEGACY = "META.md"
BUNDLE_STAMP_PATTERN = re.compile(r"BUNDLE_STAMP=(\d{8}_\d{6})")
RUN_NAME_STAMP_PATTERN = re.compile(r"^(\d{8}_\d{6})_")
BUNDLE_STAMP_PROVENANCE_CUTOFF = "20260216_000000"
META_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
META_SHA_MATCH_CUTOFF = "20260216_000000"
META_SHA_PLACEHOLDERS = {"", "tbd", "none", "null", "n/a", "na"}


def _iter_run_dirs(repo: Path) -> Iterable[Path]:
    runs_root = repo / "docs" / "agent_runs"
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def _extract_bundle_stamps(commands_text: str) -> List[str]:
    unique_stamps: List[str] = []
    for stamp in BUNDLE_STAMP_PATTERN.findall(commands_text):
        if stamp not in unique_stamps:
            unique_stamps.append(stamp)
    return unique_stamps


def _run_name_stamp(run_name: str) -> str | None:
    match = RUN_NAME_STAMP_PATTERN.match(run_name)
    if match is None:
        return None
    return match.group(1)


def _run_at_or_after_cutoff(run_name: str, cutoff: str) -> bool:
    run_stamp = _run_name_stamp(run_name)
    if run_stamp is None:
        return False
    return run_stamp >= cutoff


def _progress_mentions_bundle_stamp(progress_text: str, run_name: str, stamp: str) -> bool:
    pattern = re.compile(
        rf"artifacts/_local/gpt_bundles/{re.escape(stamp)}_[^`\s]+_{re.escape(run_name)}\.zip"
    )
    return bool(pattern.search(progress_text))


def _validate_bundle_stamp_provenance(
    run_dir: Path,
    progress_text: str,
    *,
    bundle_stamp_provenance_cutoff: str,
) -> List[str]:
    commands_path = run_dir / "COMMANDS.md"
    if not commands_path.exists():
        return []

    commands_text = commands_path.read_text(encoding="utf-8", errors="replace")
    bundle_stamps = _extract_bundle_stamps(commands_text)
    if len(bundle_stamps) <= 1:
        return []

    run_stamp = _run_name_stamp(run_dir.name)
    if run_stamp is None or run_stamp < bundle_stamp_provenance_cutoff:
        return []

    if not progress_text:
        return [
            "PROGRESS.md missing or unreadable; "
            f"cannot validate final bundle stamp for {run_dir.name}."
        ]

    final_stamp = bundle_stamps[-1]
    if _progress_mentions_bundle_stamp(progress_text, run_dir.name, final_stamp):
        return []

    return [
        "final BUNDLE_STAMP provenance mismatch: "
        f"run={run_dir.name} stamps={bundle_stamps} final={final_stamp}; "
        "expected PROGRESS.md to reference "
        f"`artifacts/_local/gpt_bundles/{final_stamp}_*_{run_dir.name}.zip`."
    ]


def _validate_meta_sha_after_matches_expected_head(
    run_dir: Path,
    *,
    expected_head_sha: str | None,
    meta_sha_cutoff: str,
) -> List[str]:
    if expected_head_sha is None:
        return []
    if not _run_at_or_after_cutoff(run_dir.name, meta_sha_cutoff):
        return []
    if not META_SHA_PATTERN.fullmatch(expected_head_sha):
        return [f"expected head SHA is invalid: {expected_head_sha!r}"]

    meta_path = run_dir / META_CANONICAL
    if not meta_path.exists():
        # Missing META.json is already reported by required file checks.
        return []

    try:
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return [f"META.json is invalid JSON in {run_dir.name}: {exc.msg}"]

    raw_after_obj = meta_obj.get("git_sha_after")
    raw_after = str(raw_after_obj).strip() if raw_after_obj is not None else ""
    normalized = raw_after.lower()
    if normalized in META_SHA_PLACEHOLDERS:
        return [
            "META.json git_sha_after is placeholder for "
            f"{run_dir.name}; expected {expected_head_sha}."
        ]
    if not META_SHA_PATTERN.fullmatch(raw_after):
        return [
            "META.json git_sha_after must be a full 40-char SHA for "
            f"{run_dir.name}; got {raw_after!r}."
        ]
    if raw_after.lower() != expected_head_sha.lower():
        return [
            "META.json git_sha_after mismatch for "
            f"{run_dir.name}: {raw_after} != {expected_head_sha} (bundle head_sha)."
        ]
    return []


def _validate_run_dir(
    run_dir: Path,
    *,
    progress_text: str,
    require_meta_json: bool = False,
    bundle_stamp_provenance_cutoff: str = BUNDLE_STAMP_PROVENANCE_CUTOFF,
    expected_head_sha: str | None = None,
    meta_sha_cutoff: str = META_SHA_MATCH_CUTOFF,
) -> List[str]:
    issues: List[str] = []
    for f in REQUIRED:
        if not (run_dir / f).exists():
            issues.append(f)
    has_meta_json = (run_dir / META_CANONICAL).exists()
    has_meta_md = (run_dir / META_LEGACY).exists()
    if require_meta_json:
        if not has_meta_json:
            issues.append(META_CANONICAL)
    elif not (has_meta_json or has_meta_md):
        issues.append(f"{META_CANONICAL} (or legacy {META_LEGACY})")
    issues.extend(
        _validate_bundle_stamp_provenance(
            run_dir,
            progress_text,
            bundle_stamp_provenance_cutoff=bundle_stamp_provenance_cutoff,
        )
    )
    issues.extend(
        _validate_meta_sha_after_matches_expected_head(
            run_dir,
            expected_head_sha=expected_head_sha,
            meta_sha_cutoff=meta_sha_cutoff,
        )
    )
    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None, help="Run folder name under docs/agent_runs/")
    ap.add_argument("--path", default=None, help="Explicit path to the run folder")
    ap.add_argument("--repo", default=".", help="Path inside repo (default: .)")
    ap.add_argument("--all", action="store_true", help="Validate every run folder under docs/agent_runs/")
    ap.add_argument(
        "--require-meta-json",
        action="store_true",
        help="Require META.json (no legacy META.md fallback).",
    )
    ap.add_argument(
        "--bundle-stamp-provenance-cutoff",
        default=BUNDLE_STAMP_PROVENANCE_CUTOFF,
        help=(
            "Run-name timestamp cutoff (YYYYMMDD_HHMMSS) for enforcing final "
            "BUNDLE_STAMP provenance in PROGRESS.md when multiple bundle stamps exist."
        ),
    )
    ap.add_argument(
        "--expected-head-sha",
        default=None,
        help=(
            "Expected bundle head SHA for run-level metadata validation. "
            "When provided, runs at/after --meta-sha-cutoff fail if META.json "
            "git_sha_after is placeholder or does not match."
        ),
    )
    ap.add_argument(
        "--meta-sha-cutoff",
        default=META_SHA_MATCH_CUTOFF,
        help=(
            "Run-name timestamp cutoff (YYYYMMDD_HHMMSS) for enforcing META.json "
            "git_sha_after checks when --expected-head-sha is set."
        ),
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    progress_path = repo / "PROGRESS.md"
    progress_text = (
        progress_path.read_text(encoding="utf-8", errors="replace")
        if progress_path.exists()
        else ""
    )
    if args.all:
        if args.expected_head_sha:
            raise SystemExit("--expected-head-sha cannot be used with --all.")
        ok = True
        for run_dir in _iter_run_dirs(repo):
            issues = _validate_run_dir(
                run_dir,
                progress_text=progress_text,
                require_meta_json=args.require_meta_json,
                bundle_stamp_provenance_cutoff=args.bundle_stamp_provenance_cutoff,
                expected_head_sha=None,
                meta_sha_cutoff=args.meta_sha_cutoff,
            )
            if issues:
                ok = False
                print(f"FAIL: validation issues in {run_dir}:")
                for m in issues:
                    print(f"  - {m}")
            else:
                print(f"OK: {run_dir}")
                if not args.require_meta_json and not (run_dir / META_CANONICAL).exists() and (run_dir / META_LEGACY).exists():
                    print(f"WARN: {run_dir} uses legacy {META_LEGACY}; prefer {META_CANONICAL}.")
        return 0 if ok else 2
    if args.path:
        run_dir = Path(args.path).resolve()
    elif args.run_name:
        run_dir = repo / "docs" / "agent_runs" / args.run_name
    else:
        raise SystemExit("Provide --run-name or --path")

    if not run_dir.exists():
        print(f"FAIL: run dir missing: {run_dir}")
        return 2

    issues = _validate_run_dir(
        run_dir,
        progress_text=progress_text,
        require_meta_json=args.require_meta_json,
        bundle_stamp_provenance_cutoff=args.bundle_stamp_provenance_cutoff,
        expected_head_sha=args.expected_head_sha,
        meta_sha_cutoff=args.meta_sha_cutoff,
    )
    if issues:
        print(f"FAIL: validation issues in {run_dir}:")
        for m in issues:
            print(f"  - {m}")
        return 2
    print(f"OK: {run_dir}")
    if not args.require_meta_json and not (run_dir / META_CANONICAL).exists() and (run_dir / META_LEGACY).exists():
        print(f"WARN: {run_dir} uses legacy {META_LEGACY}; prefer {META_CANONICAL}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
