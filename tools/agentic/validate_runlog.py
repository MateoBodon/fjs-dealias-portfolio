#!/usr/bin/env python3
"""tools/agentic/validate_runlog.py

Validate that a run log folder exists and contains the minimum required files.

This is intentionally lightweight (no repo-specific heuristics).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List

REQUIRED = ["PROMPT.md", "COMMANDS.md", "RESULTS.md", "TESTS.md"]
META_CANONICAL = "META.json"
META_LEGACY = "META.md"
BUNDLE_STAMP_PATTERN = re.compile(r"BUNDLE_STAMP=(\d{8}_\d{6})")
RUN_NAME_STAMP_PATTERN = re.compile(r"^(\d{8}_\d{6})_")
BUNDLE_STAMP_PROVENANCE_CUTOFF = "20260216_000000"


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


def _validate_run_dir(
    run_dir: Path,
    *,
    progress_text: str,
    require_meta_json: bool = False,
    bundle_stamp_provenance_cutoff: str = BUNDLE_STAMP_PROVENANCE_CUTOFF,
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
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    progress_path = repo / "PROGRESS.md"
    progress_text = (
        progress_path.read_text(encoding="utf-8", errors="replace")
        if progress_path.exists()
        else ""
    )
    if args.all:
        ok = True
        for run_dir in _iter_run_dirs(repo):
            issues = _validate_run_dir(
                run_dir,
                progress_text=progress_text,
                require_meta_json=args.require_meta_json,
                bundle_stamp_provenance_cutoff=args.bundle_stamp_provenance_cutoff,
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
