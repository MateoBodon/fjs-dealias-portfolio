#!/usr/bin/env python3
"""tools/agentic/validate_runlog.py

Validate that a run log folder exists and contains the minimum required files.

This is intentionally lightweight (no repo-specific heuristics).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

REQUIRED = ["PROMPT.md", "COMMANDS.md", "RESULTS.md", "TESTS.md"]
META_CANONICAL = "META.json"
META_LEGACY = "META.md"


def _iter_run_dirs(repo: Path) -> Iterable[Path]:
    runs_root = repo / "docs" / "agent_runs"
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def _validate_run_dir(run_dir: Path, require_meta_json: bool = False) -> List[str]:
    missing: List[str] = []
    for f in REQUIRED:
        if not (run_dir / f).exists():
            missing.append(f)
    has_meta_json = (run_dir / META_CANONICAL).exists()
    has_meta_md = (run_dir / META_LEGACY).exists()
    if require_meta_json:
        if not has_meta_json:
            missing.append(META_CANONICAL)
    elif not (has_meta_json or has_meta_md):
        missing.append(f"{META_CANONICAL} (or legacy {META_LEGACY})")
    return missing


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
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if args.all:
        ok = True
        for run_dir in _iter_run_dirs(repo):
            missing = _validate_run_dir(run_dir, require_meta_json=args.require_meta_json)
            if missing:
                ok = False
                print(f"FAIL: missing files in {run_dir}:")
                for m in missing:
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

    missing = _validate_run_dir(run_dir, require_meta_json=args.require_meta_json)
    if missing:
        print(f"FAIL: missing files in {run_dir}:")
        for m in missing:
            print(f"  - {m}")
        return 2
    print(f"OK: {run_dir}")
    if not args.require_meta_json and not (run_dir / META_CANONICAL).exists() and (run_dir / META_LEGACY).exists():
        print(f"WARN: {run_dir} uses legacy {META_LEGACY}; prefer {META_CANONICAL}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
