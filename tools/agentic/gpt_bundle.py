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
from pathlib import Path


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
    cmd = [
        "make",
        "gpt-bundle",
        f"TICKET={args.ticket}",
        f"RUN_NAME={run_name}",
    ]
    proc = subprocess.run(cmd, cwd=str(repo))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
