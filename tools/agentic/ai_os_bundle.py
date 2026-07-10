#!/usr/bin/env python3
"""Build AI Project OS v2 bundles.

Profiles:
  - project_state_audit: compact Pro-facing state bundle.
  - review: compact ticket-review bundle.

This script is intentionally conservative. It copies selected text/config/source
files, summarizes large artifacts, and avoids raw data or bulky generated trees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}

CANONICAL_DOCS = [
    "PROJECT.md",
    "AGENTS.md",
    "PROGRESS.md",
    "README.md",
    "docs/strategy/GOAL_CONTEXT.md",
    "docs/strategy/STRATEGIC_OVERVIEW.md",
    "docs/strategy/PLAN_OF_RECORD.md",
    "docs/strategy/DECISIONS.md",
    "docs/strategy/RISK_REGISTER.md",
    "docs/strategy/TICKET_LEDGER.md",
    "docs/strategy/CODEX_GOALS.md",
    "docs/strategy/CONTEXT_CARRYOVER.md",
    "docs/tickets/T-000_install_ai_project_os_v2.md",
    "docs/tickets/TEMPLATE_codex_ticket.md",
    "project_state/STATE_INDEX.md",
    "project_state/RUNBOOK.md",
    "project_state/VALIDATION_MATRIX.md",
    "project_state/CLAIMS_AND_EVIDENCE.md",
    "project_state/CURRENT_RESULTS.md",
    "project_state/KNOWN_ISSUES.md",
    "project_state/CONFIG_REFERENCE.md",
]

SELECTED_SOURCE = [
    "src/fjs/gating.py",
    "src/fjs/mp.py",
    "src/fjs/overlay.py",
    "src/fjs/robust.py",
    "src/baselines/covariance.py",
    "src/eval/balance.py",
    "src/finance/portfolios.py",
    "experiments/eval/run.py",
    "experiments/eval/inject_spike.py",
    "experiments/equity_panel/run.py",
    "tools/gpt_bundle.py",
    "tools/agentic/gpt_bundle.py",
    "tools/agentic/project_state_refresh.py",
    "tools/agentic/validate_runlog.py",
    "tools/agentic/ai_os_bundle.py",
]

SELECTED_TESTS = [
    "tests/test_gpt_bundle.py",
    "tests/test_validate_runlog.py",
    "tests/test_repo_hygiene.py",
    "tests/experiments/test_eval_run.py",
    "tests/experiments/test_inject_spike.py",
    "tests/test_gating.py",
    "tests/test_evaluation_checks.py",
]

SELECTED_CONFIGS = [
    "Makefile",
    "pyproject.toml",
    "pytest.ini",
    "ruff.toml",
    "mypy.ini",
    ".github/workflows/ci.yml",
    ".github/workflows/smoke.yml",
    ".gitignore",
    ".env.example",
    "calibration_defaults.json",
    "calibration/edge_delta_thresholds.json",
    "calibration/nested_edge_delta_thresholds.json",
    "data/registry.json",
    "data/factors/registry.json",
]


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def git_root(start: Path) -> Path:
    code, out = run(["git", "rev-parse", "--show-toplevel"], start)
    if code != 0:
        raise SystemExit("not inside a git repository")
    return Path(out.strip())


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def format_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def write_text(stage: Path, rel: str, text: str) -> None:
    out = stage / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def copy_file(repo: Path, stage: Path, source_rel: str, dest_rel: str | None = None) -> bool:
    src = repo / source_rel
    if not src.exists() or not src.is_file():
        return False
    dest = stage / (dest_rel or source_rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))


def iter_repo_files(repo: Path, max_size: int | None = None) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo):
        root_path = Path(root)
        parts = set(root_path.relative_to(repo).parts)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and d != "_bundles"]
        if parts & SKIP_DIRS:
            continue
        for name in files:
            path = root_path / name
            try:
                if max_size is not None and path.stat().st_size > max_size:
                    continue
            except OSError:
                continue
            yield path


def text_head(path: Path, limit: int = 12000) -> str:
    try:
        raw = path.read_bytes()[:limit]
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def latest_archive_index(repo: Path) -> Path | None:
    base = repo / "docs" / "_archive" / "pre_ai_os_v2"
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/ARCHIVE_INDEX.md"))
    return candidates[-1] if candidates else None


def git_summary(repo: Path) -> dict[str, str]:
    outputs = {}
    for key, cmd in {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status": ["git", "status", "--short"],
        "log": ["git", "log", "-n", "12", "--oneline", "--decorate"],
        "diff_stat": ["git", "diff", "--stat"],
    }.items():
        _, out = run(cmd, repo)
        outputs[key] = out.strip()
    return outputs


def docs_index(repo: Path) -> str:
    rows = []
    for path in sorted(iter_repo_files(repo, max_size=1024 * 1024)):
        rel = relpath(path, repo)
        if path.suffix.lower() not in {".md", ".txt", ".rst", ".json", ".yaml", ".yml"}:
            continue
        if rel.startswith("reports/_bundles/"):
            continue
        if rel.startswith("docs/_archive/pre_ai_os_v2/") and "snapshot/" in rel:
            continue
        category = "historical/generated"
        if rel in CANONICAL_DOCS or rel.startswith("docs/strategy/") or rel.startswith("project_state/"):
            category = "canonical/current"
        elif rel.startswith("docs/agent_runs/"):
            category = "run-log/history"
        elif rel.startswith("reports/"):
            category = "artifact/report"
        elif rel.startswith("docs/_archive/"):
            category = "archive"
        rows.append(f"| `{rel}` | {category} | {format_size(path.stat().st_size)} |")
    return "\n".join(
        [
            "# Docs Index",
            "",
            "| Path | Classification | Size |",
            "|---|---|---|",
            *rows[:650],
            "",
            f"Indexed rows: {len(rows)}" + (" (truncated)" if len(rows) > 650 else ""),
            "",
        ]
    )


def file_purpose_index(repo: Path) -> str:
    purpose = {
        "PROJECT.md": "Project identity, scope, and repo layout.",
        "AGENTS.md": "Stop-the-line rules and agent operating contract.",
        "PROGRESS.md": "Chronological project/ticket progress ledger.",
        "README.md": "User-facing overview, commands, status, and reproducibility notes.",
        "Makefile": "Primary local command entrypoint.",
        "pyproject.toml": "Python package metadata and dependency declarations.",
        "docs/strategy/PLAN_OF_RECORD.md": "AI OS v2 current execution plan.",
        "docs/strategy/CONTEXT_CARRYOVER.md": "Compact current context for fresh model sessions.",
        "project_state/STATE_INDEX.md": "Factual current map of docs, code, outputs, and status.",
        "project_state/VALIDATION_MATRIX.md": "Validation commands mapped to claims.",
        "project_state/CLAIMS_AND_EVIDENCE.md": "Research/public claim surface with evidence and caveats.",
    }
    paths = sorted(set(CANONICAL_DOCS + SELECTED_CONFIGS + SELECTED_SOURCE + SELECTED_TESTS))
    rows = []
    for item in paths:
        path = repo / item
        if not path.exists():
            continue
        importance = "support"
        if item.startswith("src/") or item.startswith("experiments/"):
            importance = "core"
        elif item.startswith("tests/"):
            importance = "validation"
        elif item.startswith("docs/") or item.startswith("project_state/") or item in {"PROJECT.md", "AGENTS.md", "PROGRESS.md", "README.md"}:
            importance = "docs"
        rows.append(
            f"| `{item}` | {path.suffix or 'file'} | {purpose.get(item, 'Selected source/config/test file for strategic review.')} | {importance} | {format_size(path.stat().st_size)} |"
        )
    return "\n".join(
        [
            "# File Purpose Index",
            "",
            "| Path | Type | Purpose | Strategic importance | Size |",
            "|---|---|---|---|---|",
            *rows,
            "",
        ]
    )


def artifact_result_index(repo: Path) -> str:
    interesting_roots = ["reports", "docs/artifacts", "calibration", "data"]
    rows = []
    excluded = []
    for root in interesting_roots:
        base = repo / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = relpath(path, repo)
            size = path.stat().st_size
            if rel.startswith("reports/_bundles/"):
                continue
            if root == "data" and path.suffix.lower() in {".csv", ".parquet"}:
                excluded.append((rel, size, "raw/restricted data excluded from bundles"))
                continue
            if size > 1024 * 1024:
                excluded.append((rel, size, "large generated artifact indexed only"))
                continue
            if path.suffix.lower() in {".md", ".json", ".csv", ".yaml", ".yml", ".txt"}:
                rows.append(
                    f"| `{rel}` | {format_size(size)} | `{sha256_file(path)[:12]}` | artifact/result surface |"
                )
    large_rows = [
        f"| `{rel}` | {format_size(size)} | {reason} |" for rel, size, reason in excluded[:160]
    ]
    return "\n".join(
        [
            "# Artifact Result Index",
            "",
            "Small, strategy-relevant artifact surfaces are listed with short hashes. Raw data and large generated outputs are indexed but not bundled.",
            "",
            "| Artifact | Size | sha256 prefix | Purpose |",
            "|---|---|---|---|",
            *rows[:320],
            "",
            "## Excluded Large/Raw Artifacts",
            "",
            "| Path | Size | Reason excluded |",
            "|---|---|---|",
            *large_rows,
            "",
        ]
    )


def repo_map(repo: Path) -> str:
    entries = []
    for path in sorted(repo.iterdir()):
        if path.name in SKIP_DIRS:
            continue
        entries.append(f"- `{path.name}/`" if path.is_dir() else f"- `{path.name}`")
    return "\n".join(
        [
            "# Repo Map",
            "",
            f"Repo root: `{repo}`",
            "",
            "## Top-Level Layout",
            "",
            *entries,
            "",
            "## Omitted From Bundle",
            "",
            "- Dependency folders, caches, and VCS internals.",
            "- Raw data files under `data/*.csv` and parquet payloads.",
            "- Bulky report trees and old bundles; these are indexed by path/size when relevant.",
            "",
        ]
    )


def validation_baseline(repo: Path) -> str:
    matrix = text_head(repo / "project_state" / "VALIDATION_MATRIX.md", 24000)
    return "\n".join(
        [
            "# Validation Baseline",
            "",
            "This snapshot records what was known when the bundle was built. See the T-000 run log for commands actually executed in this ticket.",
            "",
            matrix or "No `project_state/VALIDATION_MATRIX.md` found.",
            "",
        ]
    )


def recent_progress(repo: Path) -> str:
    return "\n".join(
        [
            "# Recent Progress",
            "",
            "## PROGRESS.md head",
            "",
            text_head(repo / "PROGRESS.md", 24000),
            "",
            "## Current focus",
            "",
            text_head(repo / "docs" / "strategy" / "CONTEXT_CARRYOVER.md", 16000)
            or text_head(repo / "docs" / "NOW.md", 16000),
        ]
    )


def state_summary(repo: Path) -> str:
    return "\n".join(
        [
            "# State Summary",
            "",
            "This is an AI Project OS v2 initial Project State Audit Bundle for GPT 5.5 Pro Extended.",
            "",
            "## Repo Identity",
            "",
            "- Research codebase for FJS/MANOVA-style de-aliasing overlays, covariance forecasting, and portfolio-risk evaluation.",
            "- Python project with synthetic calibration, daily/weekly evaluation runners, reporting tools, and strict audit/run-log expectations.",
            "",
            "## Current Phase",
            "",
            "- Pre-Pro v2 strategy reset after T-000 OS installation.",
            "- Current recovered evidence centers on T-012 daily DoW empirical matrix; T-012 is scientifically useful but not a clean passed-review ticket because monitoring/audit trail preservation failed.",
            "- Weekly/oneway theory path remains blocked by flat-zero injection sensitivity evidence.",
            "",
            "## What Pro Should Decide",
            "",
            "- Whether the next strategic lane is empirical daily DoW scale-up, detector/theory repair, or a bounded advisor-facing empirical package.",
            "- How to treat T-012 recovered outputs in future claims and tickets.",
            "- Which validation gate is sufficient before further expensive run expansion.",
            "",
        ]
    )


def git_state_md(repo: Path) -> str:
    info = git_summary(repo)
    return "\n".join(
        [
            "# Git State",
            "",
            f"- Branch: `{info.get('branch', '')}`",
            f"- HEAD: `{info.get('head', '')}`",
            "",
            "## Status",
            "",
            "```text",
            info.get("status", ""),
            "```",
            "",
            "## Recent Log",
            "",
            "```text",
            info.get("log", ""),
            "```",
            "",
            "## Diff Stat",
            "",
            "```text",
            info.get("diff_stat", ""),
            "```",
            "",
        ]
    )


def manifest_for_stage(stage: Path, profile: str, repo: Path, output_zip: Path) -> dict[str, object]:
    files = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(stage).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    info = git_summary(repo)
    return {
        "profile": profile,
        "created_at_utc": now_iso(),
        "repo": repo.name,
        "repo_root": str(repo),
        "output_zip": str(output_zip),
        "git": {
            "branch": info.get("branch", ""),
            "head": info.get("head", ""),
            "dirty": bool(info.get("status", "").strip()),
        },
        "files": files,
        "file_count": len(files),
    }


def zip_stage(stage: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(stage).as_posix())


def build_project_state_audit(repo: Path, args: argparse.Namespace) -> Path:
    stamp = args.stamp or now_stamp()
    out_zip = Path(args.out) if args.out else repo / "reports" / "_bundles" / f"{stamp}_{repo.name}_project-state_initial.zip"
    with tempfile.TemporaryDirectory(prefix="ai_os_state_") as td:
        stage = Path(td) / "project_state_audit_bundle"
        stage.mkdir(parents=True)

        write_text(stage, "BUNDLE_INDEX.md", "# Bundle Index\n\nProfile: project_state_audit\n\nPrimary consumer: GPT 5.5 Pro Extended.\n\nRead order:\n1. STATE_SUMMARY.md\n2. DOCS_INDEX.md\n3. project_state/STATE_INDEX.md\n4. VALIDATION_BASELINE.md\n5. ARTIFACT_RESULT_INDEX.md\n6. canonical_docs/docs/strategy/CONTEXT_CARRYOVER.md\n")
        write_text(stage, "STATE_SUMMARY.md", state_summary(repo))
        write_text(stage, "repo_map.md", repo_map(repo))
        write_text(stage, "FILE_PURPOSE_INDEX.md", file_purpose_index(repo))
        write_text(stage, "ARTIFACT_RESULT_INDEX.md", artifact_result_index(repo))
        write_text(stage, "VALIDATION_BASELINE.md", validation_baseline(repo))
        write_text(stage, "DOCS_INDEX.md", docs_index(repo))
        write_text(stage, "GIT_STATE.md", git_state_md(repo))
        write_text(stage, "RECENT_PROGRESS.md", recent_progress(repo))

        archive_index = Path(args.archive_index) if args.archive_index else latest_archive_index(repo)
        if archive_index and archive_index.exists():
            shutil.copy2(archive_index, stage / "ARCHIVE_INDEX.md")
        else:
            write_text(stage, "ARCHIVE_INDEX.md", "# Archive Index\n\nNo archive index was found when this bundle was built.\n")

        for rel in CANONICAL_DOCS:
            copy_file(repo, stage, rel, f"canonical_docs/{rel}")
        for rel in SELECTED_SOURCE:
            copy_file(repo, stage, rel, f"selected_source/{rel}")
        for rel in SELECTED_TESTS:
            copy_file(repo, stage, rel, f"selected_tests/{rel}")
        for rel in SELECTED_CONFIGS:
            copy_file(repo, stage, rel, f"selected_configs/{rel}")

        manifest = manifest_for_stage(stage, "project_state_audit", repo, out_zip)
        write_text(stage, "bundle_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        zip_stage(stage, out_zip)
    print(out_zip)
    return out_zip


def status_paths(repo: Path) -> list[str]:
    code, out = run(["git", "status", "--short"], repo)
    if code != 0:
        return []
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        raw = line[3:] if len(line) > 3 else line.strip()
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.strip())
    return paths


def copy_changed_snapshots(repo: Path, stage: Path) -> None:
    for rel in status_paths(repo):
        if not rel or rel.startswith("reports/_bundles/"):
            continue
        path = repo / rel
        if not path.exists() or not path.is_file():
            continue
        if path.stat().st_size > 1024 * 1024:
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        copy_file(repo, stage, rel, f"changed_file_snapshots/{rel}")


def build_review(repo: Path, args: argparse.Namespace) -> Path:
    stamp = args.stamp or now_stamp()
    ticket = args.ticket or "T-000"
    out_zip = Path(args.out) if args.out else repo / "reports" / "_bundles" / f"{stamp}_{repo.name}_review_{ticket}.zip"
    run_log = Path(args.run_log).resolve() if args.run_log else None

    with tempfile.TemporaryDirectory(prefix="ai_os_review_") as td:
        stage = Path(td) / "review_bundle"
        stage.mkdir(parents=True)

        info = git_summary(repo)
        write_text(stage, "BUNDLE_INDEX.md", f"# T-000 Review Bundle\n\nTicket: `{ticket}`\n\nPrimary consumer: Heavy review.\n\nThis bundle reviews the AI Project OS v2 installation, not product behavior.\n")
        write_text(stage, "git_status.txt", info.get("status", "") + "\n")
        write_text(stage, "changed_file_list.txt", "\n".join(status_paths(repo)) + "\n")
        _, diff = run(["git", "diff", "--binary"], repo)
        write_text(stage, "diff.patch", diff)
        _, diff_stat = run(["git", "diff", "--stat"], repo)
        write_text(stage, "diff_stat.txt", diff_stat)

        if run_log and run_log.exists():
            copy_tree(run_log, stage / "run_log")
            for name in ["PROMPT.md", "COMMANDS.md", "RESULTS.md", "VALIDATION.md", "SUMMARY.md"]:
                if (run_log / name).exists():
                    shutil.copy2(run_log / name, stage / name)

        archive_index = Path(args.archive_index) if args.archive_index else latest_archive_index(repo)
        if archive_index and archive_index.exists():
            shutil.copy2(archive_index, stage / "ARCHIVE_INDEX.md")
            manifest = archive_index.with_name("ARCHIVE_MANIFEST.json")
            if manifest.exists():
                shutil.copy2(manifest, stage / "ARCHIVE_MANIFEST.json")

        if args.state_bundle:
            state_bundle = Path(args.state_bundle)
            if state_bundle.exists() and state_bundle.stat().st_size < 20 * 1024 * 1024:
                shutil.copy2(state_bundle, stage / state_bundle.name)

        copy_changed_snapshots(repo, stage)
        manifest = manifest_for_stage(stage, "review", repo, out_zip)
        write_text(stage, "bundle_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        zip_stage(stage, out_zip)
    print(out_zip)
    return out_zip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["project_state_audit", "review"], required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--archive-index", default=None)
    parser.add_argument("--ticket", default="T-000")
    parser.add_argument("--run-log", default=None)
    parser.add_argument("--state-bundle", default=None)
    args = parser.parse_args()

    repo = git_root(Path.cwd())
    if args.profile == "project_state_audit":
        build_project_state_audit(repo, args)
    else:
        build_review(repo, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
