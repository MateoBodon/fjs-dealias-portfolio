import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tools.agentic import gpt_bundle as agentic_bundle

REQUIRED_PATHS = [
    "AGENTS.md",
    "docs/PLAN_OF_RECORD.md",
    "docs/DOCS_AND_LOGGING_SYSTEM.md",
    "docs/CODEX_SPRINT_TICKETS.md",
    "project_state/CURRENT_RESULTS.md",
    "project_state/KNOWN_ISSUES.md",
    "project_state/CONFIG_REFERENCE.md",
    "PROGRESS.md",
]

REQUIRED_SNIPPETS = [
    "tools/gpt_bundle.py diff",
    "DIFF.patch",
    "LAST_COMMIT.txt",
    "BUNDLE_META.md",
    "artifacts/_local/gpt_bundles",
]


@pytest.mark.unit
def test_makefile_has_gpt_bundle_target_and_inputs():
    content = (Path(__file__).resolve().parents[1] / "Makefile").read_text()
    assert re.search(r"^gpt-bundle\s*:", content, re.MULTILINE), (
        "Expected gpt-bundle target in Makefile text."
    )
    for required in REQUIRED_PATHS:
        assert required in content, f"gpt-bundle recipe missing required path: {required}"
    for snippet in REQUIRED_SNIPPETS:
        assert snippet in content, f"gpt-bundle recipe missing required snippet: {snippet}"


@pytest.mark.unit
def test_agentic_bundle_fails_on_dirty_repo(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(agentic_bundle, "get_git_status_porcelain", lambda _: " M README.md\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpt_bundle.py", "--zip", "--ticket", "FJS-TKT-TEST", "--run-name", "run-1", "--no-stash"],
    )
    exit_code = agentic_bundle.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "dirty" in captured.err.lower()


@pytest.mark.unit
def test_bundle_meta_git_dirty_field_added(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        bundle.writestr("BUNDLE_META.md", "run_name: demo\n")
        bundle.writestr("OTHER.txt", "data\n")

    agentic_bundle.update_bundle_meta_zip(bundle_path, git_dirty=False)

    with zipfile.ZipFile(bundle_path, "r") as bundle:
        meta = bundle.read("BUNDLE_META.md").decode("utf-8", errors="replace")
        other = bundle.read("OTHER.txt").decode("utf-8", errors="replace")

    assert "git_dirty: false" in meta
    assert other == "data\n"


@pytest.mark.unit
def test_agentic_bundle_stashes_and_targets_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    statuses = iter([" M README.md\n", "", " M README.md\n"])
    monkeypatch.setattr(agentic_bundle, "get_git_status_porcelain", lambda _: next(statuses))
    calls: dict[str, object] = {"push": 0, "apply": 0, "drop": 0, "dirty": None, "bundle": None}

    def fake_stash_push(repo: Path, message: str) -> str:
        calls["push"] = int(calls["push"]) + 1
        return "stash@{0}"

    def fake_stash_apply(repo: Path, stash_ref: str) -> None:
        calls["apply"] = int(calls["apply"]) + 1

    def fake_stash_drop(repo: Path, stash_ref: str) -> None:
        calls["drop"] = int(calls["drop"]) + 1

    def fake_run_make(repo: Path, ticket: str, run_name: str) -> subprocess.CompletedProcess[str]:
        bundle_dir = tmp_path / "artifacts" / "_local" / "gpt_bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w") as bundle:
            bundle.writestr("BUNDLE_META.md", "run_name: demo\n")
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=str(bundle_path) + "\n",
            stderr="",
        )

    def fake_update(path: Path, git_dirty: bool) -> None:
        calls["bundle"] = Path(path)
        calls["dirty"] = git_dirty

    monkeypatch.setattr(agentic_bundle, "_stash_push", fake_stash_push)
    monkeypatch.setattr(agentic_bundle, "_stash_apply", fake_stash_apply)
    monkeypatch.setattr(agentic_bundle, "_stash_drop", fake_stash_drop)
    monkeypatch.setattr(agentic_bundle, "_run_make_bundle", fake_run_make)
    monkeypatch.setattr(agentic_bundle, "update_bundle_meta_zip", fake_update)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpt_bundle.py", "--zip", "--ticket", "TICKET-1", "--run-name", "run-1"],
    )

    exit_code = agentic_bundle.main()

    assert exit_code == 0
    assert calls["push"] == 1
    assert calls["apply"] == 1
    assert calls["drop"] == 1
    assert calls["dirty"] is True
    assert calls["bundle"] is not None
    assert "artifacts/_local/gpt_bundles" in str(calls["bundle"])
