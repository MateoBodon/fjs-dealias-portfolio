import re
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
        ["gpt_bundle.py", "--zip", "--ticket", "FJS-TKT-TEST", "--run-name", "run-1"],
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
