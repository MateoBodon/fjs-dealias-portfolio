from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tools.gpt_bundle import DiffPatchError, resolve_base_ref, write_range_diff


@pytest.mark.unit
def _init_repo(repo: Path, branch: str = "main") -> None:
    init = subprocess.run(
        ["git", "init", "-b", branch],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if init.returncode != 0:
        subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
        subprocess.run(["git", "checkout", "-b", branch], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)


@pytest.mark.unit
def test_write_range_diff_includes_multiple_commits(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, branch="main")

    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)

    subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo, check=True)
    (repo / "one.txt").write_text("change-one\n", encoding="utf-8")
    subprocess.run(["git", "add", "one.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add one"], cwd=repo, check=True)

    (repo / "two.txt").write_text("change-two\n", encoding="utf-8")
    subprocess.run(["git", "add", "two.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "add two"], cwd=repo, check=True)

    output = tmp_path / "DIFF.patch"
    meta = write_range_diff(repo, output, base_ref="main", head_ref="HEAD")

    assert output.exists()
    assert output.stat().st_size > 0
    patch_text = output.read_text(encoding="utf-8", errors="replace")
    assert "one.txt" in patch_text
    assert "two.txt" in patch_text
    assert meta["base_sha"]
    assert meta["head_sha"]


@pytest.mark.unit
def test_resolve_base_ref_requires_override_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo, branch="feature")

    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True)

    monkeypatch.delenv("BUNDLE_BASE", raising=False)

    with pytest.raises(DiffPatchError) as excinfo:
        resolve_base_ref(repo, base_override=None)

    assert "BUNDLE_BASE" in str(excinfo.value)
