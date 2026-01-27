import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}

pytestmark = pytest.mark.unit


def _is_residue(name: str) -> bool:
    return ".bak." in name or ".append" in name


def _find_residue(root: Path) -> list[Path]:
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in filenames:
            if _is_residue(filename):
                matches.append(Path(dirpath) / filename)
    return sorted(matches)


def test_repo_has_no_backup_residue() -> None:
    matches = _find_residue(ROOT)
    assert not matches, (
        "Found backup residue files (remove *.bak.* / *.append*):\n"
        + "\n".join(str(path.relative_to(ROOT)) for path in matches)
    )


def test_gitignore_blocks_reports_runs() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert any(
        line.strip() in {"reports/_runs/", "reports/_runs/**"}
        for line in content.splitlines()
    ), "Expected .gitignore to ignore reports/_runs/ (tracking policy requirement)."
