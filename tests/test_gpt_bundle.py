import re
from pathlib import Path

import pytest

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
