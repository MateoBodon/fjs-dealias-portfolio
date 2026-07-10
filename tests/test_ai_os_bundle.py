from pathlib import Path

import pytest

from tools.agentic import ai_os_bundle


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_makefile_exposes_ai_os_bundle_targets() -> None:
    content = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "project-state-audit-bundle" in content
    assert "ai-os-review-bundle" in content
    assert "tools/agentic/ai_os_bundle.py --profile project_state_audit" in content
    assert "tools/agentic/ai_os_bundle.py --profile review" in content


@pytest.mark.unit
def test_ai_os_bundle_selected_docs_are_stable() -> None:
    assert "docs/strategy/CONTEXT_CARRYOVER.md" in ai_os_bundle.CANONICAL_DOCS
    assert "project_state/VALIDATION_MATRIX.md" in ai_os_bundle.CANONICAL_DOCS
    assert "tools/agentic/ai_os_bundle.py" in ai_os_bundle.SELECTED_SOURCE


@pytest.mark.unit
def test_format_size_boundaries() -> None:
    assert ai_os_bundle.format_size(0) == "0 B"
    assert ai_os_bundle.format_size(1023) == "1023 B"
    assert ai_os_bundle.format_size(1024) == "1.0 KB"
