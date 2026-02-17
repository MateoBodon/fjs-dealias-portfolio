from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from tools.agentic import validate_runlog


def _write_minimal_run(run_dir: Path, *, commands_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "PROMPT.md").write_text("prompt\n", encoding="utf-8")
    (run_dir / "COMMANDS.md").write_text(commands_text, encoding="utf-8")
    (run_dir / "RESULTS.md").write_text("results\n", encoding="utf-8")
    (run_dir / "TESTS.md").write_text("tests\n", encoding="utf-8")
    (run_dir / "META.json").write_text("{}\n", encoding="utf-8")


@pytest.mark.unit
def test_extract_bundle_stamps_preserves_unique_order() -> None:
    commands_text = """
    BUNDLE_STAMP=20260216_231243 make gpt-bundle TICKET=34 RUN_NAME=run-1
    BUNDLE_STAMP=20260216_231243 make gpt-bundle TICKET=34 RUN_NAME=run-1
    BUNDLE_STAMP=20260216_233000 make gpt-bundle TICKET=34 RUN_NAME=run-1
    """.strip()
    assert validate_runlog._extract_bundle_stamps(commands_text) == [
        "20260216_231243",
        "20260216_233000",
    ]


@pytest.mark.unit
def test_validate_run_dir_requires_final_bundle_stamp_reference_after_cutoff(tmp_path: Path) -> None:
    run_name = "20260216_230858_ticket-34_ingest-project-review-and-fix-meta"
    run_dir = tmp_path / "docs" / "agent_runs" / run_name
    _write_minimal_run(
        run_dir,
        commands_text=(
            "BUNDLE_STAMP=20260216_231243 make gpt-bundle TICKET=34 RUN_NAME="
            f"{run_name}\n"
            "BUNDLE_STAMP=20260216_233000 make gpt-bundle TICKET=34 RUN_NAME="
            f"{run_name}\n"
        ),
    )
    progress_text = (
        "Bundle: "
        "`artifacts/_local/gpt_bundles/20260216_231243_34_"
        "20260216_230858_ticket-34_ingest-project-review-and-fix-meta.zip`\n"
    )
    issues = validate_runlog._validate_run_dir(
        run_dir,
        progress_text=progress_text,
        require_meta_json=True,
        bundle_stamp_provenance_cutoff="20260216_000000",
    )
    assert any("final BUNDLE_STAMP provenance mismatch" in issue for issue in issues)


@pytest.mark.unit
def test_validate_run_dir_accepts_final_bundle_stamp_reference_after_cutoff(tmp_path: Path) -> None:
    run_name = "20260216_230858_ticket-34_ingest-project-review-and-fix-meta"
    run_dir = tmp_path / "docs" / "agent_runs" / run_name
    _write_minimal_run(
        run_dir,
        commands_text=(
            "BUNDLE_STAMP=20260216_231243 make gpt-bundle TICKET=34 RUN_NAME="
            f"{run_name}\n"
            "BUNDLE_STAMP=20260216_233000 make gpt-bundle TICKET=34 RUN_NAME="
            f"{run_name}\n"
        ),
    )
    progress_text = (
        "Bundle: "
        "`artifacts/_local/gpt_bundles/20260216_233000_34_"
        "20260216_230858_ticket-34_ingest-project-review-and-fix-meta.zip`\n"
    )
    issues = validate_runlog._validate_run_dir(
        run_dir,
        progress_text=progress_text,
        require_meta_json=True,
        bundle_stamp_provenance_cutoff="20260216_000000",
    )
    assert not issues


@pytest.mark.unit
def test_validate_run_dir_skips_bundle_stamp_enforcement_before_cutoff(tmp_path: Path) -> None:
    run_name = "20251226_105628_ticket-25_week-between-stress"
    run_dir = tmp_path / "docs" / "agent_runs" / run_name
    _write_minimal_run(
        run_dir,
        commands_text=(
            "BUNDLE_STAMP=20251226_110750 make gpt-bundle TICKET=25 RUN_NAME="
            f"{run_name}\n"
            "BUNDLE_STAMP=20251226_111227 make gpt-bundle TICKET=25 RUN_NAME="
            f"{run_name}\n"
        ),
    )
    issues = validate_runlog._validate_run_dir(
        run_dir,
        progress_text="",
        require_meta_json=True,
        bundle_stamp_provenance_cutoff="20260216_000000",
    )
    assert not issues
