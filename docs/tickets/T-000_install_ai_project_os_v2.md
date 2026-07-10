# T-000 - Install AI Project OS v2

status: review-ready after validation
owner_flow: Pro planned -> Heavy dispatched -> Codex executes -> Heavy reviews -> Pro recenter if needed
created: 2026-07-03
updated: 2026-07-03

## Goal

Install AI Project OS v2 for this repository, preserve pre-v2 documentation in an accessible archive, create clean canonical docs, and generate the first Pro-facing Project State Audit Bundle plus a Heavy review bundle.

## In Scope

- Search and classify existing docs, run logs, prompts, tickets, state docs, reports, and generated notes.
- Copy/index pre-v2 docs under `docs/_archive/pre_ai_os_v2/20260703/`.
- Create the v2 canonical docs requested by the work order.
- Add lightweight bundle tooling for state-audit and review profiles.
- Generate T-000 run log, archive manifest, Project State Audit Bundle, and T-000 Review Bundle.
- Run safe repo-appropriate validation.

## Out Of Scope

- Product/research behavior changes.
- Large experiment reruns.
- Deleting or moving old docs.
- Treating recovered or archived evidence as newly validated.

## Context Files

- `PROJECT.md`
- `AGENTS.md`
- `PROGRESS.md`
- `docs/strategy/CONTEXT_CARRYOVER.md`
- `project_state/STATE_INDEX.md`
- `project_state/VALIDATION_MATRIX.md`
- `project_state/CLAIMS_AND_EVIDENCE.md`
- `docs/_archive/pre_ai_os_v2/20260703/ARCHIVE_INDEX.md`

## Acceptance Criteria

- Canonical docs exist and are honest about pre-Pro gaps.
- Old docs are preserved, classified, and indexed.
- Project State Audit Bundle exists under `reports/_bundles/`.
- T-000 Review Bundle exists under `reports/_bundles/`.
- Run log exists under `reports/_runs/20260703_132437_T-000_install_ai_project_os_v2/`.
- Validation commands and outcomes are recorded exactly.

## Validation Level

L1 targeted plus the repo's fast unit gate if available.

Expected commands:

```bash
git status --short
python3 -m json.tool docs/_archive/pre_ai_os_v2/20260703/ARCHIVE_MANIFEST.json >/dev/null
python3 tools/agentic/ai_os_bundle.py --profile project_state_audit --stamp 20260703_132437
python3 tools/agentic/ai_os_bundle.py --profile review --ticket T-000 --run-log reports/_runs/20260703_132437_T-000_install_ai_project_os_v2 --state-bundle reports/_bundles/20260703_132437_repo_project-state_initial.zip --stamp 20260703_132437
. .venv/bin/activate && pytest -q tests/test_ai_os_bundle.py tests/test_gpt_bundle.py
. .venv/bin/activate && make test-fast
```

## Stop-The-Line Conditions

- Old docs deleted or hidden.
- Raw data included in the state bundle.
- Unsupported claims strengthened.
- Validation failures omitted or misrepresented.
- Review bundle missing command log, archive index, or changed file evidence.
