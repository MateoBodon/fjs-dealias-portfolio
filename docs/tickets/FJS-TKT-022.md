# FJS-TKT-022

## Goal
De-scope and harden FJS-TKT-019: keep only the weekly synthetic smoke/diagnostics fix, restore repo process docs, and add regression assertions that the tiny weekly smoke produces evaluated windows + required artifacts.

## Scope
- Revert unrelated edits to `AGENTS.md`, `PROGRESS.md`, `docs/PLAN_OF_RECORD.md`, `docs/CODEX_SPRINT_TICKETS.md`.
- Delete accidental untracked artifacts/backups (`.bak.*`, `.gitignore.append`, bootstrap scaffolding) unless explicitly required.
- Keep `experiments/equity_panel/run.py` + tests focused on weekly smoke diagnostics.
- Strengthen tests to assert >=1 row in `gating_diagnostics.csv`, non-empty `detection_summary.csv` (if produced), and no `skip_reason_primary==DIAGNOSTIC_FAILURE`.
- Ensure `weekly_diagnostics.md` is always written or fail-loud.

## Acceptance Criteria
- Diff limited to intended weekly-smoke files + necessary test/assert changes.
- Git status clean (no stray backups/scaffold).
- `make run:equity_nested_smoke_tiny` produces >=1 evaluated window (`gating_diagnostics.csv` has >=1 row) and non-empty `detection_summary.csv`.
- No `skip_reason_primary==DIAGNOSTIC_FAILURE`.
- `weekly_diagnostics.md` present and contains summary sections.

## Plan
1. Revert unrelated doc edits and remove untracked backup/scaffold files.
2. Update `experiments/equity_panel/run.py` to fail-loud on missing weekly diagnostics and adjust `tests/experiments/test_gating_diagnostics.py` assertions.
3. Run `make test-fast` and `make run:equity_nested_smoke_tiny`.
4. Update `PROGRESS.md` and generate `gpt_bundle.zip` for the ticket.

## Notes
- Canonical tests: `make test-fast`.
- Smoke command: `make run:equity_nested_smoke_tiny`.
